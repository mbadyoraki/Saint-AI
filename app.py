# app.py - NexusAI: News Analyst & Fact-Checker Backend (Using Google Gemini Flash or Open-Source LLM)
# This monolithic file combines all discussed features for demonstration.
# In a real-world scenario, this would be broken down into multiple modules (blueprints, services, models).

import os
import re
import time
import random
import logging
import json
import html # Used for server-side HTML escaping/sanitization
import smtplib # For email sending (password reset, verification)
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from uuid import uuid4 # For generating unique tokens

# For web search capability (pip install duckduckgo_search)
from duckduckgo_search import DDGS
# For HTML parsing (pip install beautifulsoup4)
from bs4 import BeautifulSoup # <-- Ensure BeautifulSoup is globally available

# For making HTTP requests to external APIs
import requests

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
from dotenv import load_dotenv
load_dotenv() 
from werkzeug.security import generate_password_hash, check_password_hash

# SQLAlchemy setup (pip install Flask-SQLAlchemy)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship # Explicitly import relationship for clarity

# Google Generative AI for LLM integration (pip install google-generativeai)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings

# For Google OAuth 2.0 ID token verification (pip install google-auth)
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# For Open-Source LLM integration (pip install transformers torch accelerate sentencepiece)
# Only import if you intend to use open-source models and have these installed
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    logging.warning("Transformers library not found. Open-source LLM integration will be disabled. Install with `pip install transformers torch accelerate sentencepiece` if needed.")
    HAS_TRANSFORMERS = False

# Flask extensions for rate limiting and caching (pip install Flask-Limiter Flask-Caching)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache

# For real-time communication (pip install Flask-SocketIO)
from flask_socketio import SocketIO, emit

# --- Flask App Initialization and Configuration ---
app = Flask(__name__) # Corrected from __init__ to __name__
load_dotenv() # Load environment variables from .env file

# Core App Configurations
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_and_complex_key_for_prod_environment')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress warning
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

# Email Configurations (for password reset and email verification)
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() in ('true', '1', 't')
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# --- News API Keys (Expanded List) ---
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY')
NYT_API_KEY = os.getenv('NYT_API_KEY')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY')
MEDIASTACK_API_KEY = os.getenv('MEDIASTACK_API_KEY')

# NEW API Keys
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
BING_NEWS_API_KEY = os.getenv('BING_NEWS_API_KEY') # For Azure Cognitive Services News Search
NEWSCASTER_API_KEY = os.getenv('NEWSCASTER_API_KEY') # Renamed from NewsCatcher due to common typo/similar service
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY') # For Google News searches via SerpAPI

# Conceptual/Enterprise API Keys (for demonstration of integration pattern)
AP_NEWS_API_KEY = os.getenv('AP_NEWS_API_KEY') # Associated Press API
FINANCIAL_TIMES_API_KEY = os.getenv('FINANCIAL_TIMES_API_KEY') # Financial Times API


# --- Google OAuth 2.0 Client ID ---
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')


db = SQLAlchemy(app)

# --- Configure Flask-Limiter for rate limiting ---
limiter = Limiter(
    app=app,
    key_func=get_remote_address, # Identify users by IP address for rate limiting
    default_limits=["200 per day", "50 per hour"], # Default limits for all routes
)

# --- Configure Flask-Caching for data caching ---
# Using 'simple' cache type for monolithic demo. In production, use 'redis' or 'memcached'.
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# --- Configure Flask-SocketIO for real-time communication ---
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=None, logger=True, engineio_logger=True)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO if not app.config['DEBUG'] else logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"), # Log to a file named app.log
                        logging.StreamHandler()        # Log to the console
                    ])

# --- Configure Google Generative AI (LLM) and Open-Source LLM ---
llm_model = None # For Gemini
llm_pipeline = None # For Hugging Face models

# Attempt to configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel(
            'gemini-1.5-flash', # Or 'gemini-1.0-pro', 'gemini-1.5-pro' if preferred/available
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        logging.info("Google Generative AI (Gemini 1.5 Flash) initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Google Generative AI: {e}. AI responses will be limited to rule-based summaries.")
        llm_model = None
else:
    logging.warning("GEMINI_API_KEY environment variable not found. Gemini AI responses will be limited to rule-based summaries.")
    llm_model = None

# Attempt to configure Open-Source LLM (prefer this if HAS_TRANSFORMERS and configured)
OPEN_SOURCE_MODEL_NAME = os.getenv("OPEN_SOURCE_MODEL_NAME", None)
if HAS_TRANSFORMERS and OPEN_SOURCE_MODEL_NAME:
    try:
        logging.info(f"Attempting to load open-source model: {OPEN_SOURCE_MODEL_NAME}")
        # Use 'cuda' if available, otherwise 'cpu'
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(OPEN_SOURCE_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            OPEN_SOURCE_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None, # Use bfloat16 for efficiency on supported GPUs
            device_map="auto" # Automatically maps model to available devices (GPU/CPU)
        )
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=1024, # Adjust as needed for response length
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            num_return_sequences=1
        )
        logging.info(f"Open-source LLM '{OPEN_SOURCE_MODEL_NAME}' initialized successfully on device: {device}.")
    except Exception as e:
        logging.error(f"Error initializing open-source LLM '{OPEN_SOURCE_MODEL_NAME}': {e}. AI responses will be limited to rule-based summaries.", exc_info=True)
        llm_pipeline = None
else:
    if HAS_TRANSFORMERS: # Only warn if transformers is installed but model not set
        logging.warning("OPEN_SOURCE_MODEL_NAME environment variable not found. Open-source LLM responses will be limited to rule-based summaries.")
    llm_pipeline = None


# --- User Role Definitions ---
class UserRole:
    GUEST = 'guest'
    STANDARD = 'standard'
    PREMIUM = 'premium'
    ADMIN = 'admin'

# --- Database Models (Copied from updated models.py to make app.py self-contained for demonstration) ---
# In a real-world scenario, these would be in a separate `models.py` file.
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    avatar_url = db.Column(db.String(255), default='/static/avatars/default.png')
    theme = db.Column(db.String(50), default='light') # User's preferred theme
    email_verified = db.Column(db.Boolean, default=False) # Email verification status
    email_verification_token = db.Column(db.String(100), unique=True, nullable=True) # Token for email verification
    password_reset_token = db.Column(db.String(100), unique=True, nullable=True) # Token for password reset
    password_reset_expires_at = db.Column(db.DateTime, nullable=True) # Expiry for password reset token
    role = db.Column(db.String(50), default=UserRole.STANDARD) # User's role

    current_conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=True)
    
    conversations = relationship('Conversation', backref='user', lazy=True, cascade="all, delete-orphan",
                                 foreign_keys='Conversation.user_id')

    # New fields for personalization preferences
    _preferred_topics = db.Column(db.Text, default='[]') # Stores JSON string of a list of topics
    _preferred_sources = db.Column(db.Text, default='[]') # Stores JSON string of a list of sources
    analysis_depth = db.Column(db.String(50), default='standard') # e.g., 'brief', 'standard', 'detailed'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def generate_email_verification_token(self):
        self.email_verification_token = str(uuid4())
        return self.email_verification_token

    def generate_password_reset_token(self):
        self.password_reset_token = str(uuid4())
        self.password_reset_expires_at = datetime.utcnow() + timedelta(hours=1) # Token expires in 1 hour
        return self.password_reset_token

    # Properties for preferred_topics to handle JSON serialization/deserialization
    @property
    def preferred_topics(self):
        if self._preferred_topics:
            try:
                return json.loads(self._preferred_topics)
            except json.JSONDecodeError:
                logging.error(f"Failed to decode preferred_topics JSON: {self._preferred_topics}")
                return []
        return []

    @preferred_topics.setter
    def preferred_topics(self, value):
        self._preferred_topics = json.dumps(value)

    # Properties for preferred_sources to handle JSON serialization/deserialization
    @property
    def preferred_sources(self):
        if self._preferred_sources:
            try:
                return json.loads(self._preferred_sources)
            except json.JSONDecodeError:
                logging.error(f"Failed to decode preferred_sources JSON: {self._preferred_sources}")
                return []
        return []

    @preferred_sources.setter
    def preferred_sources(self, value):
        if isinstance(value, list):
            self._preferred_sources = json.dumps(value)
        else:
            self._preferred_sources = json.dumps([]) # Ensure it's always a JSON array string

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'avatar_url': self.avatar_url,
            'theme': self.theme,
            'current_conversation_id': self.current_conversation_id,
            'email_verified': self.email_verified,
            'role': self.role,
            'preferred_topics': self.preferred_topics,
            'preferred_sources': self.preferred_sources,
            'analysis_depth': self.analysis_depth
        }

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False, default='New Chat')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow) # Track last update for sorting
    tags = db.Column(db.Text, nullable=True) # Store tags as JSON string or comma-separated

    messages = relationship('Message', backref='conversation', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'timestamp': self.timestamp.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'tags': json.loads(self.tags) if self.tags else []
        }

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sender = db.Column(db.String(50), nullable=False) # 'user' or 'ai'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text, nullable=True) # Added details field for AI metadata
    _sources = db.Column(db.Text, nullable=True) # Store sources as JSON string
    sentiment = db.Column(db.String(50), nullable=True) # e.g., 'Positive', 'Neutral', 'Negative'
    bias = db.Column(db.String(50), nullable=True) # e.g., 'Left', 'Right', 'Neutral', 'Mixed/Slight Bias'

    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'sender': self.sender,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'sources': self.sources, # Access via property for deserialization
            'sentiment': self.sentiment,
            'bias': self.bias
        }

    @property
    def sources(self):
        if self._sources:
            try:
                return json.loads(self._sources)
            except json.JSONDecodeError:
                logging.error(f"Failed to decode sources JSON: {self._sources}")
                return []
        return []

    @sources.setter
    def sources(self, value):
        # Ensure value is a list before dumping to JSON
        if isinstance(value, list):
            self._sources = json.dumps(value)
        else:
            self._sources = json.dumps([]) # Store an empty JSON array if not a list

# --- Database Initialization & Dummy Data Creation ---
def init_db():
    """Initializes the database and creates a default admin user if none exist."""
    with app.app_context():
        db.create_all()
        logging.info("Database tables created/checked.")

        if User.query.count() == 0:
            admin_user = User(username='admin', email='admin@nexusai.com', avatar_url='/static/avatars/avatar1.png', role=UserRole.ADMIN, email_verified=True)
            admin_user.set_password('adminpassword') # IMPORTANT: Change this in production!
            db.session.add(admin_user)
            db.session.commit()
            logging.info("Default admin user 'admin' created with password 'adminpassword'. PLEASE CHANGE THIS!")

# --- Avatar Generation (for demonstration/initial setup) ---
AVATAR_OPTIONS = [
    "/static/avatars/default.png",
    "/static/avatars/avatar1.png",
    "/static/avatars/avatar2.png",
    "/static/avatars/avatar3.png",
    "/static/avatars/avatar4.png",
    "/static/avatars/nexusai_avatar.png" # AI's avatar
]

def create_dummy_avatars():
    """Ensures dummy avatar images exist in the static/avatars directory."""
    avatars_dir = os.path.join(app.root_path, 'static', 'avatars')
    os.makedirs(avatars_dir, exist_ok=True)
    logging.info(f"Ensured directory exists: {avatars_dir}")

    try:
        from PIL import Image, ImageDraw, ImageFont

        nexusai_avatar_path = os.path.join(avatars_dir, 'nexusai_avatar.png')
        if not os.path.exists(nexusai_avatar_path):
            img = Image.new('RGB', (60, 60), color=(162, 210, 255)) # Light blue background
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 36) # Path to a .ttf font file
            except IOError:
                logging.warning("Arial font not found for NexusAI avatar, using default.")
                font = ImageFont.load_default()
            d.text((15, 10), "AI", fill=(255, 255, 255), font=font)
            img.save(nexusai_avatar_path)
            logging.info(f"Created NexusAI dummy avatar: {nexusai_avatar_path}")

        user_avatar_filenames = ["default.png", "avatar1.png", "avatar2.png", "avatar3.png", "avatar4.png"]
        colors = [(205, 180, 219), (255, 200, 221), (188, 230, 255), (200, 230, 255), (144, 238, 144), (255, 160, 122)]
        for i, filename in enumerate(user_avatar_filenames):
            filepath = os.path.join(avatars_dir, filename)
            if not os.path.exists(filepath):
                color = colors[i % len(colors)]
                img = Image.new('RGB', (60, 60), color=color)
                d = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except IOError:
                    font = ImageFont.load_default()
                text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                d.text((15, 10), str(i + 1), fill=text_color, font=font)
                img.save(filepath)
                logging.info(f"Created dummy avatar: {filepath}")
    except ImportError:
        logging.warning("Pillow not installed. Cannot create dummy avatar images. Please install Pillow (`pip install Pillow`) or create images manually in 'static/avatars/'.")
    except Exception as e:
        logging.error(f"Error creating dummy avatars: {e}. Please ensure 'static/avatars/' is writable.")

# --- Before Request Hook for User Session Handling ---
@app.before_request
def load_logged_in_user():
    """Loads the user from session into Flask's `g` object for request-wide access."""
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
        if 'unauth_prompt_count' not in session:
            session['unauth_prompt_count'] = 0
        if 'unauth_conversation_messages' not in session:
            session['unauth_conversation_messages'] = []
        logging.debug("Before request: No user_id in session. User is guest.")
    else:
        g.user = User.query.get(user_id)
        if g.user:
            logging.debug(f"Before request: User {g.user.username} (ID: {user_id}, Role: {g.user.role}) loaded from session.")
        else:
            logging.warning(f"Before request: User ID {user_id} in session but not found in DB. Clearing invalid session.")
            session.pop('user_id', None)
            g.user = None
            session['unauth_prompt_count'] = 0
            session['unauth_conversation_messages'] = []

    session.modified = True # Ensure session is marked as modified if its contents change

# --- User Role Management Decorator ---
def role_required(required_role):
    """
    Decorator to restrict access to routes based on user roles.
    Uses a predefined role hierarchy.
    """
    def decorator(f):
        @limiter.exempt # Exempt role check from general rate limits, as auth is checked first
        def wrapper(*args, **kwargs):
            if not g.user:
                logging.warning(f"Unauthorized access attempt to {request.path}: No user logged in.")
                return jsonify({"message": "Authentication required."}), 401
            
            # Define a hierarchy for roles
            roles_order = {
                UserRole.GUEST: 0,
                UserRole.STANDARD: 1,
                UserRole.PREMIUM: 2,
                UserRole.ADMIN: 3
            }

            user_role_level = roles_order.get(g.user.role, 0)
            required_role_level = roles_order.get(required_role, 0)

            if user_role_level < required_role_level:
                logging.warning(f"Forbidden access attempt to {request.path}: User {g.user.username} (Role: {g.user.role}) lacks required role {required_role}.")
                return jsonify({"message": f"Access forbidden. Requires '{required_role}' role."}), 403
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__ # Preserve original function name for Flask
        return wrapper
    return decorator


# --- Email Utility Functions (Simulated Email Service) ---
# In a real-world scenario, this would be a dedicated email service or module.
def send_email(to_email, subject, body):
    """Sends an email using configured SMTP settings."""
    if not all([app.config['MAIL_SERVER'], app.config['MAIL_PORT'], app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'], app.config['MAIL_DEFAULT_SENDER']]):
        logging.error("Email configuration missing. Cannot send email.")
        return False

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = app.config['MAIL_DEFAULT_SENDER']
    msg['To'] = to_email

    try:
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.starttls() # Enable TLS encryption
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)
        logging.info(f"Email sent to {to_email}.")
        return True
    except Exception as e:
        logging.error(f"Failed to send email to {to_email}: {e}", exc_info=True)
        return False

# --- Web Scraping and Search Functions (Data Ingestion Service) ---
# In a real-world scenario, this would be a dedicated web scraping service or a microservice.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1"
]

@cache.memoize(timeout=3600) # Cache fetched page content for 1 hour
def get_page_content(url, max_retries=2):
    """Fetches the HTML content of a given URL, with retries and a random delay."""
    try:
        import requests
    except ImportError:
        logging.error("Requests library not installed. Cannot perform web scraping.")
        return None

    headers = {'User-Agent': random.choice(USER_AGENTS)}
    parsed_url = urlparse(url)

    # Basic robots.txt check (not full compliance, but avoids common traps)
    if "robots.txt" in parsed_url.path.lower() or "sitemap" in parsed_url.path.lower():
        logging.warning(f"Attempted to scrape robots.txt/sitemap, skipping: {url}")
        return None

    for attempt in range(max_retries):
        try:
            delay = random.uniform(1, 3) # Random delay to be polite
            time.sleep(delay)
            response = requests.get(url, headers=headers, timeout=15) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logging.info(f"Successfully fetched {url} (Attempt {attempt + 1})")
            return response.text
        except requests.exceptions.RequestException as e:
            logging.warning(f"Error fetching {url} (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying {url}...")
            else:
                logging.error(f"Failed to fetch {url} after {max_retries} attempts.")
                return None

def extract_main_content(html_content):
    """
    Extracts the main textual content from an HTML string using BeautifulSoup,
    removing boilerplate elements.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove common boilerplate elements
    for script_or_style in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'aside', 'form',
                                 '.sidebar', '.ad', '#comments', '.meta', '.share', '.byline', '.social-links',
                                 '.promo', '.newsletter-signup', 'iframe', 'svg', 'button', 'figcaption',
                                 '.breadcrumb', '.page-header', '.banner', '.cookie-consent', '.gdpr-banner',
                                 '.modal', '.overlay', '.popup', '[role="navigation"]', '[role="complementary"]',
                                 '[role="banner"]', '[role="contentinfo"]']):
        script_or_style.decompose()

    # Prioritized selectors for main content areas
    content_containers = [
        'article',
        'main',
        'div[itemprop*="articleBody"]',
        'div[class*="content-body"]',
        'div[class*="entry-content"]',
        'div[class*="article-body"]',
        'div[class*="post-content"]',
        'div[id*="content"]',
        'div[id*="article"]',
        'section[role="main"]',
        'div[role="main"]',
        'div.story-body',
        'div.StandardArticleBody_body',
        'div.article__content',
        'div.body-content',
        'div.td-post-content',
        '.article-text',
        '.post-area',
        '.single-post-content'
    ]

    main_text_parts = []
    for selector in content_containers:
        elements = soup.select(selector)
        if elements:
            for element in elements:
                text = element.get_text(separator='\n', strip=True)
                # Filter out very short lines and excessive whitespace
                cleaned_text = '\n'.join([line for line in text.splitlines() if len(line.strip()) > 30])
                # Remove leading/trailing non-alphanumeric characters or excessive symbols
                cleaned_text = re.sub(r'[\s\W_]+$', '', cleaned_text)
                cleaned_text = re.sub(r'^\s*[\W_]+', '', cleaned_text)
                
                # Exclude boilerplate phrases that might remain
                if len(cleaned_text) > 150 and not re.search(r'next\s*←\s*return|privacy\s*policy|terms\s*of\s*use|copyright|all\s*rights\s*reserved|skip\s*to\s*content|read\s*more|subscribe|newsletter|contact\s*us|advertisement|by\s+:\s+\w+\s+\w+|comments|topics|photo\s*by|get\s*the\s*app|recommended\s*articles|related\s*stories', cleaned_text, re.IGNORECASE):
                    main_text_parts.append(cleaned_text)
            if main_text_parts: # If content found with current selector, stop trying other selectors
                break

    # Fallback: if no specific content container yields good results, try to extract from paragraphs in body
    if not main_text_parts and soup.body:
        paragraphs = soup.find_all('p')
        text_content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        cleaned_text = re.sub(r'[\s\W_]+$', '', text_content)
        cleaned_text = re.sub(r'^\s*[\W_]+', '', cleaned_text)
        if len(cleaned_text) > 150:
            main_text_parts.append(cleaned_text)

    full_text = "\n\n".join(main_text_parts)
    full_text = re.sub(r'\s+', ' ', full_text).strip() # Replace multiple spaces with single space
    full_text = re.sub(r'(\n\s*){3,}', '\n\n', full_text).strip() # Reduce excessive newlines

    return full_text[:8000] # Increased content limit for more comprehensive analysis


def perform_web_search_and_scrape(query, num_results=5, preferred_sources=None, preferred_topics=None, region_bias=None):
    """
    Performs a general web search using DuckDuckGo Search and scrapes content.
    Prioritizes results from `preferred_sources` and incorporates `preferred_topics`.
    `region_bias` can be a country code or 'africa' to bias the search results.
    """
    effective_query = query
    if preferred_topics:
        effective_query += " " + " ".join(preferred_topics)

    logging.info(f"WEB_SCRAPE: Performing general web search for query: '{effective_query}' with preferred sources: {preferred_sources} and region bias: {region_bias}...")
    scraped_sources = []
    unique_urls = set()

    try:
        # Determine DuckDuckGo region. 'wt-wt' is worldwide.
        # We can try to use a specific region if region_bias is set to a country code.
        # DDGS doesn't have a direct 'africa' region, so we stick to wt-wt if it's 'africa'.
        search_region = region_bias if region_bias and region_bias != 'africa' else 'wt-wt'

        search_results = DDGS().text(keywords=effective_query, region=search_region, max_results=num_results + 10) 
        
        articles_to_scrape = []
        EXCLUDE_DOMAINS = [
            'youtube.com', 'twitter.com', 'facebook.com', 'reddit.com', 'wikipedia.org',
            'google.com', 'amazon.com', 'shop.com', 'cnet.com', 'news.google.com',
            'bloomberg.com/quote', 'pinterest.com', 'instagram.com', 'tiktok.com',
            'linkedin.com', 'apple.com', 'microsoft.com', 'docs.google.com',
            'support.google.com', 'developer.mozilla.org', 'w3schools.com',
            'stackoverflow.com', 'github.com', 'medium.com', 'quora.com', 'answers.com',
            'yelp.com', 'tripadvisor.com'
        ]

        # Prioritize preferred sources in the search results before scraping
        prioritized_results = []
        other_results = []

        # If region_bias is 'africa', also bias towards common African news domains in the scraping phase
        african_domain_keywords = [
            'africanews.com', 'allafrica.com', 'news24.com', 'nation.africa', 'vanguardngr.com',
            'bbc.com/news/africa', 'reuters.com/africa', 'millardayo.com', 'citizentv.co.ke',
            'dailymaverick.co.za', 'theafricareport.com'
        ]
        
        # Combine preferred_sources and african_domain_keywords if region_bias is 'africa'
        combined_prioritization_keywords = []
        if preferred_sources:
            combined_prioritization_keywords.extend([
                re.sub(r'^(www\.)?|\.(com|org|net|co\.tz|go\.tz|io|info|me|tv|co\.za|co\.ke|ng)$', '', urlparse(s).netloc.lower() if s.startswith('http') else s.lower())
                for s in preferred_sources
            ])
        if region_bias == 'africa':
            combined_prioritization_keywords.extend([
                re.sub(r'^(www\.)?|\.(com|org|net|co\.tz|go\.tz|io|info|me|tv|co\.za|co\.ke|ng)$', '', d)
                for d in african_domain_keywords
            ])
        
        combined_prioritization_keywords = list(set(combined_prioritization_keywords)) # Remove duplicates
        logging.debug(f"Combined prioritization keywords for scraping: {combined_prioritization_keywords}")

        for r in search_results:
            url = r.get('href', '')
            if not url:
                continue
            
            parsed_url_domain = urlparse(url).netloc.lower()
            is_prioritized = False
            for p_keyword in combined_prioritization_keywords:
                if p_keyword and p_keyword in parsed_url_domain:
                    is_prioritized = True
                    break
            
            if is_prioritized:
                prioritized_results.append(r)
            else:
                other_results.append(r)
        
        # Combine, with prioritized results first
        search_results = prioritized_results + other_results 
        logging.debug(f"Prioritized {len(prioritized_results)} results based on combined criteria.")

        # Process search results, attempting to scrape content
        scraped_count = 0
        for article_info in search_results: # Use search_results directly as it's already prioritized
            if scraped_count >= num_results:
                break
            
            url = article_info.get('href')
            title = article_info.get('title')
            description = article_info.get('body')

            if url and url.startswith('http') and url not in unique_urls and \
               not any(domain in url for domain in EXCLUDE_DOMAINS):
                
                # Prioritize actual articles
                if re.search(r'/(news|article|story|report|post|fact-check|blog|press-release|opinion|analysis)/', url.lower()):
                    # Prepend to articles_to_scrape to try scraping these first
                    articles_to_scrape.insert(0, {'url': url, 'title': title, 'description': description})
                else:
                    articles_to_scrape.append({'url': url, 'title': title, 'description': description})
                unique_urls.add(url)
        
        # Scrape until desired num_results is reached or no more articles
        for article_info in articles_to_scrape:
            if scraped_count >= num_results:
                break
            
            logging.info(f"WEB_SCRAPE: Attempting to scrape content from: {article_info['url']}")
            html_content = get_page_content(article_info['url']) # This uses the cached function
            if html_content:
                full_content = extract_main_content(html_content)
                if full_content and len(full_content) > 300: # Increased minimum content length
                    scraped_sources.append({
                        "title": article_info.get('title', 'Web Scraped Article'),
                        "url": article_info['url'],
                        "full_content": full_content,
                        "snippet": article_info.get('description', full_content[:200] + '...') if article_info.get('description') else (full_content[:200] + '...') if full_content else '',
                        "source_name": urlparse(article_info['url']).netloc # Get source name from URL
                    })
                    scraped_count += 1
                else:
                    logging.warning(f"WEB_SCRAPE: Insufficient content extracted from {article_info['url']}")
            else:
                logging.warning(f"WEB_SCRAPE: Failed to fetch HTML for {article_info['url']}")
    except Exception as e:
        logging.error(f"WEB_SCRAPE: Error during web search and scraping for query '{query}': {e}", exc_info=True)
    
    logging.info(f"WEB_SCRAPE: Finished general web search and scraping. Found {len(scraped_sources)} usable sources.")
    return scraped_sources

# Global list of common African country codes for NewsData.io and intent recognition
AFRICAN_COUNTRY_CODES = [
    'ng', 'za', 'ke', 'eg', 'gh', 'tz', 'ao', 'dz', 'cd', 'et', 'ma', 'sd', 'ug', 
    'zw', 'sn', 'cm', 'mz', 'ci', 'bf', 'ml', 'ne', 'rw', 'so', 'zm', 'tn', 'ly', 
    'tg', 'bj', 'gm', 'mw', 'bi', 'cg', 'ga', 'gn', 'er', 'mr', 'dj', 'sl', 'lr', 
    'mg', 'bw', 'cv', 'sz', 'ls', 'na', 'sc', 'mu', 'td', 'cf', 'gq', 'gw', 'km', 'st'
]


# --- Dedicated News API Fetchers (Data Ingestion Service - EXPANDED) ---
# Each function encapsulates logic for a specific news API

def _fetch_newsapi(query, lang='en', num_results=5):
    """Fetches articles from NewsAPI.org."""
    if not NEWSAPI_API_KEY:
        logging.warning("NEWSAPI_API_KEY is not set. Skipping NewsAPI.org API call.")
        return []

    # NewsAPI requires specifying 'q' and 'pageSize'
    url = f"https://newsapi.org/v2/everything?q={query}&language={lang}&pageSize={num_results}&sortBy=relevancy&apiKey={NEWSAPI_API_KEY}"
    try:
        response = requests.get(url, timeout=7)
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get('articles', []):
            # NewsAPI can return articles with null titles or URLs, filter them
            if article.get('title') and article.get('url'):
                articles.append({
                    "title": article.get('title'),
                    "url": article.get('url'),
                    "full_content": article.get('content', '') or article.get('description', ''), # Prefer content, fallback to description
                    "snippet": article.get('description', 'No snippet available.'),
                    "image_url": article.get('urlToImage', None),
                    "source_name": article.get('source', {}).get('name', 'NewsAPI.org')
                })
        logging.info(f"NEWSAPI_API: Fetched {len(articles)} articles for '{query}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"NEWSAPI_API: Error fetching from NewsAPI.org API for '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"NEWSAPI_API: Error decoding NewsAPI.org API response for '{query}': {e}")
        return []

def _fetch_mediastack(query, lang='en', countries=None, num_results=5):
    """Fetches articles from MediaStack API."""
    if not MEDIASTACK_API_KEY:
        logging.warning("MEDIASTACK_API_KEY is not set. Skipping MediaStack API call.")
        return []

    params = {
        'access_key': MEDIASTACK_API_KEY,
        'keywords': query,
        'languages': lang,
        'limit': num_results,
        'sort': 'published_desc' # Latest news first
    }
    if countries:
        # MediaStack expects a comma-separated string for countries
        if isinstance(countries, list):
            params['countries'] = ','.join(countries)
        else:
            params['countries'] = countries

    # Note: Free tier of MediaStack is HTTP, paid is HTTPS. Adjust if upgrading.
    url = "http://api.mediastack.com/v1/news" 
    try:
        response = requests.get(url, params=params, timeout=7)
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get('data', []):
            if article.get('title') and article.get('url'):
                articles.append({
                    "title": article.get('title'),
                    "url": article.get('url'),
                    "full_content": article.get('description', '') + "\n" + article.get('content', ''), # Concatenate description and content
                    "snippet": article.get('description', ''),
                    "image_url": article.get('image', None),
                    "source_name": article.get('source', 'MediaStack')
                })
        logging.info(f"MEDIASTACK_API: Fetched {len(articles)} articles for '{query}' from countries '{countries}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"MEDIASTACK_API: Error fetching from MediaStack API for '{query}' (countries: {countries}): {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"MEDIASTACK_API: Error decoding MediaStack API response for '{query}' (countries: {countries}): {e}")
        return []

def _fetch_gnews(query, lang='en', country='us', num_results=5):
    """Fetches articles from GNews API."""
    if not GNEWS_API_KEY:
        logging.warning("GNEWS_API_KEY is not set. Skipping GNews API call.")
        return []

    # GNews supports country filtering
    url = f"https://gnews.io/api/v4/search?q={query}&lang={lang}&country={country}&max={num_results}&token={GNEWS_API_KEY}"
    try:
        response = requests.get(url, timeout=7) # Increased timeout
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get('articles', []):
            articles.append({
                "title": article.get('title'),
                "url": article.get('url'),
                "full_content": article.get('content', '') + "\n" + article.get('description', ''), 
                "snippet": article.get('description', article.get('content', '')),
                "image_url": article.get('image', None),
                "source_name": article.get('source', {}).get('name', 'GNews') # Get source name
            })
        logging.info(f"GNEWS_API: Fetched {len(articles)} articles for '{query}' from country '{country}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"GNEWS_API: Error fetching from GNews API for '{query}' (country: {country}): {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"GNEWS_API: Error decoding GNews API response for '{query}' (country: {country}): {e}")
        return []


def _fetch_newsdata(query, lang='en', countries=None, num_results=5):
    """
    Fetches articles from NewsData.io API, now explicitly supporting multiple countries.
    `countries` can be a single country code string, a list of country codes, or None.
    If None, it defaults to 'us' to match previous behavior, but allows for explicit 'global' or 'all' if supported.
    """
    if not NEWSDATA_API_KEY:
        logging.warning("NEWSDATA_API_KEY is not set. Skipping NewsData.io API call.")
        return []

    country_param = ""
    if isinstance(countries, list) and countries:
        country_param = f"&country={','.join(countries)}"
    elif isinstance(countries, str):
        country_param = f"&country={countries}"
    else: # Default to US if no specific country/countries are provided
        country_param = "&country=us" 
        logging.debug("NEWSDATA_API: No specific countries provided, defaulting to 'us'.")


    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&language={lang}&size={num_results}{country_param}"
    try:
        response = requests.get(url, timeout=7) # Increased timeout
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get('results', []):
            articles.append({
                "title": article.get('title'),
                "url": article.get('link'),
                "full_content": article.get('content', article.get('description', '')),
                "snippet": article.get('description', article.get('content', '')),
                "image_url": article.get('image_url', None),
                "source_name": article.get('source_id', 'NewsData.io') # Add source name
            })
        logging.info(f"NEWSDATA_API: Fetched {len(articles)} articles for '{query}' from countries '{countries}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"NEWSDATA_API: Error fetching from NewsData.io API for '{query}' (countries: {countries}): {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"NEWSDATA_API: Error decoding NewsData.io API response for '{query}' (countries: {countries}): {e}")
        return []


def _fetch_nytimes_articles(query, num_results=5):
    """Fetches articles from New York Times Article Search API."""
    if not NYT_API_KEY:
        logging.warning("NYT_API_KEY is not set. Skipping New York Times API call.")
        return []

    # NYT API uses 'fq' for filtered query, 'q' for general query
    # Also, num_results is mapped to 'fl' (fields) for snippet/headline and 'page' for pagination
    # To get `num_results`, we often just ask for the first page and take what's available
    url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={query}&api-key={NYT_API_KEY}"
    try:
        response = requests.get(url, timeout=7) # Increased timeout
        response.raise_for_status()
        data = response.json()
        articles = []
        for doc in data.get('response', {}).get('docs', [])[:num_results]:
            articles.append({
                "title": doc.get('headline', {}).get('main', 'Untitled'),
                "url": doc.get('web_url'),
                "full_content": doc.get('lead_paragraph', doc.get('snippet', '')), # NYT lead_paragraph is a good summary
                "snippet": doc.get('snippet', doc.get('lead_paragraph', '')),
                "image_url": None, # NYT API usually doesn't provide direct image URLs in search results
                "source_name": "New York Times" # Explicit source name
            })
        logging.info(f"NYT_API: Fetched {len(articles)} articles for '{query}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"NYT_API: Error fetching from New York Times API for '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"NYT_API: Error decoding NYT API response for '{query}': {e}")
        return []


# --- NEW NEWS API INTEGRATIONS ---

def _fetch_guardian_articles(query, num_results=5, lang='en'):
    """Fetches articles from The Guardian Open Platform API."""
    if not GUARDIAN_API_KEY:
        logging.warning("GUARDIAN_API_KEY is not set. Skipping The Guardian API call.")
        return []

    # The Guardian API uses 'q' for query, 'page-size' for limit
    url = f"https://content.guardianapis.com/search?q={query}&api-key={GUARDIAN_API_KEY}&page-size={num_results}&show-fields=headline,standfirst,trailText,body"
    try:
        response = requests.get(url, timeout=7)
        response.raise_for_status()
        data = response.json()
        articles = []
        for result in data.get('response', {}).get('results', []):
            fields = result.get('fields', {})
            articles.append({
                "title": fields.get('headline', result.get('webTitle', 'Untitled')),
                "url": result.get('webUrl'),
                "full_content": fields.get('body', fields.get('trailText', fields.get('standfirst', ''))),
                "snippet": fields.get('standfirst', fields.get('trailText', 'No snippet available.')),
                "image_url": None, # Guardian API doesn't always provide image in search results directly
                "source_name": "The Guardian"
            })
        logging.info(f"GUARDIAN_API: Fetched {len(articles)} articles for '{query}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"GUARDIAN_API: Error fetching from The Guardian API for '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"GUARDIAN_API: Error decoding The Guardian API response for '{query}': {e}")
        return []


def _fetch_bing_news(query, num_results=5, market='en-US'):
    """Fetches articles from Bing News Search API (part of Azure Cognitive Services)."""
    if not BING_NEWS_API_KEY:
        logging.warning("BING_NEWS_API_KEY is not set. Skipping Bing News API call.")
        return []

    # Bing News Search API requires 'Ocp-Apim-Subscription-Key' header
    headers = {"Ocp-Apim-Subscription-Key": BING_NEWS_API_KEY}
    params = {
        'q': query,
        'mkt': market, # Market (e.g., 'en-US', 'en-GB', 'fr-FR')
        'count': num_results
    }
    url = "https://api.bing.microsoft.com/v7.0/news/search"
    try:
        response = requests.get(url, headers=headers, params=params, timeout=7)
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get('value', []):
            articles.append({
                "title": article.get('name', 'Untitled'),
                "url": article.get('url'),
                "full_content": article.get('description', ''), # Bing News provides description
                "snippet": article.get('description', 'No snippet available.'),
                "image_url": article.get('image', {}).get('thumbnail', {}).get('contentUrl'),
                "source_name": article.get('provider', [{}])[0].get('name', 'Bing News')
            })
        logging.info(f"BING_NEWS_API: Fetched {len(articles)} articles for '{query}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"BING_NEWS_API: Error fetching from Bing News API for '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"BING_NEWS_API: Error decoding Bing News API response for '{query}': {e}")
        return []


def _fetch_newscaster_articles(query, num_results=5, lang='en'):
    """Fetches articles from NewsCaster (formerly NewsCatcher) API."""
    if not NEWSCASTER_API_KEY:
        logging.warning("NEWSCASTER_API_KEY is not set. Skipping NewsCaster API call.")
        return []

    headers = {
        'x-api-key': NEWSCASTER_API_KEY
    }
    params = {
        'q': query,
        'lang': lang,
        'page_size': num_results
    }
    # Note: NewsCaster API base URL might change, check their documentation.
    url = "https://api.newscatcherapi.com/v2/search" # Example URL, verify from NewsCaster docs
    try:
        response = requests.get(url, headers=headers, params=params, timeout=7)
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get('articles', []):
            articles.append({
                "title": article.get('title', 'Untitled'),
                "url": article.get('link'),
                "full_content": article.get('content', article.get('excerpt', '')),
                "snippet": article.get('excerpt', 'No snippet available.'),
                "image_url": article.get('media', None),
                "source_name": article.get('rights', 'NewsCaster') # 'rights' often has the publisher name
            })
        logging.info(f"NEWSCASTER_API: Fetched {len(articles)} articles for '{query}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"NEWSCASTER_API: Error fetching from NewsCaster API for '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"NEWSCASTER_API: Error decoding NewsCaster API response for '{query}': {e}")
        return []


def _fetch_serpapi_google_news(query, num_results=5, gl='us', hl='en'):
    """
    Fetches news articles via SerpAPI using the Google News API endpoint.
    This effectively uses Google News as a source aggregator.
    Requires `pip install google-search-results` if you want to use the SerpAPI Python client.
    Here, we'll use direct requests for simplicity.
    """
    if not SERPAPI_API_KEY:
        logging.warning("SERPAPI_API_KEY is not set. Skipping SerpAPI Google News API call.")
        return []

    params = {
        'q': query,
        'tbm': 'nws', # Specifies "news" tab for Google Search
        'api_key': SERPAPI_API_KEY,
        'num': num_results, # Number of results
        'gl': gl, # Geo-location for search (country code)
        'hl': hl # Host language
    }
    url = "https://serpapi.com/search"
    try:
        response = requests.get(url, params=params, timeout=10) # Increased timeout for external service
        response.raise_for_status()
        data = response.json()
        articles = []
        # SerpAPI returns 'news_results' for news searches
        for article in data.get('news_results', []):
            articles.append({
                "title": article.get('title', 'Untitled'),
                "url": article.get('link'),
                "full_content": article.get('snippet', ''),
                "snippet": article.get('snippet', 'No snippet available.'),
                "image_url": article.get('thumbnail', None), # Google News often has thumbnails
                "source_name": article.get('source', 'SerpAPI (Google News)')
            })
        logging.info(f"SERPAPI_GOOGLE_NEWS: Fetched {len(articles)} articles for '{query}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"SERPAPI_GOOGLE_NEWS: Error fetching from SerpAPI (Google News) for '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"SERPAPI_GOOGLE_NEWS: Error decoding SerpAPI (Google News) response for '{query}': {e}")
        return []


def _fetch_ap_news_articles(query, num_results=5):
    """
    Conceptual: Fetches articles from Associated Press (AP) API.
    NOTE: AP API usually requires commercial licensing and direct contracts.
    This is a placeholder demonstrating how to integrate such an API.
    """
    if not AP_NEWS_API_KEY:
        logging.warning("AP_NEWS_API_KEY is not set. Skipping Associated Press API call (conceptual).")
        return []

    # Placeholder URL and parameters - replace with actual AP API endpoint and params
    # Example: url = f"https://api.ap.org/content/v2/search?q={query}&count={num_results}&apiKey={AP_NEWS_API_KEY}"
    # Headers or other authentication methods might be required.
    logging.info(f"AP_NEWS_API (Conceptual): Attempting to fetch articles for '{query}'. This API typically requires commercial access.")
    
    # Simulate a call for demonstration
    articles = []
    if AP_NEWS_API_KEY == "DEMO_KEY_AP": # Replace with actual logic
        articles.append({
            "title": f"Conceptual AP Article for '{query}'",
            "url": "https://www.apnews.com/conceptual-article",
            "full_content": "This is a simulated article from the Associated Press, demonstrating where real API content would go. AP provides breaking news and investigative journalism.",
            "snippet": "Simulated snippet from AP.",
            "image_url": None,
            "source_name": "Associated Press (Conceptual)"
        })
    else:
        # In a real scenario, make the actual API request here
        pass # e.g., response = requests.get(url, headers=headers, params=params, timeout=7)
             # then parse response.json() for articles

    logging.info(f"AP_NEWS_API (Conceptual): Fetched {len(articles)} simulated articles for '{query}'.")
    return articles


def _fetch_financial_times_articles(query, num_results=5):
    """
    Conceptual: Fetches articles from Financial Times (FT) API.
    NOTE: FT API is highly specialized for financial news and typically requires enterprise access.
    This is a placeholder demonstrating how to integrate such an API.
    """
    if not FINANCIAL_TIMES_API_KEY:
        logging.warning("FINANCIAL_TIMES_API_KEY is not set. Skipping Financial Times API call (conceptual).")
        return []

    # Placeholder URL and parameters - replace with actual FT API endpoint and params
    # Example: url = f"https://api.ft.com/content/v1/search?q={query}&limit={num_results}"
    # FT API often uses OAuth2 or API key in headers.
    logging.info(f"FINANCIAL_TIMES_API (Conceptual): Attempting to fetch articles for '{query}'. This API typically requires commercial access.")

    # Simulate a call for demonstration
    articles = []
    if FINANCIAL_TIMES_API_KEY == "DEMO_KEY_FT": # Replace with actual logic
        articles.append({
            "title": f"Conceptual FT Article: {query} and Markets",
            "url": "https://www.ft.com/conceptual-article",
            "full_content": "This is a simulated article from the Financial Times, focusing on global markets and economic analysis. FT provides in-depth business and financial news.",
            "snippet": "Simulated snippet from FT.",
            "image_url": None,
            "source_name": "Financial Times (Conceptual)"
        })
    else:
        # In a real scenario, make the actual API request here
        pass

    logging.info(f"FINANCIAL_TIMES_API (Conceptual): Fetched {len(articles)} simulated articles for '{query}'.")
    return articles


def get_aggregated_news(query, num_results=5, preferred_sources=None, preferred_topics=None, region_focus=None):
    """
    Aggregates news from various sources, prioritizing web scraping, then dedicated APIs.
    Filters/prioritizes by `preferred_sources` and incorporates `preferred_topics`.
    `region_focus` can be 'africa' or a specific country code to prioritize.
    """
    aggregated_sources = []
    unique_urls = set() # Track unique URLs to avoid duplicates

    # Strategy: First, try web scraping for a portion of results, then fill remainder with APIs.
    # This prioritizes generic web search for broader, potentially fresher content, as requested.
    
    # How many results to try and get from web scraping first
    num_from_web_scrape = min(num_results, 3) # Try to get at least 3 from scraping if num_results allows

    logging.info(f"AGGREGATOR: Attempting to fetch {num_from_web_scrape} results from general web scraping for '{query}'.")
    web_scraped_articles = perform_web_search_and_scrape(
        query, 
        num_results=num_from_web_scrape, 
        preferred_sources=preferred_sources, 
        preferred_topics=preferred_topics,
        region_bias=region_focus # Pass region_focus to web scraping
    )
    for article in web_scraped_articles:
        if article.get('url') and article['url'] not in unique_urls:
            aggregated_sources.append(article)
            unique_urls.add(article['url'])
            if len(aggregated_sources) >= num_results:
                break
    
    logging.info(f"AGGREGATOR: After web scraping, found {len(aggregated_sources)} articles.")

    # Determine the order of API calls based on region_focus for the remaining needed articles
    remaining_needed = num_results - len(aggregated_sources)
    if remaining_needed > 0:
        api_call_order = []
        
        # Define API call functions with their default/primary region handling
        # NewsData.io is good for specific countries, GNews is also country-specific, NewsAPI is broader, MediaStack is global with country filters.

        # Prioritize region-specific APIs if `region_focus` is a country code
        if region_focus and region_focus != 'africa':
            # Add NewsData.io and GNews for the specific country if provided
            api_call_order.append(
                (lambda q, n: _fetch_newsdata(q, countries=[region_focus], num_results=n), f"NewsData.io API ({region_focus.upper()})")
            )
            api_call_order.append(
                (lambda q, n: _fetch_gnews(q, country=region_focus, num_results=n), f"GNews API ({region_focus.upper()})")
            )
        
        # Always include NewsAPI.org and MediaStack for broader global coverage
        api_call_order.append((_fetch_newsapi, "NewsAPI.org"))
        api_call_order.append((_fetch_mediastack, "MediaStack API"))
        api_call_order.append((_fetch_guardian_articles, "The Guardian API")) # New
        api_call_order.append((_fetch_bing_news, "Bing News API")) # New
        api_call_order.append((_fetch_newscaster_articles, "NewsCaster API")) # New
        api_call_order.append((_fetch_serpapi_google_news, "SerpAPI (Google News)")) # New

        # Add NewsData.io (global/US fallback) and NYT as general options
        api_call_order.append((_fetch_newsdata, "NewsData.io API (General)"))
        api_call_order.append((_fetch_nytimes_articles, "NYT API"))
        
        # Conceptual/Enterprise APIs (often slower or rate-limited for demo)
        api_call_order.append((_fetch_ap_news_articles, "Associated Press (Conceptual)")) # New Conceptual
        api_call_order.append((_fetch_financial_times_articles, "Financial Times (Conceptual)")) # New Conceptual

        # If region_focus is 'africa', add a specific NewsData.io call with AFRICAN_COUNTRY_CODES
        if region_focus == 'africa':
            api_call_order.insert(0, # Insert at the beginning to prioritize
                (lambda q, n: _fetch_newsdata(q, countries=AFRICAN_COUNTRY_CODES, num_results=n), "NewsData.io API (African Countries)")
            )


        for fetch_func_tuple, api_name in api_call_order:
            if len(aggregated_sources) >= num_results:
                break
            
            current_api_needed = num_results - len(aggregated_sources)
            if current_api_needed <= 0:
                break

            logging.info(f"AGGREGATOR: Attempting to fetch from {api_name} for query: '{query}', needing {current_api_needed} more results.")
            try:
                api_articles = []
                # Handle specific fetch_func signatures (e.g., those needing 'countries' param)
                if "NewsData.io API (African Countries)" in api_name:
                    api_articles = fetch_func_tuple(query, current_api_needed) # `fetch_func_tuple` already defined with AFRICAN_COUNTRY_CODES
                elif "NewsData.io API (" in api_name and region_focus and region_focus != 'africa': # Specific country NewsData.io
                     api_articles = fetch_func_tuple(query, current_api_needed) # `fetch_func_tuple` already defined with specific country
                elif "GNews API (" in api_name and region_focus and region_focus != 'africa': # Specific country GNews
                    api_articles = fetch_func_tuple(query, current_api_needed) # `fetch_func_tuple` already defined with specific country
                else: # General calls
                    api_articles = fetch_func_tuple(query, current_api_needed)
                
                # Apply preferred_sources filter if specified. This is done here as APIs might not support direct filtering.
                if preferred_sources:
                    preferred_domain_keywords = [
                        re.sub(r'^(www\.)?|\.(com|org|net|co\.tz|go\.tz|io|info|me|tv|co\.za|co\.ke|ng)$', '', urlparse(s).netloc.lower() if s.startswith('http') else s.lower())
                        for s in preferred_sources
                    ]
                    filtered_articles = []
                    for article in api_articles:
                        url = article.get('url', '')
                        if url:
                            parsed_url_domain = urlparse(url).netloc.lower()
                            if any(p_keyword in parsed_url_domain for p_keyword in preferred_domain_keywords):
                                filtered_articles.append(article)
                    logging.info(f"AGGREGATOR: {len(filtered_articles)} articles from {api_name} matched preferred sources.")
                    api_articles = filtered_articles

                for article in api_articles:
                    if article.get('url') and article['url'] not in unique_urls:
                        aggregated_sources.append(article)
                        unique_urls.add(article['url'])
                        if len(aggregated_sources) >= num_results:
                            break
            except Exception as e:
                logging.error(f"AGGREGATOR: Error integrating {api_name}: {e}")
                # Continue to next API even if one fails

    logging.info(f"AGGREGATOR: Aggregated {len(aggregated_sources)} articles for '{query}'.")
    return aggregated_sources

# --- Simulated Tooling for AI (Conceptual Tool Service) ---
# In a real-world scenario, these tools could be separate modules or functions called by a tool-orchestration service.
def call_news_api_tool(query, category=None, date_range=None, preferred_sources=None, preferred_topics=None, region_focus=None):
    """
    Simulates calling a news API to get recent articles.
    This now primarily uses the `get_aggregated_news` function.
    """
    logging.info(f"TOOL: Calling News API for query='{query}', category='{category}', date_range='{date_range}' with preferred sources: {preferred_sources}, preferred topics: {preferred_topics}, region_focus: {region_focus}")
    search_term = query
    if category:
        search_term += f" {category} news"
    if date_range:
        search_term += f" {date_range}"

    sources = get_aggregated_news(
        search_term, 
        num_results=5, 
        preferred_sources=preferred_sources, 
        preferred_topics=preferred_topics,
        region_focus=region_focus # Pass region_focus to the aggregator
    )

    if not sources:
        return "No relevant news articles found."

    formatted_news = []
    for i, s in enumerate(sources):
        formatted_news.append(f"Article {i+1}: {s.get('title', 'Untitled')} (Source: {s.get('source_name', 'N/A')}, URL: {s.get('url', 'N/A')})\nSnippet: {s.get('snippet', 'No snippet available.')}")
    return "\n\n".join(formatted_news)

def perform_fact_check_tool(statement):
    """
    Performs a simulated fact check using web search and simple heuristics.
    Assigns a confidence level based on source consistency.
    Returns structured evidence.
    """
    logging.info(f"TOOL: Performing fact check for statement: '{statement}'")
    statement_lower = statement.lower()
    
    # Rule-based definitive verdicts for common cases
    if "earth flat" in statement_lower:
        return {
            "verdict": "Inaccurate",
            "confidence": "High",
            "summary": "Scientific consensus and observable facts overwhelmingly demonstrate the Earth is an oblate spheroid (a sphere slightly flattened at the poles).",
            "evidence_list": [
                {"title": "NASA Science", "url": "https://science.nasa.gov/science-process/scientific-method/", "snippet": "Observations from space, GPS, and countless scientific experiments confirm Earth's spherical shape."},
                {"title": "National Geographic", "url": "https://www.nationalgeographic.com/science/article/earths-shape-and-size", "snippet": "Early Greek philosophers provided evidence for a spherical Earth, and modern science has solidified this understanding."}
            ]
        }
    elif "nexusai" in statement_lower or "ai news analyzer" in statement_lower or "saint ai" in statement_lower:
        return {
            "verdict": "Accurate",
            "confidence": "High",
            "summary": "NexusAI (also known as SaintAI) is designed to be an AI News Analyst and Fact-Checker, providing summaries and verification of information based on web search and integrated APIs.",
            "evidence_list": [
                {"title": "NexusAI Project Overview", "url": "#", "snippet": "Internal project documentation confirms NexusAI's core mission and capabilities."},
                {"title": "AI Fact-Checking Systems", "url": "https://example.com/ai-fact-checking-tech", "snippet": "AI models are increasingly used for information verification, aligning with NexusAI's design."}
            ]
        }

    # Attempt to find supporting/contradictory evidence from web sources
    search_queries = [
        f"fact check {statement}",
        f"is {statement} true",
        f"{statement} debunked",
        f"{statement} reliable sources"
    ]
    
    all_evidence_sources = []
    for q in search_queries:
        # Fetch actual web content for better evidence snippets
        # Do not apply specific region_focus for fact-checking unless explicitly asked.
        search_results = get_aggregated_news(q, num_results=3, region_focus=None) 
        all_evidence_sources.extend(search_results)
    
    unique_evidence_urls = set()
    filtered_evidence = []
    for item in all_evidence_sources:
        if item.get('url') and item['url'] not in unique_evidence_urls:
            # Only add if full_content is available and reasonable length for real analysis
            if item.get('full_content') and len(item['full_content']) > 100:
                filtered_evidence.append(item)
                unique_evidence_urls.add(item['url'])

    if not filtered_evidence:
        return {
            "verdict": "Unproven",
            "confidence": "Low",
            "summary": "No immediate direct evidence found from available reliable sources to confirm or deny the statement.",
            "evidence_list": []
        }

    # Simple heuristic for confidence and verdict based on source consistency
    supporting_sources = []
    contradictory_sources = []
    
    for s in filtered_evidence:
        full_content_lower = s.get('full_content', '').lower()
        snippet_lower = s.get('snippet', '').lower()

        # Improved keyword checks for support/contradiction
        is_supporting = ("true" in full_content_lower or "confirmed" in full_content_lower or 
                         "fact" in full_content_lower and "not fact" not in full_content_lower and "misinformation" not in full_content_lower)
        is_contradictory = ("false" in full_content_lower or "debunked" in full_content_lower or 
                            "hoax" in full_content_lower or "misleading" in full_content_lower or "untrue" in full_content_lower)
        
        # Prioritize explicit contradiction
        if is_contradictory and not is_supporting:
            contradictory_sources.append(s)
        elif is_supporting and not is_contradictory:
            supporting_sources.append(s)
        # If both or ambiguous, categorize based on general snippet presence
        elif statement_lower in snippet_lower:
            # If the statement is present and no strong contradiction, leaning towards support
            supporting_sources.append(s)
    
    verdict = "Unproven"
    confidence = "Low"
    summary = ""

    if len(supporting_sources) > len(contradictory_sources) * 1.5 and len(supporting_sources) >= 2:
        verdict = "Accurate"
        confidence = "High" if len(supporting_sources) >= 3 else "Medium"
        summary = "Multiple reliable sources provide evidence supporting the statement."
    elif len(contradictory_sources) > len(supporting_sources) * 1.5 and len(contradictory_sources) >= 2:
        verdict = "Inaccurate"
        confidence = "High" if len(contradictory_sources) >= 3 else "Medium"
        summary = "Several reliable sources contradict or debunk the statement, indicating it is likely inaccurate."
    elif supporting_sources or contradictory_sources:
        verdict = "Partially True / Requires Further Analysis"
        confidence = "Medium"
        summary = "Some evidence supports the statement, while other information suggests it may be partially inaccurate or requires more context. Further analysis is recommended."
    else:
        verdict = "Unproven"
        confidence = "Low"
        summary = "No clear supporting or contradictory evidence was found across available sources."

    # Combine all relevant sources for the evidence list presented to the LLM
    final_evidence_list = []
    for s in supporting_sources:
        final_evidence_list.append({"title": s.get('title', 'N/A'), "url": s.get('url', '#'), "snippet": s.get('snippet', 'No snippet available.'), "type": "supporting", "source_name": s.get('source_name', 'N/A')})
    for s in contradictory_sources:
        # FIX: Corrected the unclosed single quote in 'contradictory'
        final_evidence_list.append({"title": s.get('title', 'N/A'), "url": s.get('url', '#'), "snippet": s.get('snippet', 'No snippet available.'), "type": "contradictory", "source_name": s.get('source_name', 'N/A')})

    # Prioritize displaying supporting/contradictory evidence that directly addresses the statement
    # For a more concise list, pick a few key ones.
    if len(final_evidence_list) > 5: # Limit for conciseness in LLM prompt
        final_evidence_list = final_evidence_list[:5]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary, # Summary for the LLM to use
        "evidence_list": final_evidence_list # Structured evidence for frontend and LLM
    }

def get_definition_tool(term):
    """Provides definitions, preferring predefined ones, then web search."""
    logging.info(f"TOOL: Getting definition for term: '{term}')")
    definitions = {
        "ai": {
            "content": "Artificial Intelligence: The theory and development of computer systems able to perform tasks that normally require human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages.",
            "source": "Internal Knowledge Base",
            "url": "#"
        },
        "blockchain": {
            "content": "A distributed, decentralized, and public digital ledger used to record transactions across many computers so that the record cannot be altered retroactively without the alteration of all subsequent blocks and the consensus of the network. It's the underlying technology for cryptocurrencies like Bitcoin.",
            "source": "Internal Knowledge Base",
            "url": "#"
        },
        "quantum computing": {
            "content": "An emerging area of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform computations that are impossible or impractical for classical computers. It has the potential to solve certain problems exponentially faster than classical computers.",
            "source": "Internal Knowledge Base",
            "url": "#"
        }
    }
    definition_data = definitions.get(term.lower())
    if definition_data:
        return definition_data
    
    # Use general web search for definitions, prioritize results that seem like definitions
    # Do not apply specific region_focus for definitions.
    search_results = get_aggregated_news(f"definition of {term}", num_results=2, region_focus=None)
    if search_results:
        # Try to find a snippet that looks like a direct definition
        for res in search_results:
            if "definition" in res.get('snippet', '').lower() or "means" in res.get('snippet', '').lower():
                return {"content": res['snippet'], "source": res.get('title', 'Web Search'), "url": res.get('url', '#'), "source_name": res.get('source_name', 'N/A')}
        # Fallback to the first good snippet if no explicit definition found
        return {"content": search_results[0]['snippet'], "source": search_results[0].get('title', 'Web Search'), "url": search_results[0].get('url', '#'), "source_name": search_results[0].get('source_name', 'N/A')}
    
    return {"content": f"Definition for '{term}' not found.", "source": "N/A", "url": "#", "source_name": "N/A"}

def analyze_sentiment_tool(text):
    """
    Analyzes sentiment (Positive, Neutral, Negative, Mixed) and basic, conceptual bias (Left, Right, Neutral, Mixed/Slight Bias Detected).
    This is a simplified, keyword-based analysis with more granular keywords.
    """
    logging.info(f"TOOL: Analyzing sentiment and bias for text (first 100 chars): '{text[:100]}'")
    text_lower = text.lower()
    
    # Sentiment Keywords (expanded)
    positive_keywords = ["good", "great", "excellent", "positive", "success", "hope", "improve", "win", "growth", "optimistic", "strong", "favorable", "breakthrough", "advantage", "benefit", "progress", "achieve", "thrive", "promising", "robust", "resilient"]
    negative_keywords = ["bad", "poor", "terrible", "negative", "fail", "crisis", "loss", "worry", "decline", "pessimistic", "weak", "unfavorable", "setback", "challenge", "concern", "risk", "damage", "deteriorate", "struggle", "volatile", "disappointing"]
    
    positive_score = sum(text_lower.count(k) for k in positive_keywords)
    negative_score = sum(text_lower.count(k) for k in negative_keywords)

    sentiment = "Neutral"
    if positive_score > negative_score * 1.5 and positive_score > 0:
        sentiment = "Positive"
    elif negative_score > positive_score * 1.5 and negative_score > 0:
        sentiment = "Negative"
    elif positive_score > 0 and negative_score > 0:
        sentiment = "Mixed" # Both positive and negative elements present
    
    # Basic Bias Detection (conceptual, keyword-based, expanded)
    left_leaning_keywords = ["progressive", "liberal", "social justice", "equality", "union", "climate action", "worker rights", "community", "regulation", "public services", "diversity"]
    right_leaning_keywords = ["conservative", "libertarian", "free market", "individual liberty", "tax cuts", "border security", "tradition", "property rights", "deregulation", "private enterprise"]

    left_score = sum(text_lower.count(k) for k in left_leaning_keywords)
    right_score = sum(text_lower.count(k) for k in right_leaning_keywords)

    bias = "Neutral"
    if left_score > right_score * 1.5 and left_score > 0:
        bias = "Left-leaning"
    elif right_score > left_score * 1.5 and right_score > 0:
        bias = "Right-leaning"
    elif left_score > 0 or right_score > 0: # If any keywords found, but not strongly one way
        bias = "Mixed/Slight Bias Detected"


    return {"sentiment": sentiment, "bias": bias}

def summarize_content_with_llm(content_list, analysis_depth='standard'):
    """
    Uses the LLM to summarize a list of content strings.
    Adjusts summarization prompt based on `analysis_depth`.
    """
    combined_content = "\n\n".join(content_list)
    if not combined_content:
        return "No content provided to summarize."

    # Truncate combined content to ensure it fits within typical LLM context limits for summarization
    MAX_SUMMARIZATION_CONTENT_LENGTH = 15000 
    if len(combined_content) > MAX_SUMMARIZATION_CONTENT_LENGTH:
        combined_content = combined_content[:MAX_SUMMARIZATION_CONTENT_LENGTH] + "\n\n[...content truncated for brevity...]"

    summary_instruction = "Summarize the following content. Include key findings, main arguments, and any significant details."
    if analysis_depth == 'brief':
        summary_instruction += " Keep the summary very concise, focusing on the core message (1-2 paragraphs, max 5 sentences)."
    elif analysis_depth == 'standard':
        summary_instruction += " Provide a balanced summary, hitting key points and important details (3-5 paragraphs or concise bullet points, max 10-15 sentences)."
    elif analysis_depth == 'detailed':
        summary_instruction += " Generate a comprehensive and detailed summary, including nuanced information, multiple facets, and direct quotes where relevant (5+ paragraphs or detailed bullet points)."
    
    summary_prompt = f"{summary_instruction}\n\nContent to summarize:\n{combined_content}"

    if llm_pipeline:
        messages = [{"role": "user", "content": summary_prompt}]
        formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        summary_output = llm_pipeline(formatted_prompt)
        summary_text = summary_output[0]['generated_text']
        return summary_text.replace(formatted_prompt, "").strip() if formatted_prompt in summary_text else summary_text.strip()
    elif llm_model:
        summary_response = llm_model.generate_content([{"role": "user", "parts": [{"text": summary_prompt}]}])
        return summary_response.text.strip()
    else:
        return "LLM not configured for summarization."


# --- AI Logic & Orchestration (Conceptual AI Service) ---
# In a real-world scenario, this would be a core AI orchestration service, potentially separated from Flask.
def estimate_tokens(text):
    """A rough estimate of tokens for text to check against API limits."""
    # This is a heuristic. Actual tokenization depends on the model's tokenizer.
    # A common rule of thumb is 1 word = ~1.3-1.5 tokens.
    return len(text.split()) * 1.5 

def format_conversation_history(messages, max_history_tokens):
    """
    Formats conversation history for the LLM, including message numbering and ensuring it fits within
    the token budget. Prioritizes recent messages if truncation is necessary.
    
    Args:
        messages (list[Message]): List of Message objects from the conversation, ordered chronologically.
        max_history_tokens (int): Maximum token budget for the conversation history.
                                  This budget should account for the LLM's context window minus
                                  system instructions and the current prompt/tool context.
    Returns:
        list[dict]: Formatted messages ready for LLM API, with numbering.
    """
    history_for_llm = []
    current_tokens = 0
    
    # We want messages to be numbered in chronological order (Message 1, Message 2, etc.)
    # but still prioritize recent messages if truncation is needed.
    # So, we'll build from the end (most recent), but insert at the beginning to keep order.
    
    # Store messages with their original index for reference
    indexed_messages = []
    for i, msg in enumerate(messages):
        indexed_messages.append({"index": i + 1, "msg": msg})

    skipped_messages_count = 0
    
    # Iterate from the most recent message backwards
    for item in reversed(indexed_messages):
        msg_index = item["index"]
        msg = item["msg"]
        
        # Include sender, content, sentiment, bias, and sources in the formatted message
        msg_content = msg.content
        msg_sentiment = msg.sentiment if msg.sentiment else "N/A"
        msg_bias = msg.bias if msg.bias else "N/A"
        msg_sources = ""
        if msg.sources:
            # Format sources for LLM context, making sure they are readable text
            formatted_sources = []
            for s in msg.sources:
                title = s.get('title', s.get('url', 'N/A'))
                url = s.get('url', 'N/A')
                snippet = s.get('snippet', '')
                source_name = s.get('source_name', 'N/A')
                formatted_sources.append(f"- Title: {title} (Source: {source_name})\n  URL: {url}\n  Snippet: {snippet}")
            msg_sources = "\nSources:\n" + "\n".join(formatted_sources)

        role = "model" if msg.sender == "ai" else "user"
        
        # Add message number to content for LLM to reference
        formatted_content = (
            f"Message {msg_index} ({msg.sender}):\n"
            f"Content: {msg_content}\n"
            f"Sentiment: {msg_sentiment}\n"
            f"Bias: {msg_bias}"
            f"{msg_sources}"
        )
        
        # Estimate tokens for this formatted message + a small buffer for role/structure overhead
        msg_token_estimate = estimate_tokens(formatted_content) + 10 

        # Check if adding this message (and keeping previous ones) exceeds the budget
        if (current_tokens + msg_token_estimate) > max_history_tokens:
            skipped_messages_count += 1
            continue # Skip this message and older ones

        # Add the message to the beginning of the history_for_llm list
        history_for_llm.insert(0, {"role": role, "parts": [{"text": formatted_content}]})
        current_tokens += msg_token_estimate

    if skipped_messages_count > 0:
        logging.warning(f"Truncated conversation history by {skipped_messages_count} messages to fit within {max_history_tokens} tokens.")
        # Optionally, you could prepend a note to the LLM that history was truncated,
        # or implement more advanced summarization of the truncated part here.
        # For this app, a simple truncation and warning is used for efficiency.

    return history_for_llm


def get_ai_response(user_prompt, conversation_messages=[], current_user_role=UserRole.STANDARD,
                    preferred_sources=None, analysis_depth='standard', preferred_topics=None):
    response_content = ""
    response_details = "AI analysis from NexusAI."
    response_sources = []
    response_sentiment = "Neutral"
    response_bias = "Neutral"

    # Define the system instruction for the LLM
    llm_system_instruction = (
        "You are NexusAI, an expert AI News Analyst and Fact-Checker. "
        "Your goal is to provide concise, accurate, and insightful responses, drawing from provided context and tools. "
        "**Crucially, do NOT state you are an AI or language model.** Focus on the information itself. "
        "Synthesize information from the provided web search results and tool outputs to answer the user's query comprehensively. "
        "For news analysis, summarize the most significant headlines, developments, and trends, citing sources clearly. "
        "For fact-checks, clearly state the **Verdict** (Accurate, Inaccurate, Unproven, Partially True/Requires Further Analysis), **Confidence** (High, Medium, Low), and provide supporting **Evidence** from multiple sources with brief snippets and links. "
        "**Always cite external sources (web search results, articles, reports) by their title and URL using Markdown link format: [Title of Article](URL).** "
        "Format your output clearly using Markdown (e.g., **bold**, *italics*, lists, code blocks, horizontal rules `---` for sections). "
        "Pay close attention to conversation history and message numbering (e.g., 'Message N (sender): ...') for context, but **DO NOT include 'Message N (sender):' prefixes in your own responses.** "
        "Conclude each response with a natural, engaging, and relevant follow-up question or statement."
    )
    
    # Adjust analysis depth for the LLM prompt
    if analysis_depth == 'brief':
        llm_system_instruction += " Keep your responses very brief and to the point (1-3 sentences or very concise paragraphs)."
    elif analysis_depth == 'standard':
        llm_system_instruction += " Provide a balanced response, covering key aspects concisely (2-4 paragraphs)."
    else: # detailed
        llm_system_instruction += " Provide detailed and comprehensive responses, exploring multiple facets of the topic, using paragraphs and lists. Include nuanced information."


    # Determine maximum total tokens based on model capabilities.
    # Using a large window for "whole chat memory" (Gemini 1.5 Flash supports 1M, open-source varies)
    MAX_MODEL_CONTEXT_WINDOW = 32000 # Set a practical limit, even if model supports more, for performance

    # Estimate tokens for critical parts *excluding* the history, as history budget is calculated from total.
    estimated_non_history_tokens_min = (
        estimate_tokens(llm_system_instruction) + 
        estimate_tokens(user_prompt) + 
        5000 # Generous buffer for tool output and web context to avoid truncation
    )
    
    # Calculate the remaining budget for conversation history
    max_history_tokens_budget = MAX_MODEL_CONTEXT_WINDOW - estimated_non_history_tokens_min
    if max_history_tokens_budget < 0:
        max_history_tokens_budget = 0 # Ensure it's not negative

    # Prepare historical messages using the token-aware formatting and numbering
    llm_history = format_conversation_history(conversation_messages, max_history_tokens=max_history_tokens_budget)

    # --- Intent Recognition & Tool Selection ---
    # This section now leverages the LLM more heavily for complex intent detection,
    # especially for news-related queries that might not have obvious keywords.
    # It also includes a robust rule-based fallback.

    intent_prompt_text = (
        f"You are an expert intent recognition system for a News Analyst AI. Your task is to analyze the user's prompt "
        f"and context to determine the primary intent, even if implicitly stated. Be precise in identifying keywords, "
        f"entities, or implied topics. Consider the overall conversation flow.\n\n"
        f"Possible Intents:\n"
        f"- **news_analysis**: The user is asking for current events, updates, headlines, trends, analysis of recent events, "
        f"or information about ongoing situations. Look for keywords like 'news', 'latest', 'breaking', 'updates', 'current events', "
        f"'happenings', 'recent developments', 'what's going on with', 'state of', 'trends', 'headlines'. "
        f"Also, infer news intent if the user asks about well-known companies, political figures, major events, or "
        f"broad topics (e.g., 'economy', 'tech', 'environment') without specifying 'definition' or 'fact-check'. "
        f"If a specific source is requested, note it.\n"
        f"- **african_news_analysis**: Similar to 'news_analysis' but specifically focuses on news related to Africa or specific African countries. "
        f"Look for keywords like 'Africa news', 'African continent', specific African countries (e.g., 'Kenya', 'Nigeria', 'South Africa', 'Tanzania'), "
        f"or regional terms (e.g., 'East Africa', 'West Africa').\n" 
        f"- **fact_check**: The user wants to verify the truthfulness of a statement. Look for phrases like 'fact-check', "
        f"'is it true', 'verify', 'true or false', 'confirm if', 'debunk', 'authenticity of'.\n"
        f"- **summarization**: The user wants a summary or condensation of information. Look for 'summarize', 'condense', "
        f"'main points', 'TL;DR', 'brief overview'. If no explicit content is given, assume they want a summary of the *previous AI response* if available.\n"
        f"- **explain_concept**: The user is asking for a definition or explanation of a term or concept. Look for 'explain', "
        f"'define', 'what is', 'tell me about', 'meaning of'.\n"
        f"- **sentiment_analysis**: The user wants to know the emotional tone or sentiment of a piece of text or a topic. "
        f"Look for 'sentiment', 'mood', 'tone', 'how does [article/topic] feel', 'positive or negative'.\n"
        f"- **query_previous_message**: The user is explicitly referencing a previous message by its number. Look for patterns like "
        f"'Message N', 'the Nth prompt', 'your response to my M query'. The 'query' should be the specific question about that message.\n"
        f"- **general_inquiry**: For anything else. This is the fallback.\n\n"
        f"Output your decision in a JSON format in respective prompt language answer in prompts language. If the intent is 'news_analysis', include a 'query' field and optionally 'category', 'date_range', 'sources'. "
        f"If 'african_news_analysis', include a 'query' field and optionally 'country_codes' (list of 2-letter codes) or 'region' (e.g., 'East Africa'). " 
        f"If 'fact_check', include a 'statement'. If 'explain_concept', include a 'term'. If 'query_previous_message', include 'message_index' (integer) and 'sub_query'.\n"
        f"Example for african_news_analysis: {{'intent': 'african_news_analysis', 'query': 'news in Kenya', 'country_codes': ['ke']}}\n" 
        f"User prompt: '{user_prompt}'\n"
        f"Conversation history (recent 3 for context, DO NOT include in JSON output unless needed for analysis): {llm_history[-3:]}" # Provide small history for LLM to reason on
    )

    intent_data = {"intent": "general_inquiry", "query": user_prompt} # Default intent
    
    # Try LLM-based intent detection first
    if llm_pipeline or llm_model:
        try:
            generated_text = ""
            if llm_pipeline: # Using open-source LLM for intent detection
                messages_for_intent = [{"role": "user", "content": intent_prompt_text}]
                formatted_intent_prompt = llm_pipeline.tokenizer.apply_chat_template(messages_for_intent, tokenize=False, add_generation_prompt=True)
                intent_response_output = llm_pipeline(formatted_intent_prompt)
                generated_text = intent_response_output[0]['generated_text']
                if formatted_intent_prompt in generated_text:
                    generated_text = generated_text.replace(formatted_intent_prompt, "").strip()
            elif llm_model: # Using Gemini for intent detection
                intent_response = llm_model.generate_content([{"role": "user", "parts": [{"text": intent_prompt_text}]}])
                generated_text = intent_response.text.strip()
            
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_intent = json.loads(json_str)
                # Validate the parsed intent to ensure it's one of the recognized types
                if parsed_intent.get("intent") in ["news_analysis", "african_news_analysis", "fact_check", "summarization", "explain_concept", "sentiment_analysis", "query_previous_message", "general_inquiry"]:
                    intent_data.update(parsed_intent)
                    logging.info(f"LLM detected intent: {intent_data}")
                else:
                    logging.warning(f"LLM returned an unrecognized intent type: {parsed_intent.get('intent')}. Falling back to rule-based.")
            else:
                logging.warning(f"LLM did not return parsable JSON for intent. Raw response: {generated_text}. Falling back to rule-based.")

        except Exception as e:
            logging.error(f"Error detecting intent with LLM, falling back to rule-based matching: {e}", exc_info=True)
            # Fallback to rule-based is handled below
    
    # --- Rule-based Fallback for Intent Recognition (more comprehensive) ---
    # This acts as a safeguard if the LLM-based intent detection fails or is unavailable.
    user_prompt_lower = user_prompt.lower()

    # Query Previous Message
    prev_msg_match = re.search(r'(?:about|regarding)?\s*(?:my|the)?\s*(?:message|prompt|query)?\s*(\d+)\s*(?:\s*|:|\s*what\s*was\s*|tell\s*me\s*more\s*about\s*)?(.*)', user_prompt_lower, re.IGNORECASE)
    if prev_msg_match and llm_history: # Only attempt if history exists
        message_index_str = prev_msg_match.group(1)
        sub_query_str = prev_msg_match.group(2).strip()
        try:
            message_index = int(message_index_str)
            if 1 <= message_index <= len(conversation_messages):
                intent_data = {"intent": "query_previous_message", "message_index": message_index, "sub_query": sub_query_str or "what was that message about"}
                logging.info(f"Rule-based intent: query_previous_message detected for index {message_index}.")
        except ValueError:
            pass # Not a valid number, continue with other checks

    # African News Analysis (prioritize this if African keywords are present)
    african_keywords = ["africa", "african", "east africa", "west africa", "southern africa", "central africa", "north africa"] + \
                       [name for name, code in [(cn.lower(), cc) for cn, cc in [
                            ("nigeria", "ng"), ("south africa", "za"), ("kenya", "ke"), ("egypt", "eg"),
                            ("ghana", "gh"), ("tanzania", "tz"), ("angola", "ao"), ("algeria", "dz"),
                            ("dr congo", "cd"), ("ethiopia", "et"), ("morocco", "ma"), ("sudan", "sd"),
                            ("uganda", "ug"), ("zimbabwe", "zw"), ("senegal", "sn"), ("cameroon", "cm"),
                            ("mozambique", "mz"), ("ivory coast", "ci"), ("burkina faso", "bf"),
                            ("mali", "ml"), ("niger", "ne"), ("rwanda", "rw"), ("somalia", "so"),
                            ("zambia", "zm"), ("tunisia", "tn"), ("libya", "ly"), ("togo", "tg"),
                            ("benin", "bj"), ("gambia", "gm"), ("malawi", "mw"), ("burundi", "bi"),
                            ("republic of the congo", "cg"), ("gabon", "ga"), ("guinea", "gn"),
                            ("eritrea", "er"), ("mauritania", "mr"), ("djibouti", "dj"),
                            ("sierra leone", "sl"), ("liberia", "lr"), ("madagascar", "mg"),
                            ("botswana", "bw"), ("cape verde", "cv"), ("eswatini", "sz"),
                            ("lesotho", "ls"), ("namibia", "na"), ("seychelles", "sc"),
                            ("mauritius", "mu"), ("chad", "td"), ("central african republic", "cf"),
                            ("equatorial guinea", "gq"), ("guinea-bissau", "gw"), ("comoros", "km"),
                            ("sao tome and principe", "st")
                       ]]]
    
    if any(keyword in user_prompt_lower for keyword in african_keywords) and \
       not any(k in user_prompt_lower for k in ["define", "explain", "what is", "meaning of", "fact-check", "verify", "sentiment"]):
        intent_data["intent"] = "african_news_analysis"
        intent_data["query"] = user_prompt # Default to full prompt as query
        
        # Try to extract specific country codes if mentioned
        extracted_country_codes = []
        for country_name, code in [(cn.lower(), cc) for cn, cc in [
            ("nigeria", "ng"), ("south africa", "za"), ("kenya", "ke"), ("egypt", "eg"),
            ("ghana", "gh"), ("tanzania", "tz"), ("angola", "ao"), ("algeria", "dz"),
            ("dr congo", "cd"), ("ethiopia", "et"), ("morocco", "ma"), ("sudan", "sd"),
            ("uganda", "ug"), ("zimbabwe", "zw"), ("senegal", "sn"), ("cameroon", "cm"),
            ("mozambique", "mz"), ("ivory coast", "ci"), ("burkina faso", "bf"),
            ("mali", "ml"), ("niger", "ne"), ("rwanda", "rw"), ("somalia", "so"),
            ("zambia", "zm"), ("tunisia", "tn"), ("libya", "ly"), ("togo", "tg"),
            ("benin", "bj"), ("gambia", "gm"), ("malawi", "mw"), ("burundi", "bi"),
            ("republic of the congo", "cg"), ("gabon", "ga"), ("guinea", "gn"),
            ("eritrea", "er"), ("mauritania", "mr"), ("djibouti", "dj"),
            ("sierra leone", "sl"), ("liberia", "lr"), ("madagascar", "mg"),
            ("botswana", "bw"), ("cape verde", "cv"), ("eswatini", "sz"),
            ("lesotho", "ls"), ("namibia", "na"), ("seychelles", "sc"),
            ("mauritius", "mu"), ("chad", "td"), ("central african republic", "cf"),
            ("equatorial guinea", "gq"), ("guinea-bissau", "gw"), ("comoros", "km"),
            ("sao tome and principe", "st")
        ]]:
            if country_name in user_prompt_lower:
                extracted_country_codes.append(code)
        
        if extracted_country_codes:
            intent_data["country_codes"] = list(set(extracted_country_codes)) # Ensure unique codes
            logging.info(f"Rule-based intent: african_news_analysis. Query: '{intent_data['query']}', Countries: {intent_data['country_codes']}")
        else:
            logging.info(f"Rule-based intent: african_news_analysis. Query: '{intent_data['query']}', general Africa focus.")
            intent_data["region"] = "africa" # Indicate general Africa focus

    # Fact-check (must come after query_previous_message and african_news_analysis to allow those to be specific)
    elif intent_data["intent"] == "general_inquiry" and any(keyword in user_prompt_lower for keyword in ["fact-check", "is it true", "verify", "true or false", "confirm", "authenticity", "debunk"]):
        intent_data["intent"] = "fact_check"
        statement_match = re.search(r'(fact-check|is it true|verify|true or false|confirm|authenticity|debunk)\s*(?:that|if)?\s*[:\s]*(.*)', user_prompt_lower, re.IGNORECASE)
        if statement_match and statement_match.group(2).strip():
            intent_data["statement"] = statement_match.group(2).strip()
        else:
            intent_data["statement"] = user_prompt # Fallback to full prompt
        logging.info(f"Rule-based intent: fact_check. Statement: '{intent_data['statement']}'")

    # News Analysis (general)
    elif intent_data["intent"] == "general_inquiry": # Only if not already set by more specific rules
        news_keywords = ["news", "latest", "breaking", "updates", "current events", "happenings", 
                         "recent developments", "what's going on with", "state of", "trends", "headlines",
                         "report on", "articles on", "tell me about the news", "what's new", "developments in",
                         "how is", "what about"]
        
        if any(keyword in user_prompt_lower for keyword in news_keywords):
            intent_data["intent"] = "news_analysis"
            query_parts = []
            for kw in news_keywords:
                match = re.search(r'(?:' + re.escape(kw) + r')\s*(?:about|on|regarding|involving|in)?\s*(.*)', user_prompt_lower, re.IGNORECASE)
                if match and match.group(1).strip():
                    query_parts.append(match.group(1).strip())
            
            if query_parts:
                final_news_query = " ".join(query_parts)
                final_news_query = re.sub(r'^(the\s*)?(latest|breaking|current|recent)\s*(news|updates|headlines|events)\s*(about|on|regarding|involving)?\s*', '', final_news_query, re.IGNORECASE).strip()
                intent_data["query"] = final_news_query or "general news"
            else:
                intent_data["query"] = user_prompt
            logging.info(f"Rule-based intent: news_analysis. Query: '{intent_data['query']}'")

        elif any(entity in user_prompt_lower for entity in ["economy", "politics", "technology", "environment", 
                                                            "climate change", "space exploration", "ai", "finance", 
                                                            "health", "science", "sports", "business", 
                                                            "stock market", "world affairs", "social issues",
                                                            "geopolitics", "human rights", "cybersecurity"]):
            if not any(k in user_prompt_lower for k in ["define", "explain", "what is", "meaning of"]):
                intent_data["intent"] = "news_analysis"
                intent_data["query"] = user_prompt
                logging.info(f"Rule-based inferred intent: news_analysis based on topic/entity. Query: '{intent_data['query']}'")


    # Summarization
    if intent_data["intent"] == "general_inquiry" and any(keyword in user_prompt_lower for keyword in ["summarize", "condense", "main points", "briefly", "tl;dr"]):
        intent_data["intent"] = "summarization"
        if "this" in user_prompt_lower or "previous" in user_prompt_lower and conversation_messages:
            intent_data["query"] = "previous AI response"
        else:
            summarize_query_match = re.search(r'(summarize|condense|main points|briefly|tl;dr)\s*(?:the)?\s*(.*)', user_prompt_lower, re.IGNORECASE)
            if summarize_query_match and summarize_query_match.group(2).strip():
                intent_data["query"] = summarize_query_match.group(2).strip()
            else:
                intent_data["query"] = user_prompt # Fallback
        logging.info(f"Rule-based intent: summarization. Query: '{intent_data['query']}'")

    # Explain Concept
    elif intent_data["intent"] == "general_inquiry" and any(keyword in user_prompt_lower for keyword in ["explain", "define", "what is", "tell me about", "meaning of"]):
        intent_data["intent"] = "explain_concept"
        term_match = re.search(r'(explain|define|what is|tell me about|meaning of)\s*(?:the)?\s*(.*)', user_prompt_lower, re.IGNORECASE)
        if term_match and term_match.group(2).strip():
            intent_data["term"] = term_match.group(2).strip()
        else:
            intent_data["term"] = user_prompt # Fallback
        logging.info(f"Rule-based intent: explain_concept. Term: '{intent_data['term']}'")
    
    # Sentiment Analysis
    elif intent_data["intent"] == "general_inquiry" and any(keyword in user_prompt_lower for keyword in ["sentiment", "mood", "tone", "positive or negative", "how does it feel"]):
        intent_data["intent"] = "sentiment_analysis"
        sentiment_query_match = re.search(r'(sentiment|mood|tone|positive or negative|how does it feel)\s*(?:of|about)?\s*(.*)', user_prompt_lower, re.IGNORECASE)
        if sentiment_query_match and sentiment_query_match.group(2).strip():
            intent_data["text_for_sentiment"] = sentiment_query_match.group(2).strip()
        else:
            intent_data["text_for_sentiment"] = user_prompt # Fallback
        logging.info(f"Rule-based intent: sentiment_analysis. Text: '{intent_data['text_for_sentiment']}'")


    else: # general_inquiry
        # Default for general inquiry is to attempt a broad news search
        intent_data["intent"] = "news_analysis"
        intent_data["query"] = user_prompt


    intent = intent_data.get("intent", "general_inquiry") # Re-set intent if rule-based updated it
    search_query = intent_data.get("query", user_prompt)
    term = intent_data.get("term", user_prompt)
    statement = intent_data.get("statement", user_prompt)
    text_for_sentiment = intent_data.get("text_for_sentiment", user_prompt)
    message_index_to_query = intent_data.get("message_index")
    sub_query_for_message = intent_data.get("sub_query")
    
    # NEW: Variables for African News specific region/country codes
    country_codes_for_africa_news = intent_data.get("country_codes")
    region_focus_for_africa_news = intent_data.get("region") # 'africa'

    # --- Tool Execution Based on Intent ---
    tool_output = ""
    scraped_data = [] # This will now hold aggregated results from APIs and DDGS
    
    # Determine which sources to consider for news/search, applying user preferences
    sources_for_aggregator = preferred_sources if preferred_sources else []
    topics_for_aggregator = preferred_topics if preferred_topics else []

    if intent == "query_previous_message":
        # Find the specific message content from conversation_messages
        target_message = None
        if message_index_to_query and 1 <= message_index_to_query <= len(conversation_messages):
            target_message = conversation_messages[message_index_to_query - 1]
            tool_output = (
                f"User is asking about Message {message_index_to_query}. "
                f"Content of Message {message_index_to_query} ({target_message.sender}): '{target_message.content}'. "
                f"Specific sub-query: '{sub_query_for_message}'."
            )
            # If the sub_query is about sources, try to extract them
            if "source" in sub_query_for_message.lower() and target_message.sources:
                tool_output += "\nSources for this message:\n" + "\n".join([f"- {s.get('title', s.get('url', 'N/A'))} ({s.get('url', 'N/A')})" for s in target_message.sources])
            elif "details" in sub_query_for_message.lower() and target_message.details:
                tool_output += f"\nDetails for this message: {target_message.details}"
            elif "sentiment" in sub_query_for_message.lower() and target_message.sentiment:
                tool_output += f"\nSentiment of this message: {target_message.sentiment}"
            elif "bias" in sub_query_for_message.lower() and target_message.bias:
                tool_output += f"\nBias of this message: {target_message.bias}"
        else:
            tool_output = f"Could not find Message {message_index_to_query} in the current conversation history."
        
        # For 'query_previous_message', we don't necessarily need a web search, but LLM still processes it.
        # Set search_query to the sub_query for LLM to reason on.
        search_query = sub_query_for_message or user_prompt

    elif intent == "african_news_analysis": # NEW INTENT HANDLING
        num_search_results = 7 # Get more results for comprehensive news analysis
        scraped_data = get_aggregated_news(
            search_query, 
            num_results=num_search_results, 
            preferred_sources=sources_for_aggregator, 
            preferred_topics=topics_for_aggregator,
            region_focus='africa' if region_focus_for_africa_news == 'africa' else country_codes_for_africa_news
        )
        if not scraped_data:
            tool_output = "No relevant news articles found from African sources or web search for your query."
        else:
            tool_output_parts = [f"Found {len(scraped_data)} relevant African news articles:"]
            for i, s in enumerate(scraped_data):
                tool_output_parts.append(f"Article {i+1}: {s.get('title', 'Untitled')} (Source: {s.get('source_name', 'N/A')}, URL: {s.get('url', 'N/A')})\nSnippet: {s.get('snippet', 'No snippet available.')}")
            tool_output = "\n\n".join(tool_output_parts)
    
    elif intent == "news_analysis": # Existing general news analysis
        num_search_results = 7
        scraped_data = get_aggregated_news(
            search_query, 
            num_results=num_search_results, 
            preferred_sources=sources_for_aggregator, 
            preferred_topics=topics_for_aggregator,
            region_focus=None # Ensure no specific region focus for general news
        )
        if not scraped_data:
            tool_output = "No relevant news articles found from integrated APIs or web search."
        else:
            tool_output_parts = [f"Found {len(scraped_data)} relevant news articles:"]
            for i, s in enumerate(scraped_data):
                tool_output_parts.append(f"Article {i+1}: {s.get('title', 'Untitled')} (Source: {s.get('source_name', 'N/A')}, URL: {s.get('url', 'N/A')})\nSnippet: {s.get('snippet', 'No snippet available.')}")
            tool_output = "\n\n".join(tool_output_parts)
    
    elif intent == "fact_check":
        fact_check_result = perform_fact_check_tool(statement)
        
        # Analyze sentiment and bias of the statement itself and its evidence
        sentiment_input_text = statement + " " + " ".join([e.get('snippet', '') for e in fact_check_result.get('evidence_list', [])])
        response_sentiment_bias_analysis = analyze_sentiment_tool(sentiment_input_text)
        response_sentiment = response_sentiment_bias_analysis["sentiment"]
        response_bias = response_sentiment_bias_analysis["bias"]

        tool_output = (
            f"STATEMENT: {statement}\n"
            f"VERDICT: {fact_check_result['verdict']}\n"
            f"CONFIDENCE: {fact_check_result['confidence']}\n"
            f"SUMMARY: {fact_check_result['summary']}\n"
            f"EVIDENCE_SOURCES:\n" + "\n".join([
                f"- {e['title']} (Source: {e.get('source_name', 'N/A')}, Type: {e.get('type', 'General')}): {e['snippet']} [Link: {e['url']}]"
                for e in fact_check_result.get('evidence_list', [])
            ])
        )
        # Populate response_sources directly from the structured evidence list
        response_sources = fact_check_result.get('evidence_list', [])
        
    elif intent == "explain_concept":
        definition_result = get_definition_tool(term)
        tool_output = definition_result["content"]
        response_sources = [{"title": definition_result.get('source', 'N/A'), "url": definition_result.get('url', '#'), "snippet": definition_result.get('content', ''), "source_name": definition_result.get('source_name', 'N/A')}]
        
        # Fetch additional web content related to the term for broader context, but only if needed
        if definition_result["source"] == "N/A" or "Web Search" in definition_result["source"]:
            scraped_data = get_aggregated_news(f"what is {term}", num_results=2, preferred_sources=sources_for_aggregator, preferred_topics=topics_for_aggregator, region_focus=None)
            # Add these to response_sources if they provide additional, distinct info
            for s in scraped_data:
                if s['url'] not in [src['url'] for src in response_sources]:
                    response_sources.append({"title": s.get('title', 'Web Search'), "url": s.get('url', '#'), "snippet": s.get('snippet', ''), "source_name": s.get('source_name', 'N/A')})
    
    elif intent == "summarization":
        content_to_summarize = []
        if search_query != user_prompt and search_query != "previous AI response":
            scraped_data = get_aggregated_news(search_query, num_results=3, preferred_sources=sources_for_aggregator, preferred_topics=topics_for_aggregator, region_focus=None)
            content_to_summarize = [s['full_content'] for s in scraped_data if s.get('full_content')]
        elif conversation_messages and conversation_messages[-1].sender == 'ai':
            # Summarize the content of the *previous AI message*
            content_to_summarize = [conversation_messages[-1].content]
            # Inherit sources from the message being summarized
            response_sources = conversation_messages[-1].sources
        
        if content_to_summarize:
            tool_output = summarize_content_with_llm(content_to_summarize, analysis_depth)
        else:
            tool_output = "No specific content to summarize. Please provide text, a topic, or refer to a previous AI response (e.g., 'summarize Message 3')."
        
    elif intent == "sentiment_analysis":
        text_for_analysis = ""
        # If user refers to a previous message
        if "previous" in user_prompt_lower and conversation_messages and conversation_messages[-1].sender == 'ai':
            text_for_analysis = conversation_messages[-1].content
        elif text_for_sentiment and text_for_sentiment != user_prompt: # If specific text provided by LLM intent
            text_for_analysis = text_for_sentiment
        else: # Analyze sentiment of the search results for the original prompt
            scraped_data = get_aggregated_news(search_query, num_results=3, preferred_sources=sources_for_aggregator, preferred_topics=topics_for_aggregator, region_focus=None)
            text_for_analysis = " ".join([s.get('full_content', '') for s in scraped_data if s.get('full_content')])
            response_sources = [
                {"title": s.get('title', 'Untitled'), "url": s.get('url', '#'), "snippet": s.get('snippet', ''), "image_url": s.get('image_url', None), "source_name": s.get('source_name', 'N/A')}
                for s in scraped_data
            ]
        
        if text_for_analysis:
            sentiment_bias_analysis = analyze_sentiment_tool(text_for_analysis)
            response_sentiment = sentiment_bias_analysis["sentiment"]
            response_bias = sentiment_bias_analysis["bias"]
            tool_output = f"Analyzed Sentiment: {response_sentiment}\nAnalyzed Bias: {response_bias}\n\nContent analyzed (excerpt):\n{text_for_analysis[:500]}..." # Show excerpt for context
        else:
            tool_output = "No sufficient content found for sentiment and bias analysis. Please provide more text or a topic."


    web_context = ""
    if scraped_data:
        # Max content for LLM from web search to avoid overflow
        MAX_WEB_CONTEXT_LENGTH = 10000 
        current_web_context_length = 0
        for idx, source in enumerate(scraped_data):
            source_content = source.get('full_content', source.get('snippet', ''))
            
            # Estimate length for this source and add if it fits
            if current_web_context_length + len(source_content) + 200 < MAX_WEB_CONTEXT_LENGTH:
                web_context += f"--- Source {idx+1}: {source.get('title', 'Untitled')}\n"
                web_context += f"URL: {source.get('url', 'N/A')}\n"
                web_context += f"Content: {source_content}\n---\n\n"
                current_web_context_length += len(source_content) + 200
            else:
                logging.warning(f"Truncated web context for LLM. Skipping source: {source.get('title')}")
                break # Stop adding more sources if context budget is met
        
        # Populate response_sources if not already populated by a specific tool (e.g., fact-check)
        if not response_sources:
            # Ensure response_sources contains only the most relevant articles actually used or found
            # Limit to a reasonable number to avoid cluttering the UI, e.g., max 5
            response_sources = [
                {"title": s.get('title', 'Untitled'), "url": s.get('url', '#'), "snippet": s.get('snippet', ''), "image_url": s.get('image_url', None), "source_name": s.get('source_name', 'N/A')}
                for s in scraped_data[:min(len(scraped_data), 5)]
            ]
        
        # Re-analyze sentiment and bias of the aggregated web content if it exists
        if (not response_sentiment or not response_bias) and web_context: # Only if not already set by a specific tool AND context exists
            sentiment_bias_analysis_for_context = analyze_sentiment_tool(web_context[:3000]) # Analyze first 3KB of context
            response_sentiment = sentiment_bias_analysis_for_context["sentiment"]
            response_bias = sentiment_bias_analysis_for_context["bias"]
    else:
        # If no web content, base sentiment/bias only on user prompt
        sentiment_bias_analysis_for_prompt = analyze_sentiment_tool(user_prompt)
        response_sentiment = sentiment_bias_analysis_for_prompt["sentiment"]
        response_bias = sentiment_bias_analysis_for_prompt["bias"]


    # --- LLM Call for intelligent response generation ---
    if llm_pipeline or llm_model: # Use either open-source or Gemini
        try:
            full_context_for_llm = ""
            if tool_output:
                full_context_for_llm += f"--- Tool Output ---\n{tool_output}\n\n"
            if web_context:
                full_context_for_llm += f"--- Web Search Context ---\n{web_context}"

            final_user_prompt_for_llm = f"Based on the following context, please respond to my query: \"{user_prompt}\"\n\nContext:\n{full_context_for_llm}"
            
            # Construct the full conversation for the LLM.
            # LLM input format depends on the model (Gemini uses 'parts', HF uses chat templates)
            
            if llm_pipeline: # Using open-source LLM
                messages_for_llm_chat_template = []
                messages_for_llm_chat_template.append({"role": "system", "content": llm_system_instruction}) # System instruction

                for msg_dict in llm_history: # Conversation history
                    content_str = msg_dict['parts'][0]['text'] if msg_dict['parts'] else ""
                    messages_for_llm_chat_template.append({"role": msg_dict['role'], "content": content_str})

                messages_for_llm_chat_template.append({"role": "user", "content": final_user_prompt_for_llm}) # Current prompt
                
                full_prompt_string = llm_pipeline.tokenizer.apply_chat_template(messages_for_llm_chat_template, tokenize=False, add_generation_prompt=True)
                
                logging.info(f"Sending prompt to open-source LLM. Full prompt length: {len(full_prompt_string)} bytes.")
                
                llm_response_output = llm_pipeline(full_prompt_string)
                generated_text_full = llm_response_output[0]['generated_text']
                
                if full_prompt_string in generated_text_full:
                    response_content = generated_text_full.replace(full_prompt_string, "").strip()
                else:
                    response_content = generated_text_full.strip()

                logging.info(f"Received open-source LLM response (first 200 chars): {response_content[:200]}...")
                response_details = "Analysis provided by advanced open-source AI and web/API search."

            elif llm_model: # Using Gemini LLM
                full_llm_conversation = [
                    {"role": "user", "parts": [{"text": llm_system_instruction}]}, # Initial system instruction (as user to start chat)
                ] + llm_history + [ # Formatted conversation history
                    {"role": "user", "parts": [{"text": final_user_prompt_for_llm}]} # Current user prompt with context
                ]
                
                logging.info(f"Sending prompt to Gemini LLM. History length: {len(full_llm_conversation)}. Total context size: {len(full_context_for_llm)} bytes.")

                llm_response_obj = llm_model.generate_content(full_llm_conversation)
                
                response_content = llm_response_obj.text.strip()
                logging.info(f"Received Gemini LLM response (first 200 chars): {response_content[:200]}...")
                
                response_details = "Analysis provided by advanced AI (Gemini Flash) and web/API search."

        except Exception as e:
            logging.error(f"Error calling LLM API: {e}", exc_info=True)
            response_details = "AI analysis fallback (LLM error)."
            response_content = "I'm sorry, I couldn't process your request with my advanced AI at the moment. This might be due to a temporary issue with my AI models.\n\n"
            response_content += f"However, based on my web search for '{search_query}', here are some relevant snippets:\n\n"
            if scraped_data:
                for i, res in enumerate(scraped_data[:3]):
                    response_content += f"**{res.get('title', 'N/A')}**: {res.get('snippet', 'No snippet available.')} ([Source]({res.get('url', '#')}))\n\n"
            else:
                response_content += "No relevant search results were found for your query either."
            response_content += "\n\nPlease try again later or rephrase your question. I'm always learning!"

    else: # Fallback if no LLM (Gemini or Open-Source) is configured
        logging.warning("No LLM configured. Falling back to rule-based summary.")
        response_details = "Rule-based analysis (LLM unavailable)."
        
        response_content = f"I'm currently operating without my advanced AI models configured. "
        response_content += f"Based on my web search for '{search_query}', here's what I found:\n\n"
        
        if scraped_data:
            for i, res in enumerate(scraped_data[:min(len(scraped_data), 5)]):
                response_content += f"**{res.get('title', 'N/A')}**: {res.get('snippet', 'No snippet available.')} ([Source]({res.get('url', '#')}))\n\n"
            response_content += "\nFor more details, please check the original sources.\n\n"
        else:
            response_content += "No relevant search results were found for your query."
        
        response_content += "To unlock more in-depth analysis and complex query handling, please ensure a Gemini API key or an open-source LLM is configured."
        response_content += "\n\nWhat else can I help you with today?"
    
    # Ensure the final response contains a natural closing sentence if LLM didn't provide one
    if (llm_pipeline or llm_model) and not re.search(r'[.?!]$', response_content.strip()):
         response_content += " What else can I help you with today?"
    
    return {
        "response": response_content,
        "details": response_details,
        "sources": response_sources,
        "sentiment": response_sentiment,
        "bias": response_bias
    }

# --- Background Task Placeholder (Conceptual) ---
# In a real-world application, this would be a separate process managed by Celery,
# RabbitMQ, or a similar asynchronous task queue system.
def start_background_news_monitor():
    """
    Simulates a background task that could periodically monitor news
    for user-defined topics or send alerts.
    """
    logging.info("Starting conceptual background news monitor (not actively running in monolithic setup).")
    # while True:
    #     logging.info("Background monitor checking for news updates...")
    #     # Example: Fetch news for popular topics or user preferred topics
    #     # news_articles = get_aggregated_news("latest AI news", num_results=1)
    #     # if news_articles:
    #     #     logging.info(f"Found new AI article: {news_articles[0]['title']}")
    #     # Simulate work
    #     # time.sleep(3600) # Check every hour

# --- Flask Routes (Simulated Blueprints) ---
# In a real-world application, these could be organized into Flask Blueprints.

# --- General Routes ---
@app.route('/')
def index():
    """Renders the main chat application page."""
    user = None
    show_auth_modal = False
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if not user:
        show_auth_modal = True # Show modal if no user logged in (guest mode)

    # Get avatar options for registration/settings forms - FIX for TypeError
    avatars_path = os.path.join(app.root_path, 'static', 'avatars')
    # List .png files only and ensure paths are generated correctly by url_for
    avatar_filenames = [f for f in os.listdir(avatars_path) if f.endswith('.png')]
    avatar_filenames.sort() # Ensure consistent order
    avatar_options = []
    for f in avatar_filenames:
        try:
            avatar_options.append(url_for('static', filename=f'avatars/{f}'))
        except Exception as e:
            logging.error(f"Error generating URL for avatar {f}: {e}")
            # Optionally add a fallback URL or skip this avatar

    return render_template('index.html',
        avatar_options=avatar_options, # Pass the cleaned list
        current_user=user,
        show_auth_modal=show_auth_modal
    )

@app.route('/user_info', methods=['GET'])
def get_user_info():
    """Returns current user's login status and data, including guest prompt count."""
    if not g.user:
        logging.info(f"/user_info: User not logged in. Guest prompt count: {session.get('unauth_prompt_count', 0)}")
        return jsonify({
            "is_logged_in": False,
            "unauth_prompt_count": session.get('unauth_prompt_count', 0),
            "current_conversation_id": None
        }), 200
    
    logging.info(f"/user_info: User {g.user.username} (ID: {g.user.id}) is logged in.")
    return jsonify({
        "is_logged_in": True,
        "user": g.user.to_dict()
    }), 200

# --- Authentication Routes ---
@app.route('/register', methods=['POST'])
@limiter.limit("5 per hour", error_message="Too many registration attempts. Please try again later.")
def register():
    """Handles user registration."""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    avatar_url = data.get('avatar_url')
    theme = data.get('theme', 'light')

    # Basic input validation
    if not all([username, email, password, avatar_url]):
        return jsonify({"success": False, "message": "All fields are required for registration."}), 400
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"success": False, "message": "Invalid email address format."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters long."}), 400

    if User.query.filter_by(username=username).first():
        logging.warning(f"Registration attempt for existing username: {username}")
        return jsonify({"success": False, "message": "Username already exists. Please choose a different one."}), 409
    if User.query.filter_by(email=email).first():
        logging.warning(f"Registration attempt for existing email: {email}")
        return jsonify({"success": False, "message": "Email already exists. Please use a different email or login."}), 409

    new_user = User(username=username, email=email, avatar_url=avatar_url, theme=theme)
    new_user.set_password(password)
    verification_token = new_user.generate_email_verification_token()
    db.session.add(new_user)
    db.session.commit()

    # Send verification email (simulated)
    verification_link = url_for('verify_email', token=verification_token, _external=True)
    email_body = f"Hello {username},\n\nPlease click on the following link to verify your email address for NexusAI: {verification_link}\n\nThis link is valid for 24 hours.\n\nIf you did not register for NexusAI, please ignore this email.\n\nThank you,\nNexusAI Team"
    if send_email(email, "Verify your NexusAI Email", email_body):
        logging.info(f"Verification email sent to {email}.")
    else:
        logging.error(f"Failed to send verification email to {email}.")

    session['user_id'] = new_user.id
    logging.info(f"New user registered: {new_user.username} (ID: {new_user.id}). Email verification pending.")
    return jsonify({"success": True, "message": "Registration successful! Please check your email to verify your account.", "user": new_user.to_dict()}), 201

@app.route('/login', methods=['POST'])
@limiter.limit("10 per hour", error_message="Too many login attempts. Please try again later.")
def login():
    """Handles user login."""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if user is None or not user.check_password(password):
        logging.warning(f"Failed login attempt for username: {username}")
        return jsonify({"success": False, "message": "Invalid username or password. Please try again."}), 401
    
    if not user.email_verified:
        logging.warning(f"Login attempt for unverified email for user: {username}")
        return jsonify({"success": False, "message": "Please verify your email address before logging in. Check your inbox for a verification link."}), 401

    session['user_id'] = user.id
    session.pop('unauth_prompt_count', None) # Clear guest session data on login
    session.pop('unauth_conversation_messages', None)
    session.modified = True

    logging.info(f"User logged in: {user.username} (ID: {user.id}).")
    return jsonify({"success": True, "message": "Login successful!", "user": user.to_dict()}), 200

# NEW: Google OAuth 2.0 Login Endpoint
@app.route('/auth/google', methods=['POST'])
@limiter.limit("5 per minute") # Basic rate limiting for auth attempts
def google_login():
    """
    Handles Google ID token verification and user login/registration.
    Receives the ID token from the frontend.
    """
    data = request.get_json()
    id_token_str = data.get('id_token')

    if not id_token_str:
        return jsonify({"success": False, "message": "No ID token provided."}), 400

    if not GOOGLE_CLIENT_ID:
        logging.error("GOOGLE_CLIENT_ID environment variable not set. Cannot verify Google ID token.")
        return jsonify({"success": False, "message": "Server configuration error: Google authentication not set up."}), 500

    try:
        # Verify the ID token against your Google Client ID.
        # This confirms the token is valid and issued for your application.
        info = id_token.verify_oauth2_token(id_token_str, google_requests.Request(), GOOGLE_CLIENT_ID)

        # ID token is valid. Extract user information.
        google_user_id = info['sub'] # Unique Google ID for the user
        email = info['email']
        name = info.get('name', email.split('@')[0]) # Use name if available, else part of email
        avatar_url = info.get('picture', '/static/avatars/default.png') # Google profile picture

        user = User.query.filter_by(email=email).first()

        if user:
            # Existing user, log them in
            session['user_id'] = user.id
            session.pop('unauth_prompt_count', None) # Clear guest session data on login
            session.pop('unauth_conversation_messages', None)
            session.modified = True
            logging.info(f"User {user.username} logged in via Google (Existing account).")
            return jsonify({"success": True, "message": "Logged in successfully via Google."}), 200
        else:
            # New user, register them
            # Generate a unique username based on name or email
            username_base = re.sub(r'[^a-zA-Z0-9]', '', name.lower()) if name else email.split('@')[0]
            username = username_base
            counter = 1
            while User.query.filter_by(username=username).first():
                username = f"{username_base}{counter}"
                counter += 1

            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(str(uuid4())), # Create a dummy password, as Google handles authentication
                avatar_url=avatar_url,
                email_verified=True, # Email is verified by Google
                role=UserRole.STANDARD # Assign a default role
            )
            db.session.add(new_user)
            db.session.commit()

            session['user_id'] = new_user.id
            session.pop('unauth_prompt_count', None) # Clear guest session data
            session.pop('unauth_conversation_messages', None)
            session.modified = True
            logging.info(f"New user {new_user.username} registered and logged in via Google.")
            return jsonify({"success": True, "message": "Registered and logged in successfully via Google."}), 201

    except ValueError as e:
        # Invalid token (e.g., token expired, wrong client ID, corrupted token)
        logging.error(f"Google ID token verification failed: {e}")
        return jsonify({"success": False, "message": "Invalid Google ID token. Please try again."}), 401
    except Exception as e:
        # Catch any other unexpected errors during the process
        logging.error(f"An unexpected error occurred during Google login: {e}", exc_info=True)
        return jsonify({"success": False, "message": "An internal error occurred during Google login."}), 500


@app.route('/logout', methods=['POST'])
def logout():
    """Logs out the current user by clearing session data."""
    user_id = session.pop('user_id', None)
    session.pop('unauth_prompt_count', None)
    session.pop('unauth_conversation_messages', None)
    session.modified = True # Ensure session changes are saved
    if user_id:
        logging.info(f"User ID {user_id} logged out. Session cleared.")
    return jsonify({"success": True, "message": "Logged out successfully."}), 200

@app.route('/verify_email/<token>')
def verify_email(token):
    """Handles email verification via a token."""
    user = User.query.filter_by(email_verification_token=token).first()
    if user and not user.email_verified:
        user.email_verified = True
        user.email_verification_token = None # Clear token after use
        db.session.commit()
        logging.info(f"Email verified for user: {user.username}")
        # Redirect to a success page or login page
        return render_template('verification_success.html', message="Your email has been successfully verified! You can now log in.")
    elif user and user.email_verified:
        logging.warning(f"Attempted to verify already verified email for user: {user.username}")
        return render_template('verification_info.html', message="Your email is already verified. You can log in.")
    else:
        logging.warning(f"Invalid or expired email verification token: {token}")
        return render_template('verification_error.html', message="Invalid or expired verification link.")

@app.route('/forgot_password', methods=['POST'])
@limiter.limit("3 per hour", error_message="Too many password reset requests. Please try again later.")
def forgot_password():
    """Handles initiating a password reset by sending a token to the user's email."""
    data = request.json
    email = data.get('email')

    if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"success": False, "message": "Please provide a valid email address."}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        # For security, always respond as if an email was sent, even if user doesn't exist.
        logging.warning(f"Password reset requested for non-existent email: {email}")
        return jsonify({"success": True, "message": "If an account with that email exists, a password reset link has been sent."}), 200
    
    if not user.email_verified:
        logging.warning(f"Password reset requested for unverified email: {email}")
        return jsonify({"success": False, "message": "Your email address is not verified. Please verify it first or register."}), 403

    token = user.generate_password_reset_token()
    db.session.commit()

    reset_link = url_for('reset_password_page', token=token, _external=True)
    email_body = f"Hello {user.username},\n\nYou have requested to reset your password for NexusAI. Please click on the following link within 1 hour: {reset_link}\n\nIf you did not request a password reset, please ignore this email.\n\nThank you,\nNexusAI Team"
    if send_email(user.email, "NexusAI Password Reset Request", email_body):
        logging.info(f"Password reset email sent to {user.email}.")
        return jsonify({"success": True, "message": "If an account with that email exists, a password reset link has been sent."}), 200
    else:
        logging.error(f"Failed to send password reset email to {user.email}.")
        return jsonify({"success": False, "message": "Failed to send password reset email. Please try again later."}), 500

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password_page(token):
    """
    Renders the password reset form (GET) or handles password reset (POST).
    """
    user = User.query.filter_by(password_reset_token=token).first()

    if not user or user.password_reset_expires_at < datetime.utcnow():
        logging.warning(f"Invalid or expired password reset token: {token}")
        return render_template('reset_password_error.html', message="Invalid or expired password reset link. Please request a new one."), 400

    if request.method == 'GET':
        return render_template('reset_password.html', token=token)
    elif request.method == 'POST':
        data = request.json
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')

        if not new_password or new_password != confirm_password or len(new_password) < 6:
            return jsonify({"success": False, "message": "Passwords do not match or are too short (min 6 characters)."}), 400

        user.set_password(new_password)
        user.password_reset_token = None
        user.password_reset_expires_at = None
        db.session.commit()
        logging.info(f"Password reset successfully for user: {user.username}")
        return jsonify({"success": True, "message": "Your password has been reset successfully!"}), 200

# --- Chat Functionality Routes ---
@app.route('/chat', methods=['POST'])
@limiter.limit("60 per minute") # Limit chat message frequency
def chat():
    """
    Handles user chat messages, processes them with AI, and manages conversation history.
    Supports both logged-in users and limited guest mode.
    """
    user_prompt = request.json.get('prompt', '').strip()
    conversation_id = request.json.get('conversation_id')

    if not user_prompt:
        return jsonify({"error": "No prompt provided."}), 400

    # --- Handle Unauthenticated User (Guest) Prompt Limit ---
    if g.user is None:
        session['unauth_prompt_count'] += 1
        guest_limit = int(os.getenv('GUEST_PROMPT_LIMIT', 5)) # Configurable guest limit
        if session['unauth_prompt_count'] > guest_limit:
            logging.info(f"Guest user exceeded prompt limit: {session['unauth_prompt_count']}/{guest_limit}")
            return jsonify({"disable_input": True, "message": f"You've reached your chat limit as a guest ({guest_limit} prompts). Please sign in or sign up to continue chatting!", "unauth_prompt_count": session['unauth_prompt_count']}), 403
        
        # In guest mode, 'conversation_messages' comes from session, not DB
        # Convert raw dicts from session into Message-like objects for format_conversation_history
        conversation_messages_for_ai = []
        # Guest messages stored with 'sender' and 'content', plus any AI additions like 'details', 'sources', 'sentiment'
        for msg_data in session['unauth_conversation_messages']:
            temp_msg = Message(
                sender=msg_data.get('sender'), 
                content=msg_data.get('content'),
                details=msg_data.get('details'),
                sources=msg_data.get('sources', []), # Directly use the list, setter will handle JSON dumping
                sentiment=msg_data.get('sentiment'),
                bias=msg_data.get('bias')
            )
            conversation_messages_for_ai.append(temp_msg)
        
        # Guest users get default preferences
        ai_response_data = get_ai_response(user_prompt, conversation_messages=conversation_messages_for_ai, 
                                           current_user_role=UserRole.GUEST, preferred_sources=[], analysis_depth='standard', preferred_topics=[])
        
        # Append user message to guest session history
        session['unauth_conversation_messages'].append({'sender': 'user', 'content': user_prompt})
        # Append AI message to guest session history
        session['unauth_conversation_messages'].append({
            'sender': 'ai', 
            'content': ai_response_data['response'], # Content is Markdown, do NOT escape
            'details': html.escape(ai_response_data['details']), # Details is plain text, escape
            'sources': ai_response_data['sources'], # Sources is already a list of dicts, no escaping needed
            'sentiment': ai_response_data['sentiment'],
            'bias': ai_response_data['bias']
        })
        session.modified = True

        response_payload = {
            "response": ai_response_data['response'], # Content is Markdown, do NOT escape
            "details": html.escape(ai_response_data['details']), # Details is plain text, escape
            "sources": ai_response_data['sources'], # Sources is already a list of dicts
            "sentiment": ai_response_data['sentiment'],
            "bias": ai_response_data['bias'],
            "conversation_id": None, # No DB conversation for guests
            "is_logged_in": False,
            "unauth_prompt_count": session['unauth_prompt_count']
        }
        return jsonify(response_payload), 200

    # --- Handle Logged-in User Conversations ---
    current_conversation = None
    new_conversation_flag = False

    if conversation_id:
        current_conversation = Conversation.query.filter_by(id=conversation_id, user_id=g.user.id).first()
    
    if not current_conversation:
        new_conversation_flag = True
        title_prefix = "Chat about: "
        title_suffix = user_prompt[:50] + "..." if len(user_prompt) > 50 else user_prompt
        if not title_suffix.strip():
            title_suffix = f"New Chat ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        
        current_conversation = Conversation(user=g.user, title=title_prefix + title_suffix)
        db.session.add(current_conversation)
        db.session.flush() # Get ID for new conversation before commit
        conversation_id = current_conversation.id
        logging.info(f"New conversation created for user {g.user.username}: ID {conversation_id}, Title: '{current_conversation.title}'")

    g.user.current_conversation_id = current_conversation.id
    db.session.add(g.user)
    db.session.commit() # Commit g.user and current_conversation before adding messages

    # Add user message to DB
    user_message_obj = Message(conversation=current_conversation, sender='user', content=html.escape(user_prompt))
    db.session.add(user_message_obj)
    db.session.commit() # Commit user message

    conversation_messages_for_ai = current_conversation.messages # Fetch updated messages including the new user message
    
    # Pass user's role and preferences to AI for model selection/feature access
    ai_response_data = get_ai_response(
        user_prompt,
        conversation_messages=conversation_messages_for_ai, # Pass full conversation history
        current_user_role=g.user.role,
        preferred_sources=g.user.preferred_sources, # Pass preferred sources from user profile
        analysis_depth=g.user.analysis_depth, # Pass analysis depth from user profile
        preferred_topics=g.user.preferred_topics # Pass preferred topics from user profile
    )
    
    # Add AI message to DB
    ai_message_obj = Message(
        conversation=current_conversation, 
        sender='ai', 
        content=ai_response_data['response'], # Content is Markdown, do NOT escape
        details=html.escape(ai_response_data['details']), # Details is plain text, escape
        sources=ai_response_data['sources'], # Sources is already a list of dicts, no escaping needed
        sentiment=ai_response_data['sentiment'],
        bias=ai_response_data['bias']
    )
    db.session.add(ai_message_obj)
    
    current_conversation.last_updated = datetime.utcnow() # Update conversation timestamp
    db.session.add(current_conversation)

    db.session.commit() # Final commit for AI message and conversation update
    cache.delete_memoized(get_conversations) # Invalidate cache for conversations list to show new title/update

    # Emit message to client via WebSocket for real-time update
    socketio.emit('receive_message', {
        "conversation_id": current_conversation.id,
        "message": ai_message_obj.to_dict()
    }, room=f'user_{g.user.id}') # Emit to specific user's room

    return jsonify({
        "response": ai_response_data['response'], # Content is Markdown, do NOT escape
        "details": html.escape(ai_response_data['details']), # Details is plain text, escape
        "sources": ai_response_data['sources'], # Sources is already a list of dicts
        "sentiment": ai_response_data['sentiment'],
        "bias": ai_response_data['bias'],
        "conversation_id": current_conversation.id,
        "new_conversation_title": current_conversation.title if new_conversation_flag else None,
        "is_logged_in": True
    }), 200

@app.route('/conversations', methods=['GET'])
@role_required(UserRole.STANDARD) # Only logged-in users can get conversations
@cache.cached(timeout=60) # Cache conversation list for 1 minute to reduce DB hits
def get_conversations():
    """Returns a list of all conversations for the current user, ordered by last updated."""
    conversations = Conversation.query.filter_by(user_id=g.user.id).order_by(Conversation.last_updated.desc()).all()
    logging.info(f"Fetched {len(conversations)} conversations for user {g.user.username}.")
    return jsonify({"conversations": [conv.to_dict() for conv in conversations]}), 200

@app.route('/conversations/<int:conversation_id>/messages', methods=['GET'])
@role_required(UserRole.STANDARD) # Only logged-in users can get messages
def get_conversation_messages(conversation_id):
    """Returns all messages for a specific conversation belonging to the current user."""
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=g.user.id).first()
    if not conversation:
        logging.warning(f"Conversation {conversation_id} not found or unauthorized access for user {g.user.username}.")
        return jsonify({"error": "Conversation not found or unauthorized access"}), 404
    
    # Order messages by timestamp for correct chronological display
    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.timestamp.asc()).all()
    logging.info(f"Fetched {len(messages)} messages for conversation {conversation_id} (user {g.user.username}).")
    return jsonify({"messages": [msg.to_dict() for msg in messages], "title": conversation.title}), 200

@app.route('/conversations/<int:conversation_id>/delete', methods=['DELETE'])
@role_required(UserRole.STANDARD)
def delete_conversation(conversation_id):
    """Deletes a specific conversation and all its associated messages."""
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=g.user.id).first()
    if not conversation:
        return jsonify({"success": False, "message": "Conversation not found or unauthorized access"}), 404
    
    if g.user.current_conversation_id == conversation_id:
        g.user.current_conversation_id = None
        db.session.add(g.user) # Update user's current_conversation_id if deleting active chat
    
    db.session.delete(conversation)
    db.session.commit()
    cache.delete_memoized(get_conversations) # Invalidate cache
    logging.info(f"Conversation {conversation_id} deleted for user {g.user.username}")
    return jsonify({"success": True, "message": "Conversation deleted successfully."}), 200

@app.route('/conversations/<int:conversation_id>/edit_title', methods=['POST'])
@role_required(UserRole.STANDARD)
def edit_conversation_title(conversation_id):
    """Allows authenticated users to edit the title of their conversation."""
    data = request.json
    new_title = data.get('title', '').strip()

    if not new_title:
        return jsonify({"success": False, "message": "Conversation title cannot be empty."}), 400
    if len(new_title) > 255: # Max length for SQLAlchemy String
        return jsonify({"success": False, "message": "Conversation title is too long (max 255 characters)."}), 400


    conversation = Conversation.query.filter_by(id=conversation_id, user_id=g.user.id).first()
    if not conversation:
        return jsonify({"success": False, "message": "Conversation not found or unauthorized access"}), 404

    conversation.title = new_title
    conversation.last_updated = datetime.utcnow() # Update timestamp as well
    db.session.commit()
    cache.delete_memoized(get_conversations) # Invalidate cache
    logging.info(f"Conversation {conversation_id} title updated to '{new_title}' by user {g.user.username}.")
    return jsonify({"success": True, "message": "Conversation title updated.", "new_title": new_title}), 200


# --- User Settings Routes ---
@app.route('/settings', methods=['GET'])
@role_required(UserRole.STANDARD) # Only logged-in users can access settings
def settings_page():
    """Renders the settings page."""
    # Provide avatar options for settings page
    avatars_path = os.path.join(app.root_path, 'static', 'avatars')
    # List .png files only and ensure paths are generated correctly by url_for
    avatar_filenames = [f for f in os.listdir(avatars_path) if f.endswith('.png')]
    avatar_filenames.sort() # Ensure consistent order
    avatar_options = []
    for f in avatar_filenames:
        try:
            avatar_options.append(url_for('static', filename=f'avatars/{f}'))
        except Exception as e:
            logging.error(f"Error generating URL for avatar {f}: {e}")
            # Optionally add a fallback URL or skip this avatar

    logging.info(f"Rendering settings page for user {g.user.username}.")
    return render_template('settings.html', current_user=g.user, avatar_options=avatar_options), 200

@app.route('/update_user', methods=['POST'])
@role_required(UserRole.STANDARD)
def update_user():
    """Allows authenticated users to update their profile information."""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    avatar_url = data.get('avatar_url')
    theme = data.get('theme') # Get theme from request

    if not all([username, email, avatar_url]):
        return jsonify({"success": False, "message": "All fields are required."}), 400
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"success": False, "message": "Invalid email address format."}), 400

    if User.query.filter(User.username == username, User.id != g.user.id).first():
        logging.warning(f"User update failed: username '{username}' already taken.")
        return jsonify({"success": False, "message": "Username already taken."}), 409
    if User.query.filter(User.email == email, User.id != g.user.id).first():
        logging.warning(f"User update failed: email '{email}' already taken.")
        return jsonify({"success": False, "message": "Email already taken."}), 409
    
    # If email is changed, set email_verified to False and generate new token
    if g.user.email != email:
        g.user.email = email
        g.user.email_verified = False
        new_verification_token = g.user.generate_email_verification_token()
        db.session.flush() # Ensure user object is updated for url_for to work
        verification_link = url_for('verify_email', token=new_verification_token, _external=True)
        email_body = f"Hello {username},\n\nYour email for NexusAI has been changed. Please click on the following link to verify your new email address: {verification_link}\n\nThis link is valid for 24 hours.\n\nIf you did not make this change, please contact support.\n\nThank you,\nNexusAI Team"
        if send_email(email, "Verify your new NexusAI Email", email_body):
            logging.info(f"New verification email sent to {email} due to email change.")
            db.session.commit() # Commit after email sent
            return jsonify({"success": True, "message": "Profile updated! A new verification email has been sent to your new address. Please verify it.", "user": g.user.to_dict()}), 200
        else:
            logging.error(f"Failed to send new verification email to {email}.")
            db.session.rollback() # Rollback if email sending failed to prevent partial update
            return jsonify({"success": False, "message": "Profile updated, but failed to send new verification email. Please contact support.", "user": g.user.to_dict()}), 500

    g.user.username = username
    g.user.avatar_url = avatar_url
    if theme: # Update theme if provided in the update_user call
        g.user.theme = theme
    db.session.commit()
    logging.info(f"User {g.user.id} updated their profile (username: {g.user.username}, theme: {g.user.theme}).")
    return jsonify({"success": True, "message": "Profile updated successfully!", "user": g.user.to_dict()}), 200

@app.route('/change_password', methods=['POST'])
@role_required(UserRole.STANDARD)
def change_password():
    """Allows authenticated users to change their password."""
    data = request.json
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not current_password or not new_password:
        return jsonify({"success": False, "message": "Current password and new password are required."}), 400

    if not g.user.check_password(current_password):
        logging.warning(f"User {g.user.username} failed to change password: Incorrect current password.")
        return jsonify({"success": False, "message": "Incorrect current password."}), 401
    
    if len(new_password) < 6:
        return jsonify({"success": False, "message": "New password must be at least 6 characters long."}), 400

    g.user.set_password(new_password)
    db.session.commit()
    logging.info(f"User {g.user.id} changed their password.")
    return jsonify({"success": True, "message": "Password changed successfully!"}), 200

@app.route('/update_theme', methods=['POST'])
@role_required(UserRole.STANDARD)
def update_theme():
    """Allows authenticated users to update their theme preference."""
    data = request.json
    theme = data.get('theme')

    if not theme or theme not in ['light', 'dark']:
        return jsonify({"success": False, "message": "Invalid theme provided."}), 400

    g.user.theme = theme
    db.session.commit()
    logging.info(f"User {g.user.id} updated theme to {g.user.theme}.")
    return jsonify({"success": True, "message": "Theme updated successfully!", "theme": g.user.theme}), 200

# --- Admin Routes (Requires ADMIN role) ---
@app.route('/admin', methods=['GET'])
@role_required(UserRole.ADMIN)
def admin_dashboard():
    """Renders a basic admin dashboard."""
    logging.info(f"Admin dashboard accessed by {g.user.username}")
    return render_template('admin_dashboard.html', current_user=g.user), 200

@app.route('/admin/users', methods=['GET'])
@role_required(UserRole.ADMIN)
def admin_get_users():
    """Returns a list of all users for admin viewing."""
    users = User.query.all()
    return jsonify({"users": [user.to_dict() for user in users]}), 200

@app.route('/admin/user/<int:user_id>/toggle_role', methods=['POST'])
@role_required(UserRole.ADMIN)
def admin_toggle_user_role(user_id):
    """Allows admin to toggle user roles (e.g., promote to premium/admin)."""
    target_user = User.query.get(user_id)
    if not target_user:
        return jsonify({"success": False, "message": "User not found."}), 404
    
    if target_user.id == g.user.id and target_user.role == UserRole.ADMIN:
        logging.warning(f"Admin {g.user.username} tried to change their own admin role.")
        return jsonify({"success": False, "message": "You cannot change your own admin role."}), 403
    
    data = request.json
    new_role = data.get('role')

    if new_role not in [UserRole.GUEST, UserRole.STANDARD, UserRole.PREMIUM, UserRole.ADMIN]:
        return jsonify({"success": False, "message": "Invalid role specified."}), 400

    target_user.role = new_role
    db.session.commit()
    logging.info(f"Admin {g.user.username} changed role of user {target_user.username} to {new_role}.")
    return jsonify({"success": True, "message": f"User {target_user.username}'s role updated to {new_role}.", "user": target_user.to_dict()}), 200

@app.route('/admin/metrics', methods=['GET'])
@role_required(UserRole.ADMIN)
def admin_get_metrics():
    """Provides basic application metrics for admin."""
    total_users = User.query.count()
    total_conversations = Conversation.query.count()
    total_messages = Message.query.count()
    
    # Simulate some metrics for demo purposes
    ai_usage_24h = random.randint(100, 500)
    scrape_success_rate = round(random.uniform(80.0, 99.9), 2)

    metrics = {
        "total_users": total_users,
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "ai_calls_24h": ai_usage_24h,
        "web_scrape_success_rate": scrape_success_rate,
        "integrated_news_apis_count": 12, # Reflects the number of API fetchers in get_aggregated_news
        "current_guests_active": len(list(filter(lambda s: 'unauth_prompt_count' in session, list(app.session_cookie_store.keys())))) # Very rough estimate
    }
    logging.info(f"Admin {g.user.username} retrieved metrics.")
    return jsonify(metrics), 200

# New API endpoints for suggested prompts and about NexusAI (from previous turns)
@app.route('/api/suggested_prompts', methods=['GET'])
def get_suggested_prompts():
    """Returns a list of suggested prompts for users."""
    prompts = [
        "Summarize the latest news on renewable energy.",
        "Fact check the statement: 'The economic growth for Q1 was 5%'.",
        "Explain blockchain technology.",
        "What are the top headlines in sports today?",
        "Analyze the sentiment of the article about climate change.",
        "Compare recent reports on inflation from different sources.",
        "Give me updates on the latest space missions.",
        "What's new in AI ethics?",
        "Tell me about the recent political developments.",
        "Are there any breaking news about the global economy?"
    ]
    logging.info("Suggested prompts requested.")
    return jsonify({"suggestions": prompts}), 200 # Return as a dict with 'suggestions' key

@app.route('/api/about_nexusai', methods=['GET'])
def get_about_nexusai():
    """Returns information about the NexusAI application."""
    about_info = {
        "name": "NexusAI News Analyzer",
        "version": "1.2.0 (Expanded API Integration)",
        "description": "NexusAI is an AI-powered platform designed to provide comprehensive news analysis, fact-checking, and personalized insights. It helps users cut through the noise, verify information, and understand diverse perspectives.",
        "how_it_works": "NexusAI leverages advanced large language models (LLMs), a growing suite of integrated news APIs (GNews, NewsData.io, NYT, NewsAPI.org, MediaStack, The Guardian, Bing News, NewsCaster, SerpAPI for Google News, and conceptual integrations for premium services like AP and Financial Times), and intelligent web scraping to process vast amounts of information. When you ask a question, it identifies your intent, searches reliable sources (prioritizing your preferred ones), extracts key information, and synthesizes a concise, accurate response. It can also perform sentiment analysis, basic bias detection, and fact-checking based on available data.",
        "integrated_apis_overview": [
            {"name": "NewsAPI.org", "type": "Aggregator", "description": "Provides articles from thousands of news sources and blogs."},
            {"name": "MediaStack", "type": "Aggregator", "description": "Delivers real-time news data from global sources."},
            {"name": "GNews", "type": "Aggregator", "description": "Aggregates news from various publishers worldwide."},
            {"name": "NewsData.io", "type": "Aggregator", "description": "Offers news data from over 8000 news websites in multiple languages."},
            {"name": "NYT Article Search API", "type": "Publisher Specific", "description": "Accesses a vast archive of New York Times articles."},
            {"name": "The Guardian Open Platform API", "type": "Publisher Specific", "description": "Provides detailed content from The Guardian newspaper."},
            {"name": "Bing News API", "type": "Search Aggregator", "description": "Retrieves news articles indexed by Bing, covering a wide range of global sources."},
            {"name": "NewsCaster API", "type": "Aggregator", "description": "Offers real-time news data with advanced filtering capabilities."},
            {"name": "SerpAPI (Google News)", "type": "Search Integration", "description": "Extracts structured news results directly from Google News searches, providing broad coverage."},
            {"name": "Associated Press (AP) API", "type": "Wire Service (Conceptual)", "description": "A placeholder for integration with the global wire service API (typically commercial)."},
            {"name": "Financial Times (FT) API", "type": "Publisher Specific (Conceptual)", "description": "A placeholder for integration with the premium financial news API (typically commercial)."},
            {"name": "DuckDuckGo Search (Web Scrape)", "type": "General Web Search", "description": "Utilized for broader web searches and content scraping to find relevant articles beyond dedicated APIs."}
        ],
        "strengths": [
            "AI-powered news summarization and multi-document analysis.",
            "Enhanced fact-checking with confidence levels and evidence from multiple sources.",
            "Sentiment and basic bias analysis of articles and topics.",
            "Definition lookups for complex terms with sources.",
            "Ability to handle complex, multi-turn conversations with history awareness.",
            "Personalized news gathering based on user-preferred sources.",
            "Configurable analysis depth (brief, standard, detailed).",
            "Broadened news coverage through multiple API integrations."
        ],
        "common_mistakes": [
            "May occasionally provide information that is too general if the query is ambiguous or if sources are limited.",
            "Fact-checks are based on publicly available web data and integrated APIs; highly obscure or very recent, unconfirmed claims might be difficult to verify definitively.",
            "Sentiment and bias analysis are simplified; they may misinterpret sarcasm or subtle nuances, and do not represent a definitive journalistic assessment.",
            "Web scraping can be limited by robots.txt, paywalls, or website structures, affecting data access.",
            "Information is as current as available web data/APIs; real-time breaking news might have a slight delay.",
            "The model might sometimes 'hallucinate' or generate plausible but incorrect information, especially with limited context or complex reasoning tasks."
        ],
        "best_for": [
            "Quickly getting summaries of complex news stories and topics.",
            "Getting initial verification on claims and statements.",
            "Understanding new concepts or technical terms.",
            "Staying informed across various topics, prioritizing specific sources.",
            "Researchers needing initial overviews and source identification for further human investigation."
        ],
        "developers": "NexusAI Team",
        "contact_email": "support@nexusai.com"
    }
    logging.info("About NexusAI info requested.")
    return jsonify(about_info), 200


# NEW: User Preferences API Endpoints
@app.route('/api/user/preferences', methods=['GET'])
@role_required(UserRole.STANDARD)
def get_user_preferences():
    """Retrieves the current user's personalization preferences."""
    user = g.user
    if user:
        logging.info(f"User {user.username} requested preferences.")
        return jsonify({
            "preferred_topics": user.preferred_topics,
            "preferred_sources": user.preferred_sources,
            "analysis_depth": user.analysis_depth
        }), 200
    # The role_required decorator should handle this, but included for robustness
    return jsonify({"message": "User not found or not logged in."}), 404 

@app.route('/api/user/preferences', methods=['POST'])
@role_required(UserRole.STANDARD)
def update_user_preferences():
    """Updates the current user's personalization preferences."""
    user = g.user
    if not user: # Should be caught by role_required
        return jsonify({"message": "User not found or not logged in."}), 404 

    data = request.get_json()
    updated_fields = []

    if 'preferred_topics' in data and isinstance(data['preferred_topics'], list):
        user.preferred_topics = data['preferred_topics']
        updated_fields.append("preferred_topics")
    
    if 'preferred_sources' in data and isinstance(data['preferred_sources'], list):
        user.preferred_sources = data['preferred_sources']
        updated_fields.append("preferred_sources")
        if "millard ayo" in [s.lower() for s in data['preferred_sources']]:
            logging.info(f"User {user.username} added 'Millard Ayo' to preferred sources.")

    if 'analysis_depth' in data:
        # Validate analysis_depth input
        new_depth = data['analysis_depth']
        if new_depth in ['brief', 'standard', 'detailed']:
            user.analysis_depth = new_depth
            updated_fields.append("analysis_depth")
        else:
            return jsonify({"message": "Invalid value for analysis_depth. Must be 'brief', 'standard', or 'detailed'."}), 400
    
    if not updated_fields:
        return jsonify({"message": "No valid preference fields provided for update."}), 400

    try:
        db.session.commit()
        logging.info(f"User {user.username} updated preferences: {', '.join(updated_fields)}")
        return jsonify({"message": "Preferences updated successfully!", "user_preferences": user.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating user preferences for {user.username}: {e}", exc_info=True)
        return jsonify({"message": "An error occurred while updating preferences."}), 500


# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    logging.warning(f"404 Not Found: {request.path}")
    return jsonify({"error": "Resource not found.", "message": str(error)}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logging.warning(f"405 Method Not Allowed: {request.method} {request.path}")
    return jsonify({"error": "Method not allowed.", "message": str(error)}), 405

@app.errorhandler(429)
def ratelimit_handler(e):
    logging.warning(f"Rate limit exceeded for IP: {get_remote_address()} - {e.description}")
    return jsonify({"error": "Rate limit exceeded.", "message": f"Too many requests. Please try again in {e.retry_after} seconds."}), 429

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback() # Rollback any pending database transactions
    logging.exception("An internal server error occurred.")
    return jsonify({"error": "An unexpected server error occurred. Please try again later."}), 500

# --- WebSocket Event Handlers ---
# These handlers facilitate real-time communication between the client and server.
@socketio.on('connect')
def handle_connect():
    """Handles new Socket.IO client connections."""
    if g.user:
        # Join a room specific to the user's ID for personalized updates
        socketio.join(f'user_{g.user.id}') 
        logging.info(f"SocketIO client connected for user {g.user.username} (ID: {g.user.id}) in room 'user_{g.user.id}'. Sid: {request.sid}")
    else:
        # Guest users can be assigned a temporary room or just logged
        logging.info(f"SocketIO client connected (guest). Sid: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handles Socket.IO client disconnections."""
    if g.user:
        socketio.leave(f'user_{g.user.id}')
        logging.info(f"SocketIO client disconnected for user {g.user.username}. Sid: {request.sid}")
    else:
        logging.info(f"SocketIO client disconnected (guest). Sid: {request.sid}")

@socketio.on('user_typing')
def handle_user_typing(data):
    """Handles real-time typing indicators (conceptual)."""
    conversation_id = data.get('conversation_id')
    is_typing = data.get('is_typing')
    if conversation_id:
        logging.debug(f"User in conversation {conversation_id} is typing: {is_typing}")
        # In a multi-user chat, you would emit this to others in the conversation room:
        # emit('typing_status', {'user_id': g.user.id, 'is_typing': is_typing}, room=f'conversation_{conversation_id}', include_self=False)
    else:
        logging.debug(f"Guest user is typing: {is_typing}")

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        init_db() # Initialize the database tables and create default admin
        create_dummy_avatars() # Ensure dummy avatars exist for the frontend
    
    # Starting the background news monitor conceptually.
    # In a production environment, this would run as a separate process or service.
    # from threading import Thread
    # monitor_thread = Thread(target=start_background_news_monitor)
    # monitor_thread.daemon = True # Allow the main program to exit even if this thread is running
    # monitor_thread.start()
    # logging.info("Conceptual background news monitor thread started (if uncommented).")

    # Use socketio.run instead of app.run for Flask-SocketIO apps
    socketio.run(app, debug=app.config['DEBUG'], allow_unsafe_werkzeug=True, port=5000) # Specify port for clarity
