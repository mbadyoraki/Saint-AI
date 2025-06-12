# models.py
from datetime import datetime
from database import db # Assuming 'db' is initialized in database.py
from werkzeug.security import generate_password_hash, check_password_hash
import json # Import json for handling preferences

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    avatar_url = db.Column(db.String(255), default='/static/avatars/default.png')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # New fields for personalization preferences
    _preferred_topics = db.Column(db.Text, default='[]') # Stores JSON string of a list of topics
    _preferred_sources = db.Column(db.Text, default='[]') # Stores JSON string of a list of sources
    analysis_depth = db.Column(db.String(50), default='standard') # e.g., 'brief', 'standard', 'detailed'

    # Relationship to Conversations: a user can have many conversations
    conversations = db.relationship('Conversation', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        """Hashes the password and sets it."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks if the provided password matches the hashed password."""
        return check_password_hash(self.password_hash, password)

    # Properties for preferred_topics to handle JSON serialization/deserialization
    @property
    def preferred_topics(self):
        if self._preferred_topics:
            return json.loads(self._preferred_topics)
        return []

    @preferred_topics.setter
    def preferred_topics(self, value):
        self._preferred_topics = json.dumps(value)

    # Properties for preferred_sources to handle JSON serialization/deserialization
    @property
    def preferred_sources(self):
        if self._preferred_sources:
            return json.loads(self._preferred_sources)
        return []

    @preferred_sources.setter
    def preferred_sources(self, value):
        self._preferred_sources = json.dumps(value)

    def to_dict(self):
        """Converts user object to a dictionary for JSON serialization."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at.isoformat(),
            'preferred_topics': self.preferred_topics, # Include new fields
            'preferred_sources': self.preferred_sources, # Include new fields
            'analysis_depth': self.analysis_depth # Include new field
        }

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    title = db.Column(db.String(255), nullable=False) # Short, descriptive title for the conversation thread
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) # For sorting by most recent activity

    # Relationship to Messages: a conversation can have many messages
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade="all, delete-orphan", order_by="Message.timestamp")

    def to_dict(self):
        """Converts conversation object to a dictionary for JSON serialization."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id', ondelete='CASCADE'), nullable=False)
    sender = db.Column(db.String(10), nullable=False) # 'user' or 'ai'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text, nullable=True)

    def to_dict(self):
        """Converts message object to a dictionary for JSON serialization."""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'sender': self.sender,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }