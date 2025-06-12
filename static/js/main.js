// static/js/main.js

// --- DOM Element References ---
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const sidebar = document.getElementById('sidebar');
const newChatButton = document.getElementById('new-chat-button');
const conversationList = document.getElementById('conversation-list');
const chatTitle = document.getElementById('chat-title');
const editChatTitleButton = document.getElementById('edit-chat-title-button'); // New element
const authSection = document.getElementById('auth-section');
const authModal = document.getElementById('auth-modal');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const loginSubmitBtn = document.getElementById('login-submit');
const registerSubmitBtn = document.getElementById('register-submit');
const showRegisterBtn = document.getElementById('show-register');
const showLoginBtn = document.getElementById('show-login');
const authModalCloseBtn = document.getElementById('auth-modal-close');
const modalTitle = document.getElementById('modal-title');
const modalMessageDiv = document.getElementById('modal-message');
const avatarSelectionDiv = document.getElementById('avatar-selection');
const menuIcon = document.getElementById('menu-icon');
const loadingIndicator = document.getElementById('loading-indicator');

// Custom Alert Modal elements
const customAlertModal = document.getElementById('custom-alert-modal');
const customAlertTitle = document.getElementById('custom-alert-title');
const customAlertMessage = document.getElementById('custom-alert-message');
const customAlertOkButton = document.getElementById('custom-alert-ok-button');
const customAlertCloseBtn = document.getElementById('custom-alert-close');


// Login/Register form specific inputs (for client-side validation)
const loginUsernameInput = document.getElementById('login-username');
const loginPasswordInput = document.getElementById('login-password');
const registerUsernameInput = document.getElementById('register-username');
const registerEmailInput = document.getElementById('register-email');
const registerPasswordInput = document.getElementById('register-password');

// Admin link (new)
const adminLink = document.getElementById('admin-link');


// --- Global Variables ---
let currentConversationId = null;
let selectedAvatarUrl = window.flaskData.avatarOptions && window.flaskData.avatarOptions.length > 0
    ? window.flaskData.avatarOptions[0]
    : '/static/avatars/default.png';
let currentUser = window.flaskData.currentUser;
let unauthPromptCount = 0;
let editingMessageElement = null; // Tracks which message is currently being edited
let socket; // Socket.IO instance

console.log('[main.js] Initial window.flaskData:', window.flaskData);
console.log('[main.js] Initial currentUser (from Flask):', currentUser);

// --- Markdown and Sanitization Setup ---
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false,
});

const renderer = {
    link(href, title, text) {
        const cleanHref = DOMPurify.sanitize(href);
        const cleanText = DOMPurify.sanitize(text);
        return `<a href="${cleanHref}" title="${title || ''}" target="_blank" rel="noopener noreferrer">${cleanText}</a>`;
    },
    image(href, title, text) {
        const cleanHref = DOMPurify.sanitize(href);
        const cleanText = DOMPurify.sanitize(text || '');
        return `<img src="${cleanHref}" alt="${cleanText}" title="${title || ''}" style="max-width:100%; height:auto; border-radius:8px;">`;
    }
};
marked.use({ renderer });

/**
 * Converts Markdown text to sanitized HTML.
 * @param {string} markdownText - The Markdown text to convert.
 * @returns {string} The sanitized HTML string.
 */
function convertAndSanitizeMarkdown(markdownText) {
    if (!markdownText) return '';
    const html = marked.parse(markdownText);
    return DOMPurify.sanitize(html, { 
        USE_PROFILES: { html: true },
        ADD_ATTR: ['target', 'rel'],
        FORBID_TAGS: ['script', 'style'],
        FORBID_ATTR: ['onerror', 'onload']
    }); 
}

// --- Message Display Functions ---

/**
 * Appends a new message bubble to the chat interface.
 * @param {string} sender - 'user' or 'bot'.
 * @param {string} text - The main content of the message (Markdown supported).
 * @param {number|null} messageId - Optional ID of the message for database tracking.
 * @param {string|null} details - Optional small text details to show under the main message.
 * @param {Array<Object>} sources - Optional array of source objects ({title, url, snippet}).
 * @param {string|null} sentiment - Optional sentiment of the message/response.
 */
function appendMessage(sender, text, messageId = null, details = null, sources = [], sentiment = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
    messageDiv.dataset.sender = sender;
    if (messageId) {
        messageDiv.dataset.messageId = messageId;
    }

    const contentP = document.createElement('p');
    contentP.innerHTML = convertAndSanitizeMarkdown(text);
    messageDiv.appendChild(contentP);

    if (details) {
        const detailsSmall = document.createElement('small');
        detailsSmall.innerHTML = convertAndSanitizeMarkdown(details);
        messageDiv.appendChild(detailsSmall);
    }
    
    if (sentiment && sender === 'bot') { // Display sentiment for bot messages
        const sentimentSpan = document.createElement('span');
        sentimentSpan.classList.add('message-sentiment');
        sentimentSpan.textContent = `Sentiment: ${sentiment}`;
        sentimentSpan.style.fontSize = '0.7em';
        sentimentSpan.style.color = '#888';
        sentimentSpan.style.display = 'block';
        sentimentSpan.style.marginTop = '5px';
        messageDiv.appendChild(sentimentSpan);
    }

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.classList.add('sources-list');
        sourcesDiv.innerHTML = '<strong>Relevant Sources:</strong><br>';
        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.classList.add('source-item');
            const sourceLink = document.createElement('a');
            sourceLink.href = DOMPurify.sanitize(source.url || '#');
            sourceLink.target = "_blank";
            sourceLink.rel = "noopener noreferrer";
            sourceLink.textContent = DOMPurify.sanitize(source.title || 'Untitled Source');
            sourceItem.appendChild(sourceLink);
            if (source.snippet) {
                const snippetSpan = document.createElement('span');
                snippetSpan.classList.add('snippet');
                snippetSpan.textContent = DOMPurify.sanitize(`"${source.snippet.substring(0, 150)}..."`);
                sourceItem.appendChild(snippetSpan);
            }
            sourcesDiv.appendChild(sourceItem);
        });
        messageDiv.appendChild(sourcesDiv);
    }

    const actionsDiv = document.createElement('div');
    actionsDiv.classList.add('message-actions');
    actionsDiv.setAttribute('aria-label', 'Message actions');

    const copyBtn = document.createElement('button');
    copyBtn.innerHTML = '<i class="far fa-copy"></i> Copy';
    copyBtn.setAttribute('aria-label', 'Copy message to clipboard');
    copyBtn.onclick = () => copyMessage(text);
    actionsDiv.appendChild(copyBtn);

    if (sender === 'user') {
        const editBtn = document.createElement('button');
        editBtn.innerHTML = '<i class="fas fa-edit"></i> Edit';
        editBtn.setAttribute('aria-label', 'Edit this message');
        editBtn.onclick = () => editMessage(messageDiv, text);
        actionsDiv.appendChild(editBtn);
    }
    
    const shareBtn = document.createElement('button');
    shareBtn.innerHTML = '<i class="fas fa-share-alt"></i> Share';
    shareBtn.setAttribute('aria-label', 'Share message content');
    shareBtn.onclick = () => shareMessage(text);
    actionsDiv.appendChild(shareBtn);

    messageDiv.appendChild(actionsDiv);
    chatMessages.appendChild(messageDiv);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Toggles the visibility of the loading indicator and enables/disables input elements.
 * @param {boolean} isLoading - True to show loading, false to hide.
 */
function toggleLoadingState(isLoading) {
    if (isLoading) {
        loadingIndicator.style.display = 'flex';
        chatMessages.scrollTop = chatMessages.scrollHeight;
        sendButton.disabled = true;
        userInput.disabled = true;
        userInput.classList.add('opacity-50', 'cursor-not-allowed');
        sendButton.classList.add('opacity-50', 'cursor-not-allowed');
    } else {
        loadingIndicator.style.display = 'none';
        sendButton.disabled = false;
        userInput.disabled = false;
        userInput.classList.remove('opacity-50', 'cursor-not-allowed');
        sendButton.classList.remove('opacity-50', 'cursor-not-allowed');
        userInput.focus();
    }
}

// --- Message Action Handlers ---

/**
 * Copies message text to the clipboard.
 * Uses fallback for older browsers or environments where navigator.clipboard might not work.
 * @param {string} text - The text to copy.
 */
function copyMessage(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
            .then(() => showCustomAlert('Success!', 'Message copied to clipboard!', 'success'))
            .catch(err => {
                console.error('Failed to copy using clipboard API: ', err);
                copyToClipboardFallback(text);
            });
    } else {
        copyToClipboardFallback(text);
    }
}

function copyToClipboardFallback(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        showCustomAlert('Success!', 'Message copied to clipboard!', 'success');
    } catch (err) {
        console.error('Fallback: Unable to copy to clipboard', err);
        showCustomAlert('Error', 'Failed to copy message. Please copy manually.', 'error');
    } finally {
        document.body.removeChild(textarea);
    }
}

/**
 * Puts the content of a message into the input for editing.
 * @param {HTMLElement} messageElement - The DOM element of the message.
 * @param {string} originalText - The original text of the message.
 */
function editMessage(messageElement, originalText) {
    if (editingMessageElement && editingMessageElement !== messageElement) {
        showCustomAlert("Warning", "Please finish editing the current message first.", "error");
        return;
    }
    if (messageElement.dataset.sender !== 'user') return;

    editingMessageElement = messageElement;
    userInput.value = originalText;
    userInput.focus();
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';

    sendButton.innerHTML = '<i class="fas fa-save"></i> Update';
    sendButton.onclick = () => updateEditedMessage(messageElement);
    showCustomAlert("Editing Mode", "You are now editing your previous message. Press 'Update' to send changes.", "info");
}

/**
 * Updates an edited message and resends it to the AI.
 * The old message is visually updated, and a new response is fetched.
 * @param {HTMLElement} messageElement - The DOM element of the edited message.
 */
async function updateEditedMessage(messageElement) {
    const newText = userInput.value.trim();
    if (newText === '') {
        showCustomAlert("Error", "Message cannot be empty. Please enter some text.", "error");
        return;
    }

    messageElement.querySelector('p').innerHTML = convertAndSanitizeMarkdown(newText);
    
    sendButton.innerHTML = '<i class="fas fa-paper-plane"></i> Send';
    sendButton.onclick = sendMessage;
    userInput.value = '';
    userInput.style.height = 'auto';
    editingMessageElement = null;

    toggleLoadingState(true);
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: newText, 
                conversation_id: currentConversationId 
            })
        });
        const data = await response.json();
        toggleLoadingState(false);

        if (response.ok) {
            appendMessage('bot', data.response, null, data.details, data.sources, data.sentiment);
            if (data.new_conversation_title && data.conversation_id && !currentConversationId) {
                currentConversationId = data.conversation_id;
                addConversationToList(data.conversation_id, data.new_conversation_title);
                const newConvItem = conversationList.querySelector(`.conversation-item[data-conversation-id="${data.conversation_id}"]`);
                if (newConvItem) newConvItem.classList.add('active');
                chatTitle.textContent = data.new_conversation_title;
                if (currentUser) { // Only show edit button if logged in
                    editChatTitleButton.style.display = 'inline-block';
                }
            } else if (data.conversation_id === currentConversationId) {
                fetchConversations();
            }
        } else {
            const errorMessage = data.message || data.error || 'Failed to update message.';
            showCustomAlert('Error', errorMessage, 'error');
            appendMessage('bot', `Error: ${errorMessage}`, null, 'small');
            if (data.prompt_login_required) {
                showLoginRegisterModal(data.message, "login");
                if (messageElement && messageElement.dataset.sender === 'user') {
                    messageElement.remove();
                }
            }
        }
    } catch (error) {
        console.error('Error sending edited message:', error);
        toggleLoadingState(false);
        showCustomAlert('Network Error', 'An error occurred while re-processing your edited message. Please check your connection.', 'error');
        appendMessage('bot', `An unexpected error occurred: ${error.message}`, null, 'small');
    }
}

/**
 * Shares message content using the Web Share API if available, otherwise falls back to copy.
 * @param {string} text - The text to share.
 */
function shareMessage(text) {
    if (navigator.share) {
        navigator.share({
            title: 'NexusAI Chat',
            text: text,
        }).then(() => {
            showCustomAlert('Success!', 'Message shared successfully!', 'success');
        }).catch((error) => {
            console.error('Error sharing:', error);
            showCustomAlert('Error', 'Failed to share message. You can copy it instead.', 'error');
            copyMessage(text);
        });
    } else {
        showCustomAlert('Information', 'Web Share API is not supported in this browser. The message has been copied to your clipboard instead.', 'info');
        copyMessage(text);
    }
}

// --- Custom Alert Modal Functions ---
/**
 * Shows a custom alert modal.
 * @param {string} title - The title of the alert.
 * @param {string} message - The message content.
 * @param {string} type - 'success', 'error', or 'info'.
 */
function showCustomAlert(title, message, type = 'info') {
    customAlertTitle.textContent = title;
    customAlertMessage.textContent = message;
    customAlertMessage.className = 'modal-message'; // Reset classes
    customAlertMessage.classList.add(`message-${type}`);
    customAlertModal.style.display = 'flex'; // Show the modal
    customAlertOkButton.focus(); // Focus OK button for accessibility
    customAlertModal.setAttribute('aria-hidden', 'false');
    customAlertModal.setAttribute('aria-labelledby', 'custom-alert-title');
}

/**
 * Hides the custom alert modal.
 */
function hideCustomAlert() {
    customAlertModal.style.display = 'none';
    customAlertModal.setAttribute('aria-hidden', 'true');
    userInput.focus(); // Return focus to chat input
}

// --- Authentication & UI Rendering ---

/**
 * Fetches current user information from the backend and updates the UI.
 * Determines if user is logged in or a guest and renders appropriate sidebar section.
 */
async function fetchUserInfo() {
    console.log('[main.js] fetchUserInfo: Attempting to fetch user info...');
    toggleLoadingState(true);
    try {
        const response = await fetch('/user_info');
        const data = await response.json();
        console.log('[main.js] fetchUserInfo: Received user info data:', data);

        currentUser = data.user;
        unauthPromptCount = data.unauth_prompt_count;

        if (!data.user) {
            currentUser = null;
            currentConversationId = null;
            console.log('[main.js] fetchUserInfo: Backend reported not logged in. currentUser set to null.');
        } else {
            currentConversationId = currentUser.current_conversation_id;
        }

        renderAuthSection(!!currentUser, currentUser);
        if (currentUser) {
            console.log('[main.js] fetchUserInfo: User is logged in, fetching conversations.');
            fetchConversations();
            editChatTitleButton.style.display = 'inline-block'; // Show edit button for logged in users
            // Show admin link if user is admin
            if (currentUser.role === 'admin') {
                adminLink.style.display = 'flex';
            } else {
                adminLink.style.display = 'none';
            }
        } else {
            console.log('[main.js] fetchUserInfo: User is guest, not fetching conversations.');
            conversationList.innerHTML = '';
            chatTitle.textContent = "Welcome to NexusAI!";
            editChatTitleButton.style.display = 'none'; // Hide edit button for guests
            adminLink.style.display = 'none'; // Hide admin link for guests
        }
    } catch (error) {
        console.error('[main.js] fetchUserInfo: Error fetching user info:', error);
        renderAuthSection(false);
        showCustomAlert('Error', 'Failed to load user information. Please refresh the page.', 'error');
    } finally {
        toggleLoadingState(false);
    }
}

/**
 * Renders the authentication section in the sidebar based on login status.
 * @param {boolean} isLoggedIn - True if a user is logged in, false otherwise.
 * @param {Object|null} user - The user object if logged in.
 */
function renderAuthSection(isLoggedIn, user = null) {
    console.log(`[main.js] renderAuthSection: Rendering. Is logged in: ${isLoggedIn}, User:`, user);
    authSection.innerHTML = '';
    if (isLoggedIn && user) {
        authSection.innerHTML = `
            <div class="user-info-sidebar" role="region" aria-label="User information and logout">
                <img src="${DOMPurify.sanitize(user.avatar_url)}" alt="${DOMPurify.sanitize(user.username)} Avatar">
                <span class="username">${DOMPurify.sanitize(user.username)}</span>
                <button class="logout-btn" id="logout-button" aria-label="Log out"><i class="fas fa-sign-out-alt"></i> Logout</button>
            </div>
        `;
        document.getElementById('logout-button').addEventListener('click', logout);
        document.getElementById('settings-link').style.display = 'flex';
    } else {
        authSection.innerHTML = `
            <div class="guest-mode-info" role="region" aria-label="Guest mode information">
                You are chatting as a guest. <br>Prompts left: <span class="guest-count">${5 - unauthPromptCount}</span>
                <button id="show-login-signup-btn" aria-label="Login or sign up">Login / Sign Up</button>
            </div>
        `;
        document.getElementById('show-login-signup-btn').addEventListener('click', () => showLoginRegisterModal("","login"));
        document.getElementById('settings-link').style.display = 'none';
    }
}

/**
 * Displays a message in the authentication modal.
 * @param {string} message - The message text.
 * @param {string} type - 'success', 'error', or 'info'.
 */
function showAuthModalMessage(message, type = 'error') {
    modalMessageDiv.textContent = message;
    modalMessageDiv.className = 'modal-message';
    modalMessageDiv.classList.add(`message-${type}`);
    modalMessageDiv.style.display = 'block';
    modalMessageDiv.setAttribute('role', type === 'error' ? 'alert' : 'status');
}

/**
 * Hides the authentication modal message.
 */
function hideAuthModalMessage() {
    modalMessageDiv.style.display = 'none';
    modalMessageDiv.removeAttribute('role');
}

/**
 * Shows the login/register modal.
 * @param {string} message - An initial message to display (e.g., login required).
 * @param {string} mode - 'login' or 'register'.
 */
function showLoginRegisterModal(message = "", mode = "login") {
    authModal.style.display = 'flex';
    hideAuthModalMessage();
    if (message) showAuthModalMessage(message, 'error');

    loginForm.setAttribute('aria-hidden', mode !== 'login');
    registerForm.setAttribute('aria-hidden', mode !== 'register');

    if (mode === "login") {
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
        modalTitle.textContent = 'Login';
        loginUsernameInput.focus();
    } else {
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
        modalTitle.textContent = 'Register';
        registerUsernameInput.focus();
    }
    populateAvatarOptions();
}

/**
 * Hides the login/register modal and clears input fields.
 */
function hideLoginRegisterModal() {
    authModal.style.display = 'none';
    hideAuthModalMessage();
    loginUsernameInput.value = '';
    loginPasswordInput.value = '';
    registerUsernameInput.value = '';
    registerEmailInput.value = '';
    registerPasswordInput.value = '';
    userInput.focus();
}

/**
 * Handles user login. Includes client-side validation.
 */
async function login() {
    const username = loginUsernameInput.value.trim();
    const password = loginPasswordInput.value.trim();

    if (!username || !password) {
        showAuthModalMessage('Please enter both username and password.', 'error');
        return;
    }

    toggleLoadingState(true);
    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        if (response.ok) {
            hideLoginRegisterModal();
            currentUser = data.user;
            console.log('[main.js] Login successful. New currentUser:', currentUser);
            await fetchUserInfo();
            newChat();
            showCustomAlert('Login Successful!', data.message, 'success');
        } else {
            showAuthModalMessage(data.message || 'Login failed. Please try again.', 'error');
        }
    } catch (error) {
        console.error('[main.js] Login error:', error);
        showAuthModalMessage('An error occurred during login. Please check your network connection.', 'error');
    } finally {
        toggleLoadingState(false);
    }
}

/**
 * Handles user registration. Includes client-side validation.
 */
async function register() {
    const username = registerUsernameInput.value.trim();
    const email = registerEmailInput.value.trim();
    const password = registerPasswordInput.value.trim();

    if (!username || !email || !password || !selectedAvatarUrl) {
        showAuthModalMessage("All fields are required and an avatar must be selected.", "error");
        return;
    }
    if (!/\S+@\S+\.\S+/.test(email)) {
        showAuthModalMessage('Please enter a valid email address.', 'error');
        return;
    }
    if (password.length < 6) {
        showAuthModalMessage('Password must be at least 6 characters long.', 'error');
        return;
    }

    toggleLoadingState(true);
    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password, avatar_url: selectedAvatarUrl })
        });
        const data = await response.json();
        if (response.ok) {
            hideLoginRegisterModal();
            currentUser = data.user;
            console.log('[main.js] Registration successful. New currentUser:', currentUser);
            await fetchUserInfo();
            newChat();
            showCustomAlert('Registration Successful!', data.message, 'success');
        } else {
            showAuthModalMessage(data.message || 'Registration failed. Please try again.', 'error');
        }
    } catch (error) {
        console.error('[main.js] Register error:', error);
        showAuthModalMessage('An error occurred during registration. Please check your network connection.', 'error');
    } finally {
        toggleLoadingState(false);
    }
}

/**
 * Handles user logout.
 */
async function logout() {
    toggleLoadingState(true);
    try {
        const response = await fetch('/logout', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            currentUser = null;
            currentConversationId = null;
            console.log('[main.js] Logout successful. currentUser is now null.');
            chatMessages.innerHTML = `
                <div class="message bot-message">
                    <p>Hello! I'm your AI News Analyst and Fact-Checker. What would you like to know or verify today?</p>
                    <small>Examples: "Analyze recent news about AI ethics.", "Fact-check: Is the earth flat?", "Summarize the latest on climate policy."</small>
                </div>
            `;
            chatTitle.textContent = "Welcome to NexusAI!";
            await fetchUserInfo(); // Use await to ensure UI updates before next step
            showCustomAlert('Logged Out', data.message, 'success');
        } else {
            showCustomAlert('Logout Failed', data.message || 'Logout failed. Please try again.', 'error');
        }
    } catch (error) {
        console.error('[main.js] Logout error:', error);
        showCustomAlert('Network Error', 'An error occurred during logout. Please check your network connection.', 'error');
    } finally {
        toggleLoadingState(false);
    }
}

/**
 * Populates the avatar selection options in the register modal.
 * Also sets up click handlers and "selected" class.
 */
function populateAvatarOptions() {
    avatarSelectionDiv.innerHTML = '';
    const avatarOptions = window.flaskData.avatarOptions;
    avatarOptions.forEach(url => {
        const img = document.createElement('img');
        img.src = DOMPurify.sanitize(url);
        img.alt = `Avatar ${url.split('/').pop().split('.')[0]}`;
        img.classList.add('avatar-option');
        img.setAttribute('role', 'radio');
        img.setAttribute('tabindex', '0');
        img.setAttribute('aria-checked', 'false');

        if (url === selectedAvatarUrl) {
            img.classList.add('selected');
            img.setAttribute('aria-checked', 'true');
        }

        img.onclick = () => {
            document.querySelectorAll('.avatar-option').forEach(i => {
                i.classList.remove('selected');
                i.setAttribute('aria-checked', 'false');
            });
            img.classList.add('selected');
            img.setAttribute('aria-checked', 'true');
            selectedAvatarUrl = url;
        };
        img.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                img.click();
            }
        });
        avatarSelectionDiv.appendChild(img);
    });
}

// --- Conversation Management ---

/**
 * Fetches and displays the list of user conversations in the sidebar.
 * Automatically loads the most recent conversation if none is active on initial fetch.
 */
async function fetchConversations() {
    console.log('[main.js] fetchConversations: Attempting to fetch conversations...');
    if (!currentUser) {
        console.log('[main.js] fetchConversations: currentUser is null, clearing conversation list.');
        conversationList.innerHTML = '';
        return;
    }
    toggleLoadingState(true);
    try {
        const response = await fetch('/conversations');
        if (!response.ok) {
            if (response.status === 401 || response.status === 403) {
                // User is logged in but might be denied by role_required or session issue
                console.error('[main.js] Unauthorized to fetch conversations. Forcing logout and re-render.');
                // Simulate logout and re-render as guest to recover
                currentUser = null;
                currentConversationId = null;
                renderAuthSection(false);
                conversationList.innerHTML = '';
                showCustomAlert('Authentication Error', 'Your session might have expired. Please log in again.', 'error');
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('[main.js] fetchConversations: Received conversations data:', data);

        conversationList.innerHTML = '';
        if (data.conversations && data.conversations.length > 0) {
            data.conversations.forEach(conv => {
                const listItem = document.createElement('li');
                listItem.appendChild(createConversationItem(conv.id, conv.title));
                conversationList.appendChild(listItem);
            });

            const activeConvExists = conversationList.querySelector(`.conversation-item[data-conversation-id="${currentConversationId}"]`);
            if (!currentConversationId || !activeConvExists) {
                console.log('[main.js] fetchConversations: No active conversation or active not found. Loading most recent.');
                loadConversation(data.conversations[0].id);
            } else {
                const activeConv = conversationList.querySelector(`[data-conversation-id="${currentConversationId}"]`);
                if (activeConv) {
                    activeConv.classList.add('active');
                    chatTitle.textContent = activeConv.querySelector('.title').textContent;
                    editChatTitleButton.style.display = 'inline-block'; // Show edit button when conversation is active
                    console.log(`[main.js] fetchConversations: Highlighted active conversation ${currentConversationId}.`);
                }
            }
        } else {
            console.log('[main.js] fetchConversations: No conversations found for user.');
            chatTitle.textContent = "Welcome to NexusAI!";
            chatMessages.innerHTML = `
                <div class="message bot-message">
                    <p>Hello! I'm your AI News Analyst and Fact-Checker. What would you like to know or verify today?</p>
                    <small>Examples: "Analyze recent news about AI ethics.", "Fact-check: Is the earth flat?", "Summarize the latest on climate policy."</small>
                </div>
            `;
            currentConversationId = null;
            editChatTitleButton.style.display = 'none'; // Hide edit button if no conversations
        }
    } catch (error) {
        console.error('[main.js] fetchConversations: Error fetching conversations:', error);
        showCustomAlert('Error', 'Failed to load your conversations. Please try again.', 'error');
    } finally {
        toggleLoadingState(false);
    }
}

/**
 * Creates a single conversation item DOM element for the sidebar list.
 * @param {number} id - The conversation ID.
 * @param {string} title - The conversation title.
 * @returns {HTMLElement} The created conversation item div.
 */
function createConversationItem(id, title) {
    const convItem = document.createElement('div');
    convItem.classList.add('conversation-item');
    convItem.dataset.conversationId = id;
    convItem.setAttribute('role', 'button');
    convItem.setAttribute('tabindex', '0');

    convItem.innerHTML = `
        <span class="title">${DOMPurify.sanitize(title)}</span>
        <button class="delete-btn" title="Delete conversation" aria-label="Delete conversation ${DOMPurify.sanitize(title)}">
            <i class="fas fa-trash-alt"></i>
        </button>
    `;
    convItem.addEventListener('click', (event) => {
        if (!event.target.closest('.delete-btn')) {
            loadConversation(id);
        }
    });
    convItem.querySelector('.delete-btn').addEventListener('click', (event) => {
        event.stopPropagation();
        deleteConversation(id, convItem);
    });
    convItem.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            convItem.click();
        }
    });
    return convItem;
}

/**
 * Adds a new conversation to the sidebar list (used when a new chat is created).
 * @param {number} id - The conversation ID.
 * @param {string} title - The conversation title.
 */
function addConversationToList(id, title) {
    const listItem = document.createElement('li');
    listItem.appendChild(createConversationItem(id, title));
    conversationList.prepend(listItem);
    console.log(`[main.js] addConversationToList: Added conversation ID ${id}: "${title}"`);
}

/**
 * Loads a specific conversation's messages into the chat area.
 * Handles UI updates like active class and chat title.
 * @param {number} convId - The ID of the conversation to load.
 */
async function loadConversation(convId) {
    console.log(`[main.js] loadConversation: Attempting to load conversation ID ${convId}. Current ID: ${currentConversationId}`);
    if (currentConversationId === convId) {
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('active');
        }
        return;
    }

    document.querySelectorAll('.conversation-item').forEach(btn => btn.classList.remove('active'));

    currentConversationId = convId;
    chatMessages.innerHTML = '';
    chatTitle.textContent = "Loading Chat...";
    toggleLoadingState(true);

    const activeConvElement = conversationList.querySelector(`[data-conversation-id="${currentConversationId}"]`);
    if (activeConvElement) {
        activeConvElement.classList.add('active');
    }

    try {
        const response = await fetch(`/conversations/${convId}/messages`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}. Failed to load conversation messages.`);
        }
        const data = await response.json();
        
        chatTitle.textContent = DOMPurify.sanitize(data.title || "Loaded Chat");
        if (currentUser) { // Only show edit button if logged in
            editChatTitleButton.style.display = 'inline-block';
        }
        if (data.messages) {
            console.log(`[main.js] loadConversation: Received ${data.messages.length} messages for conversation ${convId}.`);
            data.messages.forEach(msg => {
                appendMessage(msg.sender, msg.content, msg.id, msg.details, msg.sources, msg.sentiment);
            });
        }
    } catch (error) {
        console.error('[main.js] Error loading messages:', error);
        showCustomAlert('Error', 'Error loading this conversation. It might have been deleted or there was a network issue.', 'error');
        appendMessage('bot', 'Error loading this conversation. It might have been deleted or there was a network issue.', null, 'small');
        chatTitle.textContent = "Error Loading Chat";
        editChatTitleButton.style.display = 'none'; // Hide edit button on error
    } finally {
        toggleLoadingState(false);
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('active');
        }
    }
}

/**
 * Deletes a conversation from the database and updates the UI.
 * @param {number} convId - The ID of the conversation to delete.
 * @param {HTMLElement} convItemElement - The DOM element of the conversation item to remove.
 */
async function deleteConversation(convId, convItemElement) {
    // Custom confirmation logic
    const confirmed = await new Promise(resolve => {
        showCustomAlert('Confirm Delete', 'Are you sure you want to delete this conversation? This cannot be undone.', 'error');
        customAlertOkButton.onclick = () => { hideCustomAlert(); resolve(true); };
        customAlertCloseBtn.onclick = () => { hideCustomAlert(); resolve(false); };
    });

    if (!confirmed) {
        return;
    }
    
    console.log(`[main.js] deleteConversation: Attempting to delete conversation ID ${convId}.`);
    toggleLoadingState(true);

    try {
        const response = await fetch(`/conversations/${convId}/delete`, {
            method: 'DELETE'
        });
        const data = await response.json();
        if (response.ok) {
            showCustomAlert('Success!', data.message, 'success');
            if (currentConversationId === convId) {
                console.log('[main.js] deleteConversation: Deleted active conversation, starting new chat.');
                newChat();
            }
            if (convItemElement && convItemElement.parentNode) {
                convItemElement.parentNode.remove();
            }
            fetchConversations();
        } else {
            showCustomAlert('Error', data.message || 'Failed to delete conversation.', 'error');
        }
    } catch (error) {
        console.error('[main.js] deleteConversation: Error deleting conversation:', error);
        showCustomAlert('Network Error', 'An error occurred while deleting the conversation. Please check your network.', 'error');
    } finally {
        toggleLoadingState(false);
    }
}

/**
 * Allows user to edit the title of the current conversation.
 */
async function editConversationTitle() {
    if (!currentConversationId) {
        showCustomAlert('Info', 'No conversation selected to edit.', 'info');
        return;
    }

    const currentTitle = chatTitle.textContent.trim();
    const newTitle = prompt('Enter new conversation title:', currentTitle); // Using native prompt for simplicity here

    if (newTitle === null || newTitle.trim() === '' || newTitle.trim() === currentTitle) {
        showCustomAlert('Info', 'Title not changed.', 'info');
        return;
    }

    toggleLoadingState(true);
    try {
        const response = await fetch(`/conversations/${currentConversationId}/edit_title`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle.trim() })
        });
        const data = await response.json();
        if (response.ok) {
            chatTitle.textContent = DOMPurify.sanitize(data.new_title);
            // Update the title in the sidebar list without re-fetching all conversations
            const convItemElement = conversationList.querySelector(`.conversation-item[data-conversation-id="${currentConversationId}"] .title`);
            if (convItemElement) {
                convItemElement.textContent = DOMPurify.sanitize(data.new_title);
            }
            showCustomAlert('Success!', data.message, 'success');
        } else {
            showCustomAlert('Error', data.message || 'Failed to update title.', 'error');
        }
    } catch (error) {
        console.error('Error editing conversation title:', error);
        showCustomAlert('Network Error', 'An error occurred while updating the title. Please check your connection.', 'error');
    } finally {
        toggleLoadingState(false);
    }
}


/**
 * Starts a new blank chat session.
 * Resets chat area and ensures no conversation is active.
 */
function newChat() {
    console.log('[main.js] newChat: Starting a new chat session.');
    currentConversationId = null;
    chatMessages.innerHTML = `
        <div class="message bot-message">
            <p>Hello! I'm your AI News Analyst and Fact-Checker. What would you like to know or verify today?</p>
            <small>Examples: "Analyze recent news about AI ethics.", "Fact-check: Is the earth flat?", "Summarize the latest on climate policy."</small>
        </div>
    `;
    chatTitle.textContent = "Welcome to NexusAI!";
    editChatTitleButton.style.display = 'none'; // Hide edit button for new chat
    userInput.value = '';
    userInput.style.height = 'auto';
    userInput.focus();
    document.querySelectorAll('.conversation-item.active').forEach(el => el.classList.remove('active'));
    toggleLoadingState(false);
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('active');
    }
}

// --- Main Send Message Function ---

/**
 * Handles sending a user message to the backend API.
 * Manages loading states, displays messages, and updates conversation list.
 */
async function sendMessage() {
    const prompt = userInput.value.trim();
    if (prompt === '') return;

    appendMessage('user', prompt);
    userInput.value = '';
    userInput.style.height = 'auto';

    toggleLoadingState(true);

    try {
        console.log(`[main.js] sendMessage: Sending prompt for conversation ID: ${currentConversationId}.`);
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: prompt, 
                conversation_id: currentConversationId 
            })
        });

        const data = await response.json();
        toggleLoadingState(false);

        if (response.status === 403 && data.prompt_login_required) {
            showLoginRegisterModal(data.message, "login");
            unauthPromptCount = data.unauth_prompt_count;
            renderAuthSection(false);
            if (chatMessages.lastChild && chatMessages.lastChild.dataset.sender === 'user') {
                chatMessages.lastChild.remove();
            }
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return;
        } else if (!response.ok) {
            const errorMessage = data.message || data.error || `Server error: ${response.status}`;
            console.error('[main.js] sendMessage: Server error:', errorMessage);
            showCustomAlert('Error', errorMessage, 'error');
            appendMessage('bot', `Error: ${errorMessage}. Please try again.`, null, 'small');
            return;
        }

        appendMessage('bot', data.response, null, data.details, data.sources, data.sentiment);
        
        if (data.new_conversation_title && data.conversation_id) {
            currentConversationId = data.conversation_id;
            console.log(`[main.js] sendMessage: New conversation created. ID: ${data.conversation_id}, Title: "${data.new_conversation_title}"`);
            addConversationToList(data.conversation_id, data.new_conversation_title);
            const newConvItem = conversationList.querySelector(`[data-conversation-id="${data.conversation_id}"]`);
            if (newConvItem) {
                newConvItem.classList.add('active');
                chatTitle.textContent = data.new_conversation_title;
                if (currentUser) { // Only show edit button if logged in
                    editChatTitleButton.style.display = 'inline-block';
                }
            }
        } else if (data.conversation_id === currentConversationId && currentUser) {
            console.log(`[main.js] sendMessage: Existing conversation updated. Re-fetching conversations list.`);
            fetchConversations();
        }
    } catch (error) {
        console.error('[main.js] sendMessage: Network error:', error);
        toggleLoadingState(false);
        showCustomAlert('Network Error', 'A network error occurred. Please check your internet connection.', 'error');
        appendMessage('bot', `An unexpected network error occurred: ${error.message}. Please try again.`, null, 'small');
    }
}

// --- Socket.IO Initialization and Event Listeners ---
function initializeSocketIO() {
    socket = io(); // Connect to the Socket.IO server
    console.log('[main.js] Attempting to connect to Socket.IO...');

    socket.on('connect', () => {
        console.log('Socket.IO connected!');
    });

    socket.on('disconnect', () => {
        console.log('Socket.IO disconnected.');
    });

    socket.on('receive_message', (data) => {
        console.log('Received real-time message:', data);
        // Only append if it's for the currently active conversation
        if (currentConversationId === data.conversation_id) {
            appendMessage(data.message.sender, data.message.content, data.message.id, data.message.details, data.message.sources, data.message.sentiment);
        }
        // Invalidate conversations cache to ensure sidebar updates
        fetchConversations(); // Re-fetch to update last_updated order
    });

    socket.on('error', (error) => {
        console.error('Socket.IO error:', error);
        showCustomAlert('WebSocket Error', 'A real-time connection error occurred. Functionality might be limited.', 'error');
    });
}


// --- Event Listeners ---

// Send button click
sendButton.addEventListener('click', sendMessage);

// User input keypress (Enter to send, Shift+Enter for new line)
userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { 
        e.preventDefault();
        sendMessage();
    }
});

// New chat button click
newChatButton.addEventListener('click', newChat);

// Mobile menu icon click to toggle sidebar
menuIcon.addEventListener('click', () => {
    sidebar.classList.toggle('active');
});

// Click outside sidebar on mobile to close it
document.addEventListener('click', (event) => {
    if (window.innerWidth <= 768 && sidebar.classList.contains('active')) {
        if (!sidebar.contains(event.target) && !menuIcon.contains(event.target)) {
            sidebar.classList.remove('active');
        }
    }
});

// Dynamic textarea height adjustment with debouncing
let resizeTimeout;
userInput.addEventListener('input', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    }, 50);
});

// Modal button listeners
showRegisterBtn.addEventListener('click', () => showLoginRegisterModal("", "register"));
showLoginBtn.addEventListener('click', () => showLoginRegisterModal("", "login"));
loginSubmitBtn.addEventListener('click', login);
registerSubmitBtn.addEventListener('click', register);
authModalCloseBtn.addEventListener('click', hideLoginRegisterModal);

// Custom Alert Modal listeners
customAlertOkButton.addEventListener('click', hideCustomAlert);
customAlertCloseBtn.addEventListener('click', hideCustomAlert);

// Trap focus inside modal (basic example, more complex for full accessibility)
authModal.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        hideLoginRegisterModal();
    }
});
customAlertModal.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        hideCustomAlert();
    }
});

// Edit chat title button click
editChatTitleButton.addEventListener('click', editConversationTitle);


// --- Initial Page Load Setup ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[main.js] DOMContentLoaded: Initializing app...');
    // Initial fetch of user info to determine login state and render UI accordingly
    await fetchUserInfo();

    // Show auth modal if Flask passed a flag and user is NOT logged in
    if (!currentUser && window.flaskData.showAuthModal) {
        showLoginRegisterModal("Please login or sign up to begin chatting!", "login");
    }
    populateAvatarOptions();
    initializeSocketIO(); // Initialize Socket.IO connection
});
