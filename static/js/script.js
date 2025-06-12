// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const authButton = document.getElementById('auth-button');
    const logoutButton = document.getElementById('logout-button');
    const newChatButton = document.getElementById('new-chat-button');
    const conversationList = document.getElementById('conversation-list');
    const userAvatar = document.getElementById('user-avatar');
    const usernameDisplay = document.getElementById('username-display');
    const currentChatTitle = document.getElementById('current-chat-title');
    const toggleSidebarButton = document.getElementById('toggle-sidebar-button');
    const sidebar = document.getElementById('sidebar');

    let currentConversationId = null; // Stores the ID of the active conversation
    let currentUnauthPromptCount = 0; // Tracks prompts for unauthenticated users

    // --- UI Utility Functions ---

    function displayMessage(sender, content, type = 'text', details = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        // Convert newlines in content to <br> for HTML rendering, but keep original for pre if needed
        let htmlContent = content.replace(/\n\n/g, '<p>').replace(/\n/g, '<br>');

        // Check if content might be code-like
        if (content.includes('```')) {
            // Simple markdown for code blocks
            htmlContent = content.replace(/```(.*?)```/gs, (match, code) => {
                return `<pre>${code.trim()}</pre>`;
            });
        }
        
        messageDiv.innerHTML = htmlContent;

        if (details) {
            const detailsElement = document.createElement('small');
            detailsElement.innerHTML = details.replace(/\n/g, '<br>');
            messageDiv.appendChild(detailsElement);
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
    }

    function showLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('loading-indicator');
        loadingDiv.textContent = "AI is thinking...";
        loadingDiv.id = 'loading-indicator';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideLoadingIndicator() {
        const loadingDiv = document.getElementById('loading-indicator');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    function clearChat() {
        chatMessages.innerHTML = `
            <div class="message bot-message initial-message">
                Hello! I'm your AI News Analyst and Fact-Checker. What would you like to know or verify today?
                <br><small>Examples: "Analyze recent news about AI ethics.", "Fact-check: Is the earth flat?", "List 5 ways to improve focus."</small>
            </div>
        `;
    }

    function toggleAuthButtons(isLoggedIn) {
        if (isLoggedIn) {
            authButton.style.display = 'none';
            logoutButton.style.display = 'block';
            newChatButton.style.display = 'block'; // Only logged in users can create new persistent chats
            usernameDisplay.textContent = localStorage.getItem('username') || 'User';
            userAvatar.src = localStorage.getItem('avatar_url') || '/static/avatars/default.png';
        } else {
            authButton.style.display = 'block';
            logoutButton.style.display = 'none';
            newChatButton.style.display = 'none';
            usernameDisplay.textContent = 'Guest';
            userAvatar.src = '/static/avatars/default.png';
        }
    }

    function updateUserInfo(userData) {
        if (userData && userData.is_logged_in) {
            localStorage.setItem('user_id', userData.user.id);
            localStorage.setItem('username', userData.user.username);
            localStorage.setItem('avatar_url', userData.user.avatar_url);
            toggleAuthButtons(true);
        } else {
            localStorage.removeItem('user_id');
            localStorage.removeItem('username');
            localStorage.removeItem('avatar_url');
            toggleAuthButtons(false);
            // Update unauth prompt count if provided
            currentUnauthPromptCount = userData.unauth_prompt_count || 0;
            if (currentUnauthPromptCount > 0) {
                 usernameDisplay.textContent = `Guest (${5 - currentUnauthPromptCount} left)`;
            } else {
                usernameDisplay.textContent = `Guest`;
            }

        }
    }
    
    // --- Conversation Thread Management ---

    async function loadConversations() {
        if (!localStorage.getItem('user_id')) {
            conversationList.innerHTML = ''; // Clear if not logged in
            return;
        }
        try {
            const response = await fetch('/conversations');
            if (response.ok) {
                const data = await response.json();
                conversationList.innerHTML = ''; // Clear existing list
                if (data.conversations.length === 0) {
                    conversationList.innerHTML = '<li>No conversations yet. Start a new chat!</li>';
                    return;
                }
                data.conversations.forEach(conv => {
                    const li = document.createElement('li');
                    li.dataset.conversationId = conv.id;
                    li.textContent = conv.title;
                    li.onclick = () => switchConversation(conv.id, conv.title);
                    conversationList.appendChild(li);
                });
                // Automatically select the last active conversation if any
                if (currentConversationId) {
                    selectConversationInList(currentConversationId);
                } else if (data.conversations.length > 0) {
                    switchConversation(data.conversations[0].id, data.conversations[0].title);
                }
            } else {
                console.error('Failed to load conversations:', response.statusText);
            }
        } catch (error) {
            console.error('Error loading conversations:', error);
        }
    }

    async function switchConversation(id, title) {
        if (currentConversationId === id) return; // Already on this conversation

        currentConversationId = id;
        currentChatTitle.textContent = title;
        selectConversationInList(id);
        clearChat(); // Clear current chat display

        showLoadingIndicator();
        try {
            const response = await fetch(`/conversations/${id}/messages`);
            if (response.ok) {
                const data = await response.json();
                hideLoadingIndicator();
                data.messages.forEach(msg => {
                    displayMessage(msg.sender, msg.content);
                });
            } else {
                hideLoadingIndicator();
                displayMessage('bot', 'Failed to load messages for this conversation.');
                console.error('Failed to load messages:', response.statusText);
            }
        } catch (error) {
            hideLoadingIndicator();
            displayMessage('bot', 'Error loading messages for this conversation.');
            console.error('Error loading messages:', error);
        }
    }

    function selectConversationInList(id) {
        document.querySelectorAll('#conversation-list li').forEach(li => {
            li.classList.remove('active');
            if (parseInt(li.dataset.conversationId) === id) {
                li.classList.add('active');
            }
        });
    }

    function createNewConversation() {
        currentConversationId = null; // Clear active conversation
        currentChatTitle.textContent = "New Conversation";
        clearChat();
        document.querySelectorAll('#conversation-list li').forEach(li => {
            li.classList.remove('active'); // Deselect all
        });
        userInput.focus();
    }

    // --- Send Message Logic ---

    async function sendMessage() {
        const prompt = userInput.value.trim();
        if (prompt === '') return;

        displayMessage('user', prompt);
        userInput.value = ''; // Clear input immediately
        showLoadingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: prompt, conversation_id: currentConversationId })
            });

            const data = await response.json();
            hideLoadingIndicator();

            if (data.prompt_login_required) {
                displayMessage('bot', data.message);
                showAuthModal('login'); // Show login/signup modal
                return;
            }

            if (data.error) {
                displayMessage('bot', `Error: ${data.error}`);
                console.error('API Error:', data.error);
                return;
            }

            displayMessage('bot', data.response, 'text', data.details);
            
            // Update conversation ID if a new conversation was created
            if (data.conversation_id && !currentConversationId) {
                currentConversationId = data.conversation_id;
                currentChatTitle.textContent = data.new_conversation_title || "New Conversation"; // Set new title
                loadConversations(); // Reload sidebar to show new conversation
            } else if (data.conversation_id && currentConversationId === data.conversation_id && data.new_conversation_title) {
                 // If conversation title was updated after first AI response
                 const existingLi = document.querySelector(`#conversation-list li[data-conversation-id="${currentConversationId}"]`);
                 if(existingLi) {
                     existingLi.textContent = data.new_conversation_title;
                     currentChatTitle.textContent = data.new_conversation_title;
                 }
            }
            
            // Update user info and prompt count for unauthenticated users
            if (data.is_logged_in !== undefined) { // Check if login status was explicitly sent
                 updateUserInfo(data);
            }

        } catch (error) {
            hideLoadingIndicator();
            displayMessage('bot', 'Sorry, I encountered an error communicating with the server.');
            console.error('Fetch Error:', error);
        }
    }

    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    newChatButton.addEventListener('click', createNewConversation);

    toggleSidebarButton.addEventListener('click', () => {
        sidebar.classList.toggle('hidden');
        sidebar.classList.toggle('visible'); // Add a visible class if needed for specific mobile styles
    });

    // Handle initial load and user status check
    async function initializeApp() {
        try {
            const response = await fetch('/user_info');
            const data = await response.json();
            updateUserInfo(data);
            if (data.is_logged_in) {
                loadConversations();
            }
        } catch (error) {
            console.error('Error fetching user info:', error);
            updateUserInfo({ is_logged_in: false }); // Assume not logged in on error
        }
    }

    initializeApp();
});