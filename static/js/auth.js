// static/js/auth.js
// static/js/auth.js

// Declare variables globally if they need to be accessed by multiple functions,
// or inside the DOMContentLoaded listener if they are only used there.
// For now, let's put them inside to ensure they are defined after HTML loads.

document.addEventListener('DOMContentLoaded', function() {
    // Get references to modal elements
    const authModal = document.getElementById('authModal');
    const closeButton = authModal.querySelector('.close-button');
    const authButton = document.getElementById('auth-button');
    const logoutButton = document.getElementById('logout-button');

    // Get references to step containers
    const authStep1 = document.getElementById('auth-step-1');
    const authStep2 = document.getElementById('auth-step-2');
    const authTypeSelect = document.getElementById('auth-type-select');
    const authError = document.getElementById('auth-error');

    // Form elements for Step 1
    const authFormStep1 = document.getElementById('authFormStep1');
    const authUsername = document.getElementById('auth-username'); // This is the one that was 'not defined'
    const authEmail = document.getElementById('auth-email');
    const emailLabel = document.getElementById('email-label');
    const authPassword = document.getElementById('auth-password');
    const authConfirmPassword = document.getElementById('auth-confirm-password');
    const confirmPasswordLabel = document.getElementById('confirm-password-label');

    // Navigation buttons for Step 1
    const gotoLoginButton = document.getElementById('goto-login');
    const gotoRegisterButton = document.getElementById('goto-register');

    // Avatar selection elements for Step 2
    const avatarOptionsContainer = document.getElementById('avatar-options');
    let selectedAvatarUrl = '/static/avatars/default.png'; // Default selected avatar

    // Modal footer buttons
    const backButton = document.getElementById('back-button');
    const nextButton = document.getElementById('next-button');
    const submitButton = document.getElementById('submit-button');

    let currentAuthMode = 'login'; // 'login' or 'register'
    let currentStep = 1; // 1 or 2

    // --- Helper Functions ---
    function showModal() {
        authModal.style.display = 'flex';
        resetModal(); // Reset modal state when showing
    }

    function hideModal() {
        authModal.style.display = 'none';
        resetModal(); // Reset modal state when hiding
    }

    function resetModal() {
        // This is the line that was throwing the error! It needs to be inside DOMContentLoaded.
        authFormStep1.reset();
        authError.textContent = '';
        authEmail.style.display = 'none';
        emailLabel.style.display = 'none';
        authConfirmPassword.style.display = 'none';
        confirmPasswordLabel.style.display = 'none';

        gotoLoginButton.style.display = 'none'; // Initially hide, show based on mode
        gotoRegisterButton.style.display = 'block'; // Initially show

        // Reset to default step and mode
        currentStep = 1;
        currentAuthMode = 'login'; // Default to login when opening
        updateModalView();
    }

    function updateModalView() {
        // Adjust UI based on currentAuthMode and currentStep
        authStep1.style.display = (currentStep === 1) ? 'block' : 'none';
        authStep2.style.display = (currentStep === 2) ? 'block' : 'none';

        if (currentAuthMode === 'login') {
            authTypeSelect.textContent = 'Login';
            emailLabel.style.display = 'none';
            authEmail.style.display = 'none';
            confirmPasswordLabel.style.display = 'none';
            authConfirmPassword.style.display = 'none';
            gotoLoginButton.style.display = 'none';
            gotoRegisterButton.style.display = 'block';

            backButton.style.display = 'none';
            nextButton.style.display = 'none';
            submitButton.style.display = 'block';

            // Clear email and confirm password fields if switching to login
            authEmail.value = '';
            authConfirmPassword.value = '';

        } else { // register mode
            authTypeSelect.textContent = 'Register';
            emailLabel.style.display = 'block';
            authEmail.style.display = 'block';
            confirmPasswordLabel.style.display = 'block';
            authConfirmPassword.style.display = 'block';
            gotoLoginButton.style.display = 'block';
            gotoRegisterButton.style.display = 'none';

            if (currentStep === 1) {
                backButton.style.display = 'none';
                nextButton.style.display = 'block';
                submitButton.style.display = 'none';
            } else { // currentStep === 2
                backButton.style.display = 'block';
                nextButton.style.display = 'none';
                submitButton.style.display = 'block';
            }
        }
    }

    async function populateAvatars() {
        // In a real app, you might fetch available avatars from the server
        const avatarPaths = [
            '/static/avatars/default.png',
            '/static/avatars/avatar1.png',
            '/static/avatars/avatar2.png',
            '/static/avatars/avatar3.png',
            '/static/avatars/avatar4.png'
        ];

        avatarOptionsContainer.innerHTML = ''; // Clear existing
        avatarPaths.forEach(path => {
            const img = document.createElement('img');
            img.src = path;
            img.alt = 'Avatar';
            img.classList.add('avatar-option');
            img.dataset.url = path;
            if (path === selectedAvatarUrl) {
                img.classList.add('selected');
            }
            img.addEventListener('click', () => {
                document.querySelectorAll('.avatar-option').forEach(opt => opt.classList.remove('selected'));
                img.classList.add('selected');
                selectedAvatarUrl = path;
            });
            avatarOptionsContainer.appendChild(img);
        });
    }

    // --- Event Listeners ---
    authButton.addEventListener('click', showModal);
    closeButton.addEventListener('click', hideModal);

    // Close modal if clicking outside content
    window.addEventListener('click', function(event) {
        if (event.target === authModal) {
            hideModal();
        }
    });

    gotoLoginButton.addEventListener('click', function() {
        currentAuthMode = 'login';
        currentStep = 1;
        updateModalView();
    });

    gotoRegisterButton.addEventListener('click', function() {
        currentAuthMode = 'register';
        currentStep = 1; // Always start at step 1 for registration
        updateModalView();
    });

    nextButton.addEventListener('click', function() {
        if (currentAuthMode === 'register' && currentStep === 1) {
            // Validate step 1 fields before moving to step 2
            const username = authUsername.value.trim();
            const email = authEmail.value.trim();
            const password = authPassword.value;
            const confirmPassword = authConfirmPassword.value;

            if (!username || !email || !password || !confirmPassword) {
                authError.textContent = 'All fields are required for registration.';
                return;
            }
            if (password !== confirmPassword) {
                authError.textContent = 'Passwords do not match.';
                return;
            }
            if (password.length < 6) {
                authError.textContent = 'Password must be at least 6 characters.';
                return;
            }
            // Basic email validation
            if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
                authError.textContent = 'Please enter a valid email address.';
                return;
            }
            
            authError.textContent = ''; // Clear any previous errors
            currentStep = 2;
            populateAvatars(); // Populate avatars when moving to step 2
            updateModalView();
        }
    });

    backButton.addEventListener('click', function() {
        if (currentAuthMode === 'register' && currentStep === 2) {
            currentStep = 1;
            updateModalView();
        }
    });

    submitButton.addEventListener('click', async function(event) {
        event.preventDefault(); // Prevent default form submission

        const username = authUsername.value.trim();
        const password = authPassword.value;
        let url = '';
        let payload = {};

        if (currentAuthMode === 'login') {
            url = '/login';
            payload = { username, password };
        } else { // register
            url = '/register';
            const email = authEmail.value.trim();
            const confirmPassword = authConfirmPassword.value;

            if (password !== confirmPassword) {
                authError.textContent = 'Passwords do not match.';
                return;
            }
            if (password.length < 6) {
                authError.textContent = 'Password must be at least 6 characters.';
                return;
            }
            
            payload = { username, email, password, avatar_url: selectedAvatarUrl };
        }

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.success) {
                authError.textContent = ''; // Clear errors
                hideModal();
                updateUserDisplay(data.user); // Update UI based on new user
                // Optionally reload conversations or update sidebar
                // This might be better handled by a function in script.js or a global update
                if (typeof window.fetchConversations === 'function') {
                    window.fetchConversations();
                }
                if (typeof window.updateChatLimitDisplay === 'function') {
                    window.updateChatLimitDisplay();
                }

                console.log(data.message);
                // Also update the logout button visibility etc.
                authButton.style.display = 'none';
                logoutButton.style.display = 'block';
                document.getElementById('new-chat-button').style.display = 'block';

            } else {
                authError.textContent = data.message;
            }
        } catch (error) {
            console.error('Auth error:', error);
            authError.textContent = 'An error occurred during authentication. Please try again.';
        }
    });

    // Initial update when modal is loaded
    updateModalView();

    // Function to update user display after login/register
    function updateUserDisplay(user) {
        const usernameDisplay = document.getElementById('username-display');
        const userAvatar = document.getElementById('user-avatar');

        if (user) {
            usernameDisplay.textContent = user.username;
            userAvatar.src = user.avatar_url;
        } else {
            usernameDisplay.textContent = 'Guest';
            userAvatar.src = '/static/avatars/default.png';
        }
    }
    
    // Initial fetch of user info on page load
    async function fetchUserInfo() {
        try {
            const response = await fetch('/user_info');
            const data = await response.json();
            if (data.is_logged_in) {
                updateUserDisplay(data.user);
                authButton.style.display = 'none';
                logoutButton.style.display = 'block';
                document.getElementById('new-chat-button').style.display = 'block';
            } else {
                authButton.style.display = 'block';
                logoutButton.style.display = 'none';
                document.getElementById('new-chat-button').style.display = 'none';
            }
        } catch (error) {
            console.error('Failed to fetch user info:', error);
            authButton.style.display = 'block';
            logoutButton.style.display = 'none';
        }
    }
    fetchUserInfo(); // Call this on DOMContentLoaded

    // Event listener for logout button
    logoutButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/logout', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                console.log(data.message);
                updateUserDisplay(null); // Reset user display
                authButton.style.display = 'block';
                logoutButton.style.display = 'none';
                document.getElementById('new-chat-button').style.display = 'none';
                document.getElementById('conversation-list').innerHTML = '<li>Sign in to save and manage your conversations.</li>'; // Clear conversations
                // Potentially reload the page or reset the chat area for guest mode
                window.location.reload(); // Simple reload to go back to guest state
            } else {
                console.error("Logout failed:", data.message);
            }
        } catch (error) {
            console.error("Error during logout:", error);
        }
    });

}); 