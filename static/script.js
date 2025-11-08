class ChatApp {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.clearButton = document.getElementById('clearButton');
        this.chatMessages = document.getElementById('chatMessages');
        
        this.initEventListeners();
        this.loadChatHistory();
    }
    
    initEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        // Clear chat
        this.clearButton.addEventListener('click', () => this.clearChat());
        
        // Auto-focus input
        this.messageInput.focus();
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.setLoading(true);
        
        try {
            const response = await fetch('/send_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Update chat with agent response
                this.addMessage('agent', data.response);
                this.saveChatHistory();
            } else {
                throw new Error(data.error || 'Failed to send message');
            }
            
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('agent', 'Sorry, I encountered an error. Please try again.');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = message;
        
        messageDiv.appendChild(contentDiv);
        this.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    setLoading(loading) {
        this.sendButton.disabled = loading;
        this.messageInput.disabled = loading;
        this.sendButton.textContent = loading ? 'Sending...' : 'Send';
    }
    
    async clearChat() {
        try {
            const response = await fetch('/clear_chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (response.ok) {
                this.chatMessages.innerHTML = '';
                // Add welcome message
                this.addMessage('agent', 'Hello! I can help you check if you can return a product. Please tell me what product you\'d like to return.');
                this.saveChatHistory();
            }
        } catch (error) {
            console.error('Error clearing chat:', error);
        }
    }
    
    async loadChatHistory() {
        try {
            const response = await fetch('/get_chat_history');
            const data = await response.json();
            
            if (response.ok && data.chat_history.length > 0) {
                this.chatMessages.innerHTML = '';
                data.chat_history.forEach(msg => {
                    this.addMessage(msg.sender, msg.message);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    saveChatHistory() {
        // History is automatically saved in session
    }
}

// Initialize the chat app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
