document.addEventListener('DOMContentLoaded', function () {
    const chatBtn = document.querySelector('.chat-btn');
    const chatWidget = document.querySelector('.chat-widget');
    const closeChat = document.querySelector('.close-chat');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatBody = document.querySelector('.chat-body');

    const micBtn = document.getElementById('micBtn');

    // Toggle Chat
    function toggleChat() {
        chatWidget.classList.toggle('active');
    }

    chatBtn.addEventListener('click', toggleChat);
    closeChat.addEventListener('click', toggleChat);

    // Voice Recognition (Speech to Text)
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = function () {
            micBtn.classList.add('listening');
        };

        recognition.onend = function () {
            micBtn.classList.remove('listening');
        };

        recognition.onresult = function (event) {
            const transcript = event.results[0][0].transcript;
            chatInput.value = transcript;
            sendMessage();
        };

        micBtn.addEventListener('click', function () {
            recognition.start();
        });
    } else {
        micBtn.style.display = 'none'; // Hide if not supported
        console.log("Web Speech API not supported.");
    }

    // Text to Speech
    function speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        }
    }

    // Send Message
    function sendMessage() {
        const message = chatInput.value.trim();
        if (message) {
            // Append User Message
            appendMessage(message, 'user');
            chatInput.value = '';

            // Send to Backend
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
                .then(response => response.json())
                .then(data => {
                    // Append Bot Response
                    appendMessage(data.response, 'bot');
                    speak(data.response); // Read out the response
                })
                .catch(error => {
                    console.error('Error:', error);
                    appendMessage('Sorry, I am having trouble connecting.', 'bot');
                });
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function appendMessage(text, sender) {
        const div = document.createElement('div');
        div.classList.add('message', sender);
        div.textContent = text;
        chatBody.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
});
