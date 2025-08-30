// Hent referanser til HTML-elementene
const sessionId = crypto.randomUUID();
const form = document.getElementById('input-form');
const input = document.getElementById('message-input');
const messagesContainer = document.getElementById('chat-messages');
const initialView = document.getElementById('initial-view');
//const insightsButton = document.querySelector('.top-buttons').children[1];
const insightsButton = document.getElementById('insights-button');
const submitButton = document.getElementById('send-button');
const stopButton = document.getElementById('stop-button');
const agentTypeSelect = document.getElementById('agent-type-select'); // NY linje

let agentController;
let currentAgentType = "expert"; // Standardvalg
let isChatActive = false;

// const endpointUrl = 'https://agent-86613370495.europe-west1.run.app/ask-agent'
const endpointUrl = "http://0.0.0.0:8080/ask-agent"

// --- Dropdown Logic ---
const dropdownContainer = document.querySelector('.dropdown-container');
const dropdownBtn = document.getElementById('agent-dropdown-btn');
const dropdownMenu = document.getElementById('agent-dropdown-menu');
const dropdownBtnText = dropdownBtn.querySelector('span');

// Toggle dropdown
dropdownBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Forhindrer at 'window' eventet lukker den med en gang
    dropdownMenu.classList.toggle('hidden');
    dropdownContainer.classList.toggle('open');
});

// Handle selection
dropdownMenu.addEventListener('click', (e) => {
    if (e.target.tagName === 'A') {
        e.preventDefault();
        const selectedText = e.target.textContent;
        //const selectedValue = e.target.dataset.value; // Kan brukes senere
        dropdownBtnText.textContent = selectedText;
        dropdownMenu.classList.add('hidden');
        dropdownContainer.classList.remove('open');
    }
});

// Close dropdown when clicking outside
window.addEventListener('click', () => {
    if (!dropdownMenu.classList.contains('hidden')) {
        dropdownMenu.classList.add('hidden');
        dropdownContainer.classList.remove('open');
    }
});

insightsButton.addEventListener('click', () => { window.location.href = 'src/dashboard.html'; });

stopButton.addEventListener('click', () => {
    if (agentController) {
        agentController.abort(); // Avbryter fetch-kallet
        console.log("Agent request aborted by user.");
    }
});


agentTypeSelect.addEventListener('change', (event) => {
    currentAgentType = event.target.value;
});

// Auto-resize for textarea
function autoResizeTextarea() {
    input.style.height = 'auto';
    input.style.height = input.scrollHeight + 'px';
}
input.addEventListener('input', autoResizeTextarea);

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitButton.click();
    }
});

function activateChat() {
    if (isChatActive) return;
    initialView.classList.add('hidden');
    messagesContainer.classList.remove('hidden');
    addMessageToUI("Hei! Hvordan kan jeg hjelpe deg med boligdata i dag?", 'agent');
    isChatActive = true;
}

function linkify(text) {
    const urlRegex = /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])|(\bwww\.[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig;
    return text.replace(urlRegex, (url) => {
        const fullUrl = url.startsWith('www.') ? 'http://' + url : url;
        return `<a href="${fullUrl}" target="_blank" rel="noopener noreferrer">${url}</a>`;
    });
}

function addMessageToUI(text, sender, options = {}) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);

    const textElement = document.createElement('p');
    textElement.innerHTML = linkify(text);
    messageElement.appendChild(textElement);

    if (options.isThinking) {
        const details = document.createElement('details');
        details.className = 'thinking-details';
        details.open = true;

        const summary = document.createElement('summary');
        summary.className = 'thinking-summary';

        const spinner = document.createElement('div');
        spinner.className = 'spinner';

        const summaryText = document.createElement('span');
        summaryText.textContent = 'Agenten tenker...';

        summary.appendChild(spinner);
        summary.appendChild(summaryText);

        const content = document.createElement('div');
        content.className = 'thinking-content';

        details.appendChild(summary);
        details.appendChild(content);
        messageElement.appendChild(details);
    }

    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return messageElement;
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const messageText = input.value.trim();
    if (messageText === '') return;

    if (!isChatActive) activateChat();

    addMessageToUI(messageText, 'user');
    input.value = '';
    autoResizeTextarea();
    input.focus();

    // Veksle knapper: vis stopp, skjul send
    stopButton.classList.remove('hidden');
    submitButton.classList.add('hidden');

    const agentMessageElement = addMessageToUI("...", 'agent', { isThinking: true });
    const agentTextElement = agentMessageElement.querySelector('p');
    const thinkingDetails = agentMessageElement.querySelector('.thinking-details');
    const thinkingContent = agentMessageElement.querySelector('.thinking-content');
    const thinkingSpinner = agentMessageElement.querySelector('.spinner');
    const thinkingSummaryText = agentMessageElement.querySelector('.thinking-summary span');

    agentController = new AbortController(); // Opprett en ny controller for dette kallet

    try {
        const response = await fetch(endpointUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: messageText,
                session_id: sessionId,
                agent_type: currentAgentType // Legg til denne linjen
            }),
            signal: agentController.signal // Koble controlleren til fetch-kallet
        });

        if (!response.ok) throw new Error(`Serverfeil: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let finalAnswer = "";
        let isFirstChunk = true;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            if (isFirstChunk) {
                agentTextElement.textContent = "";
                isFirstChunk = false;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n\n').filter(line => line.startsWith('data:'));

            for (const line of lines) {
                const data = line.substring(5).trim();
                const parsedData = JSON.parse(data);
                if (parsedData.type === 'final_answer') {
                    finalAnswer += parsedData.content + '\n';
                } else if (parsedData.type === 'status') {
                    const logEntry = document.createElement('p');
                    logEntry.textContent = parsedData.content;
                    thinkingContent.appendChild(logEntry);
                    thinkingContent.scrollTop = thinkingContent.scrollHeight;
                }
            }
        }

        agentTextElement.innerHTML = linkify(finalAnswer.trim() || "Fikk ikke et gyldig svar.");

    } catch (error) {
        if (error.name === 'AbortError') {
            agentTextElement.textContent = "Agentens svar ble avbrutt.";
        } else {
            console.error("Feil under strømming:", error);
            agentTextElement.textContent = "Beklager, en teknisk feil oppstod.";
        }
    } finally {
        // Rydd opp og veksle knapper tilbake, uansett resultat
        thinkingSpinner.classList.add('hidden');
        thinkingSummaryText.textContent = 'Vis tankeprosess';
        thinkingDetails.open = false;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        stopButton.classList.add('hidden');
        submitButton.classList.remove('hidden');
    }
});