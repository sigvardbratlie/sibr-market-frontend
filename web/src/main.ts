import "style.css"

// Hent referanser til HTML-elementene
const sessionId = crypto.randomUUID();
const form = document.getElementById('input-form') as HTMLFormElement;
const input = document.getElementById('message-input') as HTMLTextAreaElement;
const messagesContainer = document.getElementById('chat-messages') as HTMLElement;
const initialView = document.getElementById('initial-view') as HTMLElement;
const insightsButton = document.querySelector('.top-buttons')?.children[1] as HTMLButtonElement;
const submitButton = document.getElementById('send-button') as HTMLButtonElement;
const stopButton = document.getElementById('stop-button') as HTMLButtonElement;
let agentController: AbortController;

// Sjekker at skjemaet faktisk finnes for å unngå krasj
if (!form) {
    throw new Error("Kritisk feil: Fant ikke skjemaet med id 'input-form'.");
}

insightsButton.addEventListener('click', () => {
    window.location.href = 'src/insights.html';
});

let isChatActive = false;

stopButton.addEventListener('click', () => {
    if (agentController) {
        agentController.abort();
        console.log("Agent request aborted by user.");
    }
});

function autoResizeTextarea() {
    input.style.height = 'auto';
    input.style.height = `${input.scrollHeight}px`;
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

function linkify(text: string): string {
    const urlRegex = /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])|(\bwww\.[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig;
    return text.replace(urlRegex, (url) => {
        const fullUrl = url.startsWith('www.') ? `http://${url}` : url;
        return `<a href="${fullUrl}" target="_blank" rel="noopener noreferrer">${url}</a>`;
    });
}

function addMessageToUI(text: string, sender: 'user' | 'agent', options: { isThinking?: boolean } = {}): HTMLElement {
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

    stopButton.classList.remove('hidden');
    submitButton.classList.add('hidden');

    const agentMessageElement = addMessageToUI("...", 'agent', { isThinking: true });
    const agentTextElement = agentMessageElement.querySelector('p') as HTMLParagraphElement;
    // KORRIGERT LINJE:
    const thinkingDetails = agentMessageElement.querySelector('.thinking-details') as HTMLDetailsElement;
    const thinkingContent = agentMessageElement.querySelector('.thinking-content') as HTMLElement;
    const thinkingSpinner = agentMessageElement.querySelector('.spinner') as HTMLElement;
    const thinkingSummaryText = agentMessageElement.querySelector('.thinking-summary span') as HTMLElement;

    agentController = new AbortController();

    try {
        const response = await fetch('https://agent-homes-86613370495.europe-west1.run.app/ask-agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: messageText, session_id: sessionId }),
            signal: agentController.signal
        });

        if (!response.ok) throw new Error(`Serverfeil: ${response.status}`);
        if (!response.body) throw new Error("Response body is missing");

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
                const jsonString = line.substring(5).trim();
                if (!jsonString) continue;

                try {
                    const data = JSON.parse(jsonString);
                    if (data.type === 'final_answer') {
                        finalAnswer += data.content;
                    } else if (data.type === 'status') {
                        const logEntry = document.createElement('p');
                        logEntry.textContent = data.content;
                        thinkingContent.appendChild(logEntry);
                        thinkingContent.scrollTop = thinkingContent.scrollHeight;
                    }
                } catch (e) {
                    console.error("Failed to parse JSON from stream:", jsonString, e);
                }
            }
        }

        agentTextElement.innerHTML = linkify(finalAnswer.trim() || "Fikk ikke et gyldig svar fra agenten.");

    } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
            agentTextElement.textContent = "Agentens svar ble avbrutt.";
        } else {
            console.error("Feil under strømming:", error);
            agentTextElement.textContent = "Beklager, en teknisk feil oppstod.";
        }
    } finally {
        thinkingSpinner.classList.add('hidden');
        thinkingSummaryText.textContent = 'Vis tankeprosess';
        // Denne linjen vil nå fungere korrekt:
        thinkingDetails.open = false;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        stopButton.classList.add('hidden');
        submitButton.classList.remove('hidden');
    }
});