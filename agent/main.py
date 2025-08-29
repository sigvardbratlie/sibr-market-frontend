# main.py
from sibr_module import SecretsManager,Logger
from dotenv import load_dotenv
import asyncio
import json
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path

load_dotenv()
cred_filename = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_FILENAME")

if cred_filename:
    print(f'RUNNING LOCAL. ADAPTING LOADING PROCESS')
    project_root = Path(__file__).parent
    os.chdir(project_root)
    dotenv_path = project_root.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root.parent / cred_filename)

logger = Logger("HomesAgent-API")
logger.set_level("INFO")

# ===== SETUP ENV VARIABLES =======
api_keys = ["OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "FIRECRAWL_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY"]

secret = SecretsManager(logger = logger)
for key in api_keys:
    try:
        key_val = secret.get_secret(key)
        os.environ[key] = key_val
        logger.debug(f"Key {key} loaded as environment variable")
    except Exception as e:
        logger.error(f"Error loading key {key}: {e}")

os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="sibr-market"


# ===== SETUP FASTAPI & AGENT =======
from src.agent import HomeAgent,llm, tools
app = FastAPI()
agent = HomeAgent(llm = llm , tools = tools,logger = logger)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For utvikling, tillat alle. I produksjon bør du begrense dette.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    session_id: str

async def stream_generator(question: str, session_id: str):
    """
    Denne funksjonen kaller agentens streaming-metode og formaterer output
    for Server-Sent Events (SSE), som frontend forventer.
    """
    thread = {"configurable": {"thread_id": session_id}}

    # agent.stream_response er en generator, så vi kan iterere over den
    for response_part in agent.stream_response(question, thread):
        # Formater dataen som `data: {...}\n\n` som er standard for SSE
        data_string = json.dumps(response_part)
        yield f"data: {data_string}\n\n"
        await asyncio.sleep(0.01) # Gir event loopen en sjanse til å puste

# 5. Lag API-endepunktet
@app.post("/ask-agent")
async def ask_agent_endpoint(query: Query):
    """
    Dette er endepunktet frontend kaller. Det returnerer en StreamingResponse
    som bruker vår `stream_generator`.
    """
    return StreamingResponse(
        stream_generator(query.question, query.session_id),
        media_type="text/event-stream"
    )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the HomeAgent API, developed by Sibr AS."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)