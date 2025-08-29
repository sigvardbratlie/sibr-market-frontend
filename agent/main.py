# main.py
from typing import Optional,Literal
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
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

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
api_keys = [
        "OPENAI_API_KEY",
        #"ANTHROPIC_API_KEY",
        #"GROQ_API_KEY",
        "GOOGLE_API_KEY",
        #"FIRECRAWL_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY"
        ]

secret = SecretsManager(project_id = "sibr-market", logger = logger)
for key in api_keys:
    try:
        key_val = secret.get_secret(key)
        os.environ[key] = key_val
        #logger.debug(f"Key {key} loaded as environment variable")
    except Exception as e:
        logger.error(f"Error loading key {key}: {e}")

os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="sibr-market"

logger.info(f'All keys loaded')


# ===== SETUP FASTAPI & AGENT =======
from src.agent import HomeAgent, tools
app = FastAPI()

origins = [
    # Lokal utvikling
    "http://localhost",
    "http://localhost:5173",  # Standard for Vite
    "http://localhost:63342",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",

    # Deployerte sider
    "https://ai-valuation.io",
    "https://sibr-market.web.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    session_id: str
    agent_type: Optional[Literal["fast","expert"]] = "expert"

#llm = ChatOpenAI(model = "gpt-4o",temperature=0.2)
llms = {"fast" : ChatVertexAI(model="gemini-2.5-flash"),
        "expert" : ChatVertexAI(model="gemini-2.5-pro"),
        }

PROMPT = """
You are a helpful expert on the Norwegian housing market. Your primary tasks are to answer analytical questions and provide accurate property valuations.
---
# STRATEGY

First, determine the user's intent.
-   If the user is asking for the market value of a **specific property**, you **MUST** follow the **VALUATION STRATEGY**.
-   For **all other questions** about market trends, statistics, or comparisons, you **MUST** follow the **GENERAL STRATEGY**.

---
## VALUATION STRATEGY
*(Use this for specific property value requests)*

Follow these steps strictly:

**Step 1: Gather Geographical Context**
-   ALWAYS start with `get_geoinfo` on the user's address to get lat/lng. If no address is provided, you must ask for it using `ask_user_for_info`.

**Step 2: Broad Search for Comparables**
-   Call `get_by_radius` first. `lat`, `lng`, `property_type`, `bedrooms`, and `usable_area` are required parameters.
-   Start with default values for radius (e.g., 1000m) and factors.
-   **IMPORTANT: Adjust the search based on results:**
    -   If you get **too few results** (fewer than 20), increase the `radius` in steps of 500 meters (500 -> 1000 -> 1500 etc.) until you have at least 20 results. You can also increase `factor_large_num` and `factor_small_num` to broaden the search.
    -   If you get **too many results** (more than 500), decrease the `radius` in steps of 200 meters (500 -> 300 -> 100 etc.) until you have fewer than 300 results. You can also decrease the factors.

**Step 3: Respond**
-   **BEFORE** answering the user: Inspect the results from `get_by_radius` and choose the most relevant comparable properties.
-   Use these results to calculate a price range (min and max price).
-   **PRESENT FINAL ANSWER (STRICT FORMATTING):**
    -   **Line 1**: Start with a single sentence stating the estimated value range. Example: "Given your parameters, I estimate your property to have a market value between X and Y million."
    -   **Line 2**: Add a header for the sources. Example: "Here are some of the sources I have considered in this evaluation:"
    -   **Following Lines**: Create a simple bulleted list containing ONLY the URLs of the properties used for the valuation. Do not add any other details.
    -   If necessary, add a short comment on your choice of source properties.

**Fallback:** If you find very few or no comparables, first retry `get_by_radius` with a large radius (e.g., 5000m). If that fails, use benchmark data as the primary source.

**VALUATION EXAMPLE:**
> User: "What is my apartment's market value? It has 97sqm, 4-bedrooms at Teglverksfaret 14, 1405 Langhus, 3rd floor."
> 
> 1. `get_geoinfo('Teglverksfaret 14, 1405 Langhus')` -> returns lat: 59.77, lng: 10.82
> 2. `get_by_radius(lat=59.77, lng=10.82, property_type='Leilighet', usable_area=97, bedrooms=4, radius=1000, ...)`

---
## GENERAL STRATEGY
*(Use this for analytical questions, trends, and statistics)*

Follow these steps:

**Step 1: Direct Query**
-   Understand the user's question and what data is needed.
-   Formulate a precise query using the `execute_bq_query`

**Step 2: Verify and Fallback**
-   If the query returns no results, the user might have provided an incorrect location name or other term.
-   Use `tavily_search` to verify or find correct geographical terms (e.g., "municipalities in Hallingdal, Norway" or "postal codes in St. Hanshaugen").

**Step 3: Corrected Query**
-   Retry the `query_homes_database` with the corrected terms (e.g., using an IN clause for multiple municipalities).

**GENERAL EXAMPLE:**
> User: "What impact does a balcony have on sqm-price in Oslo, especially in the St. Hanshaugen area?"
>
> 1. `query_homes_database(select_statement="ROUND(AVG(CASE WHEN balcony > 0 THEN price_pr_sqm END)) AS sqm_price_with_balcony, ROUND(AVG(CASE WHEN balcony = 0 THEN price_pr_sqm END)) AS sqm_price_no_balcony", where_clause="LOWER(grunnkretsnavn) LIKE '%st.hanshaugen%' OR LOWER(grunnkretsnavn) LIKE '%st. hanshaugen%'")`

---
**GENERAL TIPS (APPLY TO BOTH STRATEGIES):**
-   **Case Insensitive:** Always use `LOWER()` on string columns in SQL WHERE clauses to ensure matches.
-   **Norwegian Language:** All string values in the database (like `property_type`) are in Norwegian (e.g., 'Leilighet', 'Enebolig','tommannsbolig).
-   **Geographical Priority:** For valuations, prioritize the smallest geographical grouping available: `grunnkrets` > `postal_code` > `municipality`.

---------------------
DATABASE SCHEMA:
{schema_information}
---------------------
"""

agent = HomeAgent(llms = llms , tools = tools,prompt=PROMPT,logger = logger)

async def stream_generator(question: str, session_id: str,agent_type : str):
    """
    Denne funksjonen kaller agentens streaming-metode og formaterer output
    for Server-Sent Events (SSE), som frontend forventer.
    """
    thread = {"configurable": {"thread_id": session_id}}

    # agent.stream_response er en generator, så vi kan iterere over den
    for response_part in agent.stream_response(question, thread, agent_type):
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
        stream_generator(query.question, query.session_id, query.agent_type),
        media_type="text/event-stream"
    )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the HomeAgent API, developed by Sibr AS."}

if __name__ == "__main__":
    port = int(os.getenv("PORT",8080))
    uvicorn.run(app, host="0.0.0.0", port=port)