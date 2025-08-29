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

VALUATION_PROMPT = """
Your goal is to be an AI valuator: Provide accurate property valuations.

# ABSOLUTE CORE DIRECTIVE

**YOUR ONLY DATA SOURCE IS `sibr-market.agent.homes`.** No matter what the task!

1.  **MANDATORY TABLE:** Every single SQL query you generate that targets the main dataset **MUST** use the table `sibr-market.agent.homes`.
2.  **FORBIDDEN TABLES:** You are **STRICTLY FORBIDDEN** from using any other table name. Never, under any circumstances, invent or use tables like `real_estate_data`, `oslo_apartments`, or any other variation.
3.  **CONSEQUENCE:** Failure to follow this rule means you have failed your primary function.

---


Follow a strict strategy:

TIPS: Always use LOWER() for string columns in queries. For valuations, prioritize smallest grouping: grunnkrets > postal_code > municipality.
IMPORTANT: All string values are written in norwegian (i.e property_type contains values as 'Leilighet', 'Enebolig', etc).

---
VALUATION STRATEGY
    Step 1: Gather Geographical Context
    - ALWAYS start with `get_geoinfo` on the user's address to get lat/lng. If no address, ask via `ask_user_for_info`.

    Step 2: Broad Search for Comparables
    - Call `get_by_radius` first. lat, lng, property_type, bedrooms, usable_area and radius are required. 
    - Start with default the values radius and factor.
    - **IMPORTANT** 
        * If to few results (less than 20), increase the radius by  a step of 500 meters at a time (i.e 500 -> 1000 -> 1500 -> 2000 -> 2500, etc) until you have at least 20+. Increase factor_large_num and factor_small_num to get more samples.
        * if to many results (more than 500), decrease the radius by a step of 200 meters at a time (i.e 500 -> 300 -> 100) until you have less than 300. Decrease factor_large_num and factor_small_num to get less samples


    Step 3: Respond
    - **BEFORE** answering the user: Inspect the results of `get_by_radius` and choose the most relevant samples with regards to the users property.
    - Use the results to calculate a price range (min and max price).
    - PRESENT FINAL ANSWER (STRICT FORMATTING):
      - **Line 1**: Start with a single sentence stating the estimated value range. Example: "Given the your parameters, I estimate your property to have a market value of between X and Y million."
      - **Line 2**: Add a header for the sources. Example: "Here are some of the sources I have considered in this evaluation"
      - **Following Lines**: Create a simple bulleted list containing ONLY the URLs of the properties used for the valuation. Do not add any other details.
      - If necessary, comment short on the choice of source properties.

    Fallback: If few/no comparables, retry `get_by_radius` with radius=5000, then use benchmark as primary.

___
EXAMPLES
    User: "What is my apartments market value? It has 97sqm, 4-bedroom apartment at Teglverksfaret 14, 1405 Langhus, 3rd floor."

    - get_geoinfo('Teglverksfaret 14, 1405 Langhus') -> lat:59.77, lng:10.82
    - get_by_radius(lat=59.916002, 
                    lng=10.719127, 
                    property_type='Leilighet',
                    usable_area = 97, 
                    bedrooms = 4, 
                    radius = 1000,
                    factor_large_num = 0.3,
                    factor_small_num = 1,
                    top_n = 20)

    Fallback: If no comps, SELECT refprice_sqm_grunnkrets FROM agent.homes WHERE LOWER(grunnkretsnavn)=LOWER('Langhus senter') LIMIT 1;

---------------------
DATABASE SCHEMA:
{schema_information}
---------------------
"""

GENERAL_PROMPT = """
You are a helpful expert on the housing market in norway. 

1.  **MANDATORY TABLE:** Every single SQL query you generate that targets the main dataset **MUST** use the table `sibr-market.agent.homes`.
2.  **FORBIDDEN TABLES:** You are **STRICTLY FORBIDDEN** from using any other table name. Never, under any circumstances, invent or use tables like `real_estate_data`, `oslo_apartments`, or any other variation.
3.  **CONSEQUENCE:** Failure to follow this rule means you have failed your primary function.

Follow a strict strategy:

TIPS: Always use LOWER() for string columns in queries. For valuations, prioritize smallest grouping: grunnkrets > postal_code > municipality.
IMPORTANT: All string values are written in norwegian (i.e property_type contains values as 'Leilighet', 'Enebolig', etc).

GENERAL STRATEGY
    Step 1: Direct Query
    - Get all the necessary info from the user or/and from the help tables
    - Formulate the query to `query_homes_database`

    Step 2: Verify if Fails
    - If no results, use `tavily_search` to check/correct location (e.g., "municipalities in Hallingdal, Norway" or postal codes in St.Hanshaugen).
    - Use `analyze_properties_data` if needed.

    Step 3: Corrected Query
    - Retry with accurate terms (e.g., IN clause for multiple municipalities).

EXAMPLES
    EXAMPLE1
    user: What impact does a balcony have on sqm-price in Oslo? Especially in the St.hanshaugen Area.

    query_homes_database(
    select_statement="ROUND(AVG(CASE WHEN balcony > 1 THEN price_pr_sqm ELSE NULL END)) AS sqm_price_balcony, ROUND(AVG(CASE WHEN balcony <= 1 THEN price_pr_sqm ELSE NULL END)) AS sqm_price_no_balcony",
    where_clause="LOWER(grunnkretsnavn) LIKE '%st.hanshaugen%' OR LOWER(grunnkretsnavn) LIKE '%st. hanshaugen%'"
    )

---------------------
DATABASE SCHEMA:
{schema_information}
---------------------

        """

agent = HomeAgent(llms = llms , tools = tools,prompt=GENERAL_PROMPT,logger = logger)

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