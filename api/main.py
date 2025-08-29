# main.py
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import bigquery
from google.cloud.bigquery import ScalarQueryParameter, QueryJobConfig
from google.api_core.exceptions import GoogleAPICallError
import pandas as pd
from typing import Optional, List, Dict, Any, Literal
import logging
import numpy as np
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

load_dotenv()
cred_filename = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_FILENAME")

if cred_filename:
    print(f'RUNNING LOCAL. ADAPTING LOADING PROCESS')
    project_root = Path(__file__).parent
    os.chdir(project_root)
    dotenv_path = project_root.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root.parent / cred_filename)


# Konfigurerer logging for å se detaljerte feilmeldinger i terminalen
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser FastAPI-appen
api = FastAPI(
    title="Boligdata API",
    description="Et API for å hente aggregerte boligdata fra BigQuery.",
    version="1.0.0",
)

# --- 2. LEGG TIL CORS MIDDLEWARE ---
# Definer hvilke opprinnelser (frontends) som får lov til å snakke med dette API-et.
# '*' betyr "alle", som er greit for utvikling.
origins = [
    "http://localhost:5173", # Standard port for Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost",
    "https://ai-valuation.io/",
    "https://ai-valuation.io/homes/",
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Tillat alle metoder (GET, POST, etc.)
    allow_headers=["*"], # Tillat alle headere
)
# ------------------------------------

try:
    client = bigquery.Client()
    logger.info("BigQuery client initialized successfully.")
except Exception as e:
    logger.error(f"Could not initialize BigQuery client: {e}")
    client = None


# --- Hjelpefunksjon for å kjøre spørringer ---
def run_query(query: str, params: Optional[List[ScalarQueryParameter]] = None) -> pd.DataFrame:
    """
    Kjører en BigQuery-spørring sikkert med parametere og returnerer resultatet som en Pandas DataFrame.
    """
    if not client:
        raise HTTPException(status_code=500, detail="BigQuery client is not available.")

    try:
        # Konfigurerer spørringen med parametere for å unngå SQL-injeksjon
        job_config = QueryJobConfig(query_parameters=params) if params else None

        # Kjører spørringen
        logger.info(f"Executing query: {query}")
        query_job = client.query(query, job_config=job_config)

        # Venter på at jobben skal fullføre og henter resultatene
        results = query_job.to_dataframe()
        return results
    except GoogleAPICallError as e:
        # Håndterer spesifikke API-feil fra Google Cloud og sender detaljene tilbake
        logger.error(f"A BigQuery API error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"BigQuery API error: {str(e)}")
    except Exception as e:
        # Håndterer andre generelle feil og sender detaljene tilbake
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- Konfigurasjon for aggregerte endepunkter ---
# Denne strukturen gjør det enkelt å legge til nye aggregeringsnivåer eller boligtyper
AGGREGATION_CONFIG = {
    "municipality": {
        "filter_column": "municipality",
        "param_type": "STRING",
        "tables": {
            "houses": "`sibr-market.api.homes_houses_municipality`",
            "apartments": "`sibr-market.api.homes_apartments_municipality`",
        },
    },
    "postal_code": {
        "filter_column": "postal_code",
        "param_type": "STRING",  # Postnummer kan ha ledende nuller, så STRING er tryggest
        "tables": {
            "houses": "`sibr-market.api.homes_houses_postal`",
            "apartments": "`sibr-market.api.homes_apartments_postal`",
        },
    },
    "grunnkrets": {
        "filter_column": "grunnkrets_code",  # Rettet feil fra 'postal_code'
        "param_type": "STRING",
        "tables": {
            "houses": "`sibr-market.api.homes_houses_grunnkrets`",
            "apartments": "`sibr-market.api.homes_apartments_grunnkrets`",
        },
    },
}

# --- API Endepunkter ---
@api.get("/homes/{property_type}/{aggregation_level}", response_model=List[Dict[str, Any]], tags=["Aggregated Homes Data"])
def get_aggregated_data(
    property_type: Literal["houses", "apartments"],
    aggregation_level: Literal["county", "municipality", "postal_code", "grunnkrets"],
    filter_value: Optional[str] = None
):
    """
    Henter aggregert boligdata for en gitt boligtype og geografisk nivå.
    - **property_type**: Type bolig ('houses' eller 'apartments').
    - **aggregation_level**: Geografisk nivå ('county', 'municipality', 'postal_code', 'grunnkrets').
    - **filter_value**: En spesifikk verdi å filtrere på (f.eks. et fylkesnavn eller postnummer).
    """
    config = AGGREGATION_CONFIG.get(aggregation_level)
    if not config:
        raise HTTPException(status_code=400, detail="Invalid aggregation level specified.")

    table_name = config["tables"].get(property_type)
    if not table_name:
        raise HTTPException(status_code=400, detail="Invalid property type specified for this aggregation level.")

    query_params = []
    base_query = f"SELECT * FROM {table_name}"

    if filter_value:
        filter_column = config["filter_column"]
        param_type = config["param_type"]
        # Bruker LOWER for tekstsøk for å gjøre det case-insensitivt
        if param_type == "STRING":
            base_query += f" WHERE LOWER({filter_column}) = @filter_value"
            query_params.append(ScalarQueryParameter("filter_value", "STRING", filter_value.lower()))
        else:
            base_query += f" WHERE {filter_column} = @filter_value"
            query_params.append(ScalarQueryParameter("filter_value", param_type, filter_value))

    df = run_query(base_query, params=query_params)
    if df.empty:
        return []
    return df.to_dict('records')

@api.get("/homes/by-district-oslo", response_model=List[Dict[str, Any]], tags=["Aggregated Homes Data"])
def get_homes_by_district_oslo(name: Optional[str] = None):
    """
    Henter gjennomsnittlig boligdata gruppert på bydel i Oslo.
    - Bruker parameteriserte spørringer for sikkerhet.
    """
    query_params = []
    base_query = "SELECT * FROM `sibr-market.api.homes_oslo_districts`"

    if name:
        base_query += " WHERE LOWER(district_name) = @name"
        query_params.append(ScalarQueryParameter("name", "STRING", name.lower()))

    df = run_query(base_query, params=query_params)
    if df.empty:
        return []
    return df.to_dict('records')


@api.get('/homes', response_model=List[Dict[str, Any]], tags=["Raw Homes Data"])
def get_homes(municipality: Optional[str] = None, county: Optional[str] = None, postal_code: Optional[int] = None,
              property_type: Optional[str] = None, ownership_type: Optional[str] = None):
    base_query = '''SELECT * FROM agent.homes'''
    conditions = []
    query_params = []

    if municipality:
        # 1. Bruk LOWER() på kolonnen i SQL-en
        conditions.append("LOWER(municipality) = @municipality")
        # 2. Gjør om verdien til små bokstaver i Python
        query_params.append(ScalarQueryParameter("municipality", "STRING", municipality.lower()))

    if county:
        conditions.append("LOWER(county) = @county")
        query_params.append(ScalarQueryParameter("county", "STRING", county.lower()))

    if postal_code:
        # Trenger ikke LOWER() for tall
        conditions.append("postal_code = @postal_code")
        query_params.append(
            ScalarQueryParameter("postal_code", "STRING", postal_code))

    if property_type:
        conditions.append("LOWER(property_type) = @property_type")
        query_params.append(ScalarQueryParameter("property_type", "STRING", property_type.lower()))

    if ownership_type:
        conditions.append("LOWER(ownership_type) = @ownership_type")
        query_params.append(ScalarQueryParameter("ownership_type", "STRING", ownership_type.lower()))

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    df = run_query(base_query, params=query_params)
    if df.empty:
        return []

    df = df.replace({np.nan: None})

    return df.to_dict('records')

@api.get("/", include_in_schema=False)
def root():
    """Enkel velkomstmelding og link til API-dokumentasjonen."""
    return {"message": "Velkommen til Boligdata API. Gå til /docs for å se dokumentasjonen."}


if __name__ == "__main__":
    port = int(os.getenv("PORT",8080))
    uvicorn.run(api, host="0.0.0.0", port=port)