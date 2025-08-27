#from langchain_google_firestore import FirestoreSaver
#from langgraph_checkpoint_firestore import FirestoreSaver
# from langgraph.checkpoint.memory import InMemorySaver, logger
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import FireCrawlLoader,PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
# from google.cloud import firestore
# import langgraph
#from pathlib import Path
import json
import pandas as pd
from langgraph.graph import StateGraph,END
from typing import Dict,TypedDict,List,Union,Annotated,Sequence,Optional
import random
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,BaseMessage,ToolMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv
from sibr_module import Logger
from google.cloud import bigquery
import os
from geopy.geocoders import Nominatim
from src.langchain_firestore_sb import FirestoreSaver
load_dotenv()

#logger = Logger(f'Sibr-Market-Agent')
#logger.set_level("INFO")
#langgraph.debug = True
#os.chdir("..")


# ==== SETUP TOOLS ====
@tool
def ask_user_for_info(question : str) -> str:
    """A function to ask the user for additional information."""
    return question

@tool
def execute_bq_query(query: str) -> str:
    """Executes a SQL query on Google BigQuery and returns the result as a JSON string."""
    try:
        client = bigquery.Client()
        #print(f"\n--- EXECUTING QUERY ---\n{query}\n-----------------------\n")
        df = client.query(query).to_dataframe()
        result_json = df.to_json(orient='records', date_format='iso')
        return result_json if not df.empty else "Query executed successfully, but returned no results."
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def get_by_radius(lat,lng,property_type,usable_area, bedrooms,radius = 1000,factor_large_num : float = 0.3,factor_small_num : int = 1,top_n = 20):
    """
    Finds comparable properties within a radius, using a set of specific filters to narrow down the search.
    It is crucial to use all available information from the user's query as filters.

    Args:
            lat (float): The latitude of the property
            lng (float): The longitude of the property
            property_type (str): The type of property, for example 'Leilighet' or 'Enebolig'
            usable_area (int): The usable area of the property
            bedrooms (int): The number of bedrooms in the property
            radius (int): The search radius in meters
            factor_large_num (float): A factor to control the search span of the different attributes with numbers larger than 10
            factor_small_num (int): A factor to control the search span of the different attributes with numbers smaller than 10
            top_n (int): The top number of samples to consider
    """
    usable_area_min = usable_area * (1-factor_large_num)
    usable_area_max = usable_area * (1 + factor_large_num)
    bedrooms_min  = max(0,bedrooms - factor_small_num)
    bedrooms_max = bedrooms + factor_small_num

    query = f"""
        SELECT
          (h.price_pr_sqm),
          h.price_pr_i_sqm,
          (h.url),
          h.bedrooms,
          h.floor,
          h.usable_area,
          h.internal_area,
          h.build_year,
          h.balcony,
          h.renovated,
          h.monthly_common_cost,
          h.eq_parking,
          h.eq_rental_unit,
          ST_DISTANCE(ST_GEOGPOINT(h.lng, h.lat), ST_GEOGPOINT({lng}, {lat})) AS distance_in_meters,
        FROM
          `sibr-market.agent.homes` h
          WHERE ST_DWITHIN(ST_GEOGPOINT(h.lng, h.lat), ST_GEOGPOINT({lng}, {lat}), {radius})
          AND LOWER(property_type) = LOWER('{property_type}')
          AND h.usable_area BETWEEN {usable_area_min} AND {usable_area_max}
          AND h.bedrooms BETWEEN {bedrooms_min} AND {bedrooms_max}
        ORDER BY distance_in_meters ASC
        LIMIT 200
    """

    try:
        client = bigquery.Client()
        #print(f"\n--- EXECUTING QUERY --- \n{query}\n-----------------------------------\n")
        df = client.query(query).to_dataframe()
        print(f'got {len(df)} rows')
    except Exception as e:
        return f"An error occurred: {e}"

    # df_filtered = df.copy()
    # if len(df)>20:
    #     for key, value in users_property.items():
    #         if key != "property_type" and key not in df_filtered.columns:
    #             print(f"Extra filter '{key}' not found in the dataframe.")
    #             continue
    #
    #
    #         if isinstance(value, float) or isinstance(value, int):
    #             if key == "build_year" or  key == "balcony":
    #                 min_val = max(0,value-factor_small_num*3)
    #                 max_val = value+factor_small_num*3
    #                 df_filtered = df_filtered[(df_filtered[key]>min_val) & (df_filtered[key]<max_val)]
    #                 print(f'Minimum value {min_val} and maximum value {max_val} for property {key}\nLen after filter {len(df_filtered)}')
    #
    #             elif value<=10:
    #                 min_val = int(max(0,value-factor_small_num))
    #                 max_val = int(value+factor_small_num)
    #                 print(f'Minimum value {min_val} and maximum value {max_val} for property {key}\nLen after filter {len(df_filtered)}')
    #
    #             elif value>10:
    #                 min_val = int(max(0,value * (1-factor_large_num)))
    #                 max_val = int(value * (1+factor_large_num))
    #                 df_filtered = df_filtered[(df_filtered[key] >= min_val) & (df_filtered[key] <= max_val)]
    #                 print(f'Minimum value {min_val} and maximum value {max_val} for property {key}\nLen after filter {len(df_filtered)}')
    #         elif isinstance(value, str):
    #             df_filtered = df_filtered[df_filtered[key].str.lower() == value.lower()]
    #             print(f'Value {value} for property {key}\nLen after filter {len(df_filtered)}')
    #         else:
    #             df_filtered = df_filtered[df_filtered[key] == value]
    #             print(f'Value {value} for property {key}\nLen after filter {len(df_filtered)}')

    df_sorted = df.sort_values(by='distance_in_meters').head(top_n)
    if df_sorted.empty:
        return f"No properties remained after applying extra filters. Found {len(df)} initially."

    # avg_sqm_price = df_sorted['price_pr_sqm'].mean()
    # num_properties_used = len(df_sorted)
    # urls = df_sorted['url'].tolist()
    #
    # result = {
    #     "average_sqm_price": round(avg_sqm_price),
    #     "number_of_comparables_used": num_properties_used,
    #     "urls": urls
    # }

    return json.dumps(df_sorted.to_json(orient='records', date_format='iso'))

#
# @tool
# def process_properties(properties_json: str,users_property: dict, top_n: int = 15, factor_large_num : float = 0.3,factor_small_num : int = 1):
#     """
#     Processes a list of properties (from get_by_radius) to calculate a final valuation.
#     It sorts by distance, takes the top N closest, applies the users property specifications, and calculates the average price.
#     It uses a factor to apply ranges of numerical values to the filter.
#
#     Args:
#         properties_json (str): The JSON file from get_by_radius.
#         users_property (dict): The dictionary with all the users property specifications.
#         top_n (int): The top number of samples to consider
#         factor_large_num (float): A factor to control the search span of the different attributes with numbers larger than 10.
#         factor_small_num (int): A factor to control the search span of the different attributes with numbers smaller than 10.
#     """
#
#     try:
#         properties = json.loads(properties_json)
#         if not properties:
#             return "No properties to process."
#
#         df = pd.DataFrame(properties)
#
#         if not users_property:
#             raise TypeError("Dictionary is empty")
#
#         for key, value in users_property.items():
#             if key != "property_type" and key not in df.columns:
#                 print(f"Extra filter '{key}' not found in the dataframe.")
#                 continue
#
#
#             if isinstance(value, float) or isinstance(value, int):
#                 if key == "build_year":
#                     df = df[(df[key]>value-factor_small_num*3) & (df[key]<value+factor_small_num*3)]
#
#                 elif value<=10:
#                     min_val = int(max(0,value-factor_small_num))
#                     max_val = int(value+factor_small_num)
#
#                 elif value>10:
#                     min_val = int(max(0,value * (1-factor_large_num)))
#                     max_val = int(value * (1+factor_large_num))
#
#                 df = df[(df[key] >= min_val) & (df[key] <= max_val)]
#             elif isinstance(value, str):
#                 df = df[df[key].str.lower() == value.lower()]
#             else:
#                 df = df[df[key] == value]
#
#         df_sorted = df.sort_values(by='distance_in_meters').head(top_n)
#
#         if df_sorted.empty:
#             return f"No properties remained after applying extra filters. Found {len(df)} initially."
#
#         # 3. Kalkuler og returner resultatet
#         avg_sqm_price = df_sorted['price_pr_sqm'].mean()
#         num_properties_used = len(df_sorted)
#         urls = df_sorted['url'].tolist()
#
#         result = {
#             "average_sqm_price": round(avg_sqm_price),
#             "number_of_comparables_used": num_properties_used,
#             "urls": urls
#         }
#         return json.dumps(result)
#
#     except (json.JSONDecodeError, TypeError) as e:
#         return f"Error: Input was not a valid JSON string of properties. Details: {e}"
#     except Exception as e:
#         return f"An error occurred during processing: {e}"

@tool
def analyze_properties_data(properties_json: str, question: str) -> str:
    """
    Analyzes a JSON list of properties to answer a specific question.
    Use this for questions about sales time, amenities, sizes, etc.
    Do NOT use this for the primary valuation task; use 'process_properties' for that.

    Example Question: 'What is the average sales time for these properties?'
    Example Question: 'How many of the properties built after 2010 have a balcony?'
    """
    try:
        properties = json.loads(properties_json)
        if not properties:
            return "No properties to analyze."

        df = pd.DataFrame(properties)
        return df.describe().to_json()  # Returnerer en enkel statistisk oppsummering

    except Exception as e:
        return f"An error occurred during analysis: {e}"

@tool
def get_geoinfo(address : str) -> dict:
    """A function to find basic geographical information about an address"""
    geo = Nominatim(user_agent="predict_homes")
    try:
        coor = geo.geocode(address)
        if coor:
            data = {"lat" : coor.latitude,
                    "lng" : coor.longitude,}
            data["display_name"] = coor.raw.get("display_name","")

            postal_code = (coor.raw.get("display_name")).split(",")[-2].strip()
            if len(postal_code) == 4 and isinstance(int(postal_code), int):
                data["postal_code"] = postal_code
            else:
                for i in coor.raw.get("display_name").split(","):
                    if len(i.strip()) == 4 and isinstance(int(i.strip()), int):
                        data["postal_code"] = i.strip()
            return data
        else:
            print('No data from address')
            raise Exception("No data from address")
    except Exception as e:
        print(e)

@tool
def get_grunnkrets(lat : float, lng : float):
    """A function to get the corresponding grunnkrets to a coordinate (lat,lng)"""

    query = f"""
            SELECT
              grunnkretsnavn,
              grunnkretsnummer
            FROM
              `sibr-market.admin.geo_grunnkretser_norge`
            WHERE
              ST_CONTAINS(geometry, ST_GEOGPOINT({lng},{lat}))
            """
    try:
        client = bigquery.Client()
        #print(f"\n--- EXECUTING QUERY ---\n{query}\n-----------------------\n")
        df = client.query(query).to_dataframe()
        result_json = df.to_json(orient='records', date_format='iso')
        return result_json if not df.empty else "Query executed successfully, but returned no results."
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def get_municipality(postal_code : str = None, postal_place : str = None):
    """Function to get the correct municipality"""
    if postal_code is None and postal_place is None:
        raise ValueError(f'Either postal_code or postal_place must be provided')
    if postal_code:
        query  = f"""
                SELECT municipality FROM admin.geo_norge
                WHERE postal_code = {postal_code}
                """
    elif postal_place:
        query = f"""
                SELECT municipality FROM admin.geo_norge
                WHERE LOWER(postal_place) LIKE LOWER('%{postal_place}%')
                """
    try:
        client = bigquery.Client()
        print(f"\n--- EXECUTING QUERY ---\n{query}\n-----------------------\n")
        df = client.query(query).to_dataframe()
        result_json = df.to_json(orient='records', date_format='iso')
        return result_json if not df.empty else "Query executed successfully, but returned no results."
    except Exception as e:
        return f"An error occurred: {e}"

tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

# @tool
# def retriever_tool(query: str) -> str:
#     '''
#     This tool searches and return information from the connected source documents
#     '''
#
#     docs = retriever.invoke(query)
#     if not docs:
#         return "No documents found"
#
#     results = []
#     for i, doc in enumerate(docs):
#         results.append(f"Document {i + 1}:\n{doc.page_content}")
#     return "\n".join(results)

tools = [execute_bq_query,
         ask_user_for_info,
         tavily_search,
         get_by_radius,
         #process_properties,
         get_geoinfo,
         get_grunnkrets,
         get_municipality,
         ]
llm = ChatOpenAI(model = "gpt-4o",temperature=0.2)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

class HomeAgent:
    """
    An Agent designet to give property valuations and answer questions on the norwegian housing market.
    """
    def __init__(self,llm, tools : List[tool],logger : Logger = None):
        if logger is None:
            logger = Logger("HomeAgent")
        self.logger = logger
        self.logger.set_level("DEBUG")
        self.tools = tools
        self.llm = llm.bind_tools(tools)
        self.prompt = self._load_prompt("src/instructions.txt")
        self.checkpointer = FirestoreSaver(project_id="sibr-market",database_id="homes-agent")
        # self.checkpointer = InMemorySaver()
        self.agent = self._compile_agent()

    def _load_prompt(self,instructions_filepath : str) -> str:
        with open(instructions_filepath, "r") as f:
            instructions_text = f.read()

        BASE_SYSTEM_PROMPT_222 = """
You are a helpful and expert data analyst assistant for real estate data in Norway.
Your goal is twofold:
1.  Be an AI valuator, where you help the user provide an accurate and good valuation of their property.
2.  Respond to the user's questions as a general expert on housing.

To achieve this, you must follow a strict execution strategy.
**TIPS:** Remember always to method `LOWER()` in all queries involving string columns! As some datasets has 'Oslo', others have 'OSLO' and others have 'oslo'
For valuation tasks, use the VALUATION STRATEGY, otherwise use the GENERAL STRATEGY.

        ---
## VALUATION STRATEGY
    **Step 1: Always Gather Geographical Context**
    * Regardless of the user's query, **always start** by using the `get_geoinfo` tool on the address they provide. This will give you the crucial `lat` and `lng` coordinates for the property. If the user hasn't provided an address, you must ask for it.
    
    **Step 2: Broad Search for Potential Comparables**
    * Your first action is to call the `get_by_radius` tool.
    * Parse ALL property details from the user's message (`property_type`, `bedrooms`, `usable_area`, etc.).
    * Construct the `filters` dictionary for the tool call. Use a WIDE search radius of 2000 meters and apply only the most essential filters (i.e `property_type`). The goal here is to get a good list of potential candidates. Do not include filters like build_year or any kind of facilities.

    **Step 3: Process and Refine for Final Valuation**
    * **CRITICAL:** Take the JSON output from `get_by_radius` and immediately pass it as the `properties_json` argument to the `process_properties` tool.
    * Use the `extra_filters` argument in `process_properties` to apply stricter, user-specific requirements (e.g., `{'eq_lift': True, 'eq_parking': True}`). This is how you refine the search.
    * The `process_properties` tool will give you the final, calculated `average_sqm_price` and the `number_of_comparables_used`.

    **Step 4: Formulate the Response**
    * Use the final, processed result from `process_properties`.
    * Calculate the final valuation: `final_value = result['average_sqm_price'] * user's_area`.
    * Before presenting your result, double check the price against the benchmark `refprice_sqm_grunnkrets` x the users usable_area (or `refprice_sqm_postal` if grunnkrets does not exist), to see if there is a large difference. If so, comment on it.
    * PRESENT FINAL ANSWER (STRICT RULES)
        * Always make sure to multiply the reference price with the users own area, so that the user gets a final price presented for them, not a pr sqm price.
        * Always make sure to present all urls used a sources together with the final answer, so that the user can inspect and validate the final result.
        * Mention the benchmark price `refprice_sqm_grunnkrets` if the distance is mentionable. 

        ---
## GENERAL QUERY STRATEGY

    This strategy is for answering general questions about the housing market (e.g., "What is the average price for houses in Hallingdal?").

    **Step 1: Attempt a Direct Database Query**
        * Use your best judgment to formulate a query with `execute_bq_query` based on the user's question. For example, `SELECT AVG(price) FROM agent.homes WHERE LOWER(municipality) = LOWER('hallingdal')`.

    **Step 2: Verify and Investigate if the Query Fails**
        * If the query from Step 1 returns no results, **do not immediately tell the user you can't find data.**
        * Your next action is to use the `tavily_search` tool to investigate the location. For example, run a search like "municipalities in Hallingdal, Norway".
        * The goal of this search is to determine if the user's term is a region, an old name, or misspelled, and to find the correct, official municipality or postal code names associated with it.
        * You can also use the `analyze_properties_data` tool in order to analyse the data.

    **Step 3: Execute a Corrected Query**
        * Using the list of correct locations from your web search, formulate a new, more accurate query. You will likely need to use an `IN` clause.
        * Example: `SELECT AVG(price) FROM agent.homes WHERE municipality IN ('Ål', 'Flå', 'Gol', 'Hemsedal', 'Hol', 'Nesbyen')`.
        * Finally, present this corrected data to the user, and briefly explain what you did (e.g., "Hallingdal is a region that includes several municipalities. The average price across these municipalities is...").

---

## EXAMPLES

    ### VALUATION EXAMPLE

    -- **Primary Method (Radius Search Tool Call):**
    -- User asks for a valuation of their 97sqm, 4-bedroom apartment at Teglverksfaret 14, 1405 Langhus in 3rd floor.
    -- First, get_geoinfo('Teglverksfaret 14, 1405 Langhus') -> returns lat: 59.77, lng: 10.82
    -- Then, call the wide-search tool:
    TOOL CALL: get_by_radius(filters={"lat": 59.77, "lng": 10.82, "radius": 500, "property_type": "Leilighet"})

    -- The agent now receives the raw JSON output from the tool. It should immediately take this output and process it using the next tool.
    -- Then, call the processing tool with the previous output and all the specific user details:
    TOOL CALL: process_properties(properties_json="<result_from_get_by_radius>", extra_filters={"usable_area_min": 80, "usable_area_max": 120, "bedrooms_min": 3, "bedrooms_max": 5, "floor" = 3})

        -- **Fallback Method (Pre-calculated average):**
        -- "Radius search found no direct comparables for a specific property. Get the pre-calculated grunnkrets average."
        SELECT refprice_sqm_grunnkrets, n_grunnkrets
        FROM `sibr-market.agent.homes`
        WHERE LOWER(grunnkretsnavn) = LOWER('Langhus senter')
        LIMIT 1;

    ## GENERAL EXAMPLE
        -- **General Query (Corrected after web search):**
        SELECT ROUND(AVG(price_pr_sqm)) as average_sqm_price, COUNT(*) as number_of_properties
        FROM `sibr-market.agent.homes`
        WHERE municipality IN ('Ål', 'Flå', 'Gol', 'Hemsedal', 'Hol', 'Nesbyen');

---------------------
DATABASE SCHEMA:
{schema_information}
---------------------
"""

        BASE_SYSTEM_PROMPT = """
You are a helpful and expert data analyst assistant for real estate data in Norway.

Your goal is twofold:
1. Be an AI valuator: Provide accurate property valuations.
2. Answer general housing market questions.

Follow a strict strategy: For valuations, use VALUATION STRATEGY; otherwise, GENERAL STRATEGY.

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

---
GENERAL STRATEGY
    For non-valuation questions (e.g., "Average price in Hallingdal?"):
    
    Step 1: Direct Query
    - Formulate `execute_bq_query` (e.g., SELECT AVG(price) FROM agent.homes WHERE LOWER(municipality) = LOWER('hallingdal')).
    
    Step 2: Verify if Fails
    - If no results, use `tavily_search` to check/correct location (e.g., "municipalities in Hallingdal, Norway").
    - Use `analyze_properties_data` if needed.
    
    Step 3: Corrected Query
    - Retry with accurate terms (e.g., IN clause for multiple municipalities).

---
EXAMPLES
    
    VALUATION EXAMPLE
    User: "Verdi av 97sqm, 4-bedroom apartment at Teglverksfaret 14, 1405 Langhus, 3rd floor."
    
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
    
    GENERAL EXAMPLE
    SELECT ROUND(AVG(price_pr_sqm)) as average_sqm_price FROM agent.homes WHERE municipality IN ('Ål', 'Flå', 'Gol', 'Hemsedal', 'Hol', 'Nesbyen');

---------------------
DATABASE SCHEMA:
{schema_information}
---------------------
"""

        return BASE_SYSTEM_PROMPT + instructions_text


    def _should_continue(self,state: AgentState) -> bool:
        """Determine if we should continue or end the conversation"""
        result = state["messages"][-1]
        return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    def _call_tool(self,state: AgentState) -> AgentState:
        """Executes tool calls from the LLM's response"""

        tools_dict = {our_tool.name: our_tool for our_tool in self.tools}

        if not isinstance(state["messages"][-1], AIMessage):
            raise TypeError(f'The last message is not an AI message and has not attr "tool_calls"')

        tool_calls = state["messages"][-1].tool_calls
        results = []

        if not tool_calls:
            self.logger.info(f'No tool calls found')
            # Return an empty list to avoid an error, as there's nothing to append
            return {"messages": []}

        for tool in tool_calls:
            name = tool.get("name", "")
            args = tool.get("args", "")
            self.logger.info(f'Calling Tool: {name} with query: {args}')

            if name in tools_dict:
                tool_to_call = tools_dict[name]
                if isinstance(args, dict) and "query" in args.keys():
                    input_to_tool = args.get("query")
                else:
                    input_to_tool = args
                try:
                    result = tool_to_call.invoke(input_to_tool)
                except Exception as e:
                    result = f'Something went wrong when calling tool {name} with {input_to_tool}: {e}.'
                    self.logger.info(result)

                self.logger.info(f'Result length: {len(str(result))}')

                # CORRECT: Use the proper keyword 'tool_call_id'
                results.append(ToolMessage(tool_call_id=tool["id"], name=tool["name"], content=str(result)))
            else:
                self.logger.info(f'{tool["name"]} does not exists in tools. \nTools available: {tools_dict.keys()}')
                result = "Incorrect Tool Name, Please Retry and Select tool from list of avaible tools"
                # CORRECT: Use the proper keyword 'tool_call_id'
                results.append(ToolMessage(tool_call_id=tool["id"], name=tool["name"], content=str(result)))

        self.logger.info(f'Tools execution complete')
        return {"messages": results}

    def _call_llm(self,state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        message = self.llm.invoke(state["messages"])
        return {'messages': [message]}

    def _compile_agent(self):
        graph = StateGraph(AgentState)
        graph.add_node("call_llm", self._call_llm)
        graph.add_node("call_tool", self._call_tool)
        graph.set_entry_point("call_llm")
        graph.add_edge("call_tool", "call_llm")
        graph.add_conditional_edges("call_llm",
                                    self._should_continue,
                                    {
                                        True: "call_tool",
                                        False: END
                                    })
        agent = graph.compile(checkpointer=self.checkpointer)
        return agent

    def _process_with_stream(self,user_input : str,thread : dict):
        print("The Agent is thinking...")
        final_response_content = ""

        for chunk in self.agent.stream({"messages": [HumanMessage(content=user_input)]}, config=thread):
            # Sjekk om det er en AI-melding med et verktøykall
            if "call_llm" in chunk:
                ai_msg = chunk["call_llm"]["messages"][-1]
                if ai_msg.tool_calls:
                    tool_call = ai_msg.tool_calls[0]  # Se på det første verktøykall
                    tool_name = tool_call['name']

                    # Oversett verktøynavnet til en "tanke"
                    if tool_name == 'get_geoinfo':
                        print("🧠 Finding geographical information about the address...")
                    elif tool_name == 'get_grunnkrets':
                        print("🧠 Finding the sub-district name (grunnkrets)...")
                    elif tool_name == 'execute_bq_query':
                        print("🧠 Searching in our connected databases...")
                    elif tool_name == 'tavily_search':
                        print("🧠 Using search engine to search online...")

            # Sjekk om det er en ferdig AI-melding med innhold
            if "call_llm" in chunk:
                ai_msg = chunk["call_llm"]["messages"][-1]
                if ai_msg.content:
                    final_response_content = ai_msg.content

        # Når loopen er ferdig, print det endelige svaret
        print("\n--- ANSWER ---")
        print(f"AI: \t {final_response_content}")

    def _process_without_stream(self,user_input : str,thread : dict):
        response = self.agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=thread)

        ai_response = response["messages"][-1].content if hasattr(response["messages"][-1],
                                                                  "content") else "NB: response object has no attr 'content'"
        print(f'\nAI: \t {ai_response}')

    # This is a new, UI-compatible method in your HomeAgent class
    def stream_response(self, user_input: str, thread: dict):
        """
        This is a generator function that yields status updates and the final response.
        """
        final_response_content = ""

        # The agent.stream() itself is a generator, so we loop through it.
        for chunk in self.agent.stream({"messages": [HumanMessage(content=user_input)]}, config=thread):
            if "call_llm" in chunk:
                ai_msg = chunk["call_llm"]["messages"][-1]
                if ai_msg.tool_calls:
                    tool_call = ai_msg.tool_calls[0]
                    tool_name = tool_call['name']

                    # Instead of printing, YIELD a structured message (dictionary is best)
                    if tool_name == 'get_geoinfo':
                        #self.logger.debug("🧠 Finding geographical information about the address...")
                        yield {"type": "status", "content": "🧠 Finding geographical information..."}
                    elif tool_name == 'get_grunnkrets':
                        #self.logger.debug("🧠 Finding the sub-district name (grunnkrets)...")
                        yield {"type": "status", "content": "🧠 Identifying the specific sub-district..."}
                    elif tool_name == 'execute_bq_query':
                        #self.logger.debug("🧠 Searching in our connected databases...")
                        yield {"type": "status", "content": "🧠 Searching the database..."}
                    elif tool_name == 'tavily_search':
                        #self.logger.debug("🧠 Using search engine to search online...")
                        yield {"type": "status", "content": "🧠 Searching online for more context..."}

                # Check for the final answer
                if ai_msg.content:
                    final_response_content = ai_msg.content

        # At the very end, yield the final answer with a different type
        if final_response_content:
            #self.logger.info(f'\nAI: \t {final_response_content}')
            yield {"type": "final_answer", "content": final_response_content}

    def run(self):

        session_id = random.randint(1000, 9999)
        thread = {"configurable": {"thread_id": session_id}}
        try:
            existing_state = self.agent.get_state(thread)
            is_new_conversation = not existing_state.values.get("messages", [])
        except:
            is_new_conversation = True

        if is_new_conversation:
            self.logger.info(f'Creating a new conversation with thread {thread["configurable"]["thread_id"]}')
            self.agent.update_state(thread, {"messages": SystemMessage(content=self.prompt)})
        else:
            self.logger.info(f'Continueing conversation with thread {thread["configurable"]["thread_id"]}')

        while True:
            user_input = input("\nUser: \t")
            if user_input.lower() in ["exit", "quit", "avslutt"]:
                break
            self._process_with_stream(user_input, thread)
            #self._process_without_stream(user_input,thread)


if __name__ == "__main__":
    #session_id = "session123"
    agent = HomeAgent(llm = llm,tools=tools)
    agent.run()
