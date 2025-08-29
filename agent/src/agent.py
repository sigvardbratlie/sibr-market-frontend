import json
import pandas as pd
from langgraph.graph import StateGraph,END
from typing import Dict,TypedDict,List,Union,Annotated,Sequence,Optional, Literal
import random
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,BaseMessage,ToolMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
from sibr_module import Logger
from google.cloud import bigquery
import os
from geopy.geocoders import Nominatim
from src.langchain_firestore_sb import FirestoreSaver
#load_dotenv()

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
def get_by_radius(lat : float,
                  lng : float,
                  property_type : str,
                  usable_area : int|float,
                  bedrooms : int,
                  radius : int = 1000,
                  factor_large_num : float = 0.3,
                  factor_small_num : int = 1,
                  top_n : int = 20):
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
def query_homes_database(
    select_statement: str,
    where_clause: str,
    group_by: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = 200
) -> str:
    """
    Executes a query specifically against the `sibr-market.agent.homes` table.
    Use this tool to answer analytical questions about the housing market.
    You only need to provide the clauses (SELECT, WHERE, etc.), not the full query.

    Args:
        select_statement (str): The columns to select. E.g., "AVG(price_pr_sqm), municipality".
        where_clause (str): The filter conditions. E.g., "LOWER(municipality) = 'oslo' AND bedrooms > 2".
        group_by (Optional[str]): The columns to group by. E.g., "municipality".
        order_by (Optional[str]): The columns to order by. E.g., "AVG(price_pr_sqm) DESC".
        limit (Optional[int]): The maximum number of rows to return. Defaults to 200.
    """
    base_query = f"SELECT {select_statement} FROM `sibr-market.agent.homes`"

    if where_clause:
        base_query += f" WHERE {where_clause}"
    if group_by:
        base_query += f" GROUP BY {group_by}"
    if order_by:
        base_query += f" ORDER BY {order_by}"
    if limit:
        base_query += f" LIMIT {limit}"

    try:
        client = bigquery.Client()
        print(f"\n--- EXECUTING CONSTRUCTED QUERY ---\n{base_query}\n-----------------------------------\n")
        df = client.query(base_query).to_dataframe()
        result_json = df.to_json(orient='records', date_format='iso')
        return result_json if not df.empty else "Query executed successfully, but returned no results."
    except Exception as e:
        return f"An error occurred: {e}"

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
              `sibr-market.admin.bsu_norway`
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
def get_postal_code(lat : float, lng : float):
    """A function to get the corresponding grunnkrets to a coordinate (lat,lng)"""

    query = f"""
            SELECT
              postnummer,
              poststed
            FROM
              `sibr-market.admin.postal_norway`
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
def get_municipality(lat : float, lng : float):
    """A function to get the corresponding grunnkrets to a coordinate (lat,lng)"""

    query = f"""
            SELECT
              kommunenummer
            FROM
              `sibr-market.admin.bsu_norway`
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

tools = [
        execute_bq_query,
         #query_homes_database,
         ask_user_for_info,
         tavily_search,
         get_by_radius,
         get_geoinfo,
         get_grunnkrets,
         get_postal_code,
         get_municipality,
         ]




class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

class HomeAgent:
    """
    An Agent designet to give property valuations and answer questions on the norwegian housing market.
    """
    def __init__(self,llms : dict[str, BaseChatModel], tools : List[tool],prompt : str,logger : Logger = None,):
        """
                Initializes the HomeAgent.
                Args:
                    llms (dict[str, BaseChatModel]): A dictionary mapping agent types to LLM models.
                    tools (List[tool]): A list of tools the agent can use.
                    logger (Logger, optional): The logger instance. Defaults to None.
                """
        if logger is None:
            logger = Logger("HomeAgent")
        self.logger = logger
        self.logger.set_level("DEBUG")
        self.tools = tools
        self.llms = llms
        self.prompt = prompt
        self.checkpointer = FirestoreSaver(project_id="sibr-market",database_id="homes-agent")
        # self.checkpointer = InMemorySaver()
        #self.agent = self._compile_agent()

    def _load_prompt(self,instructions_filepath : str, prompt) -> str:
        with open(instructions_filepath, "r") as f:
            instructions_text = f.read()
        return prompt + instructions_text


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

    def _call_llm(self,state: AgentState, llm_with_tools : BaseChatModel) -> AgentState:
        """Function to call the LLM with the current state.

        """
        message = llm_with_tools.invoke(state["messages"])
        return {'messages': [message]}

    def _compile_agent(self,agent_type : Literal["fast","expert"]):
        """
        Compiles the agent graph with the selected LLM.
        Args:
            agent_type (Literal["fast", "expert"]): The type of agent to compile.
        """
        print(f'AGent type input: {agent_type}')
        selected_llm = self.llms.get(agent_type).bind_tools(tools)
        if not selected_llm:
            raise ValueError(f'Invalid agent type: {agent_type}')

        llm = selected_llm.bind_tools(self.tools)

        graph = StateGraph(AgentState)
        graph.add_node("call_llm", lambda state: self._call_llm(state,llm_with_tools=llm))
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
    def __ia_stream_response(self, user_input: str, thread: dict, agent_type : Literal["fast", "expert"]):
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

    def stream_response(self, user_input: str, session_id: str, agent_type: Literal["fast", "expert"]):
        """
        This is a generator function that yields status updates and the final response.
        """
        agent_instance = self._compile_agent(agent_type)
        thread = {"configurable": {"thread_id": session_id}}

        try:
            current_state = agent_instance.get_state(thread)
            is_new_conv = not current_state.values.get("messages", [])
        except Exception:
            is_new_conv = True

        if is_new_conv:
            self.logger.info(f'Creating new conversation (thread: {session_id}. Choosing type of question...')
            prompt = self._load_prompt(instructions_filepath="src/instructions.txt", prompt=self.prompt)
            system_message = SystemMessage(content=prompt)
            agent_instance.update_state(thread, {"messages": [system_message]})
        else:
            self.logger.info(f'Continueing conversation (thread: {session_id})')

        for chunk in agent_instance.stream({"messages": [HumanMessage(content=user_input)]}, config=thread):
            if "call_llm" in chunk:
                ai_msg = chunk["call_llm"]["messages"][-1]
                if ai_msg.tool_calls:
                    tool_call = ai_msg.tool_calls[0]
                    tool_name = tool_call['name']
                    if tool_name in ['get_geoinfo','get_grunnkrets',"get_postal_code","get_municipality"]:
                        yield {"type": "status", "content": "🧠 Finding geographical information..."}
                    elif tool_name in ["get_by_radius",'execute_bq_query',"query_homes_database"]:
                        yield {"type": "status", "content": "🧠 Searching our databases..."}
                    elif tool_name == 'tavily_search':
                        yield {"type": "status", "content": "🧠 Searching online for more context..."}

                if ai_msg.content:
                    final_response_content = ai_msg.content
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    #session_id = "session123"
    agent = HomeAgent(llm = llm,tools=tools)
    agent.run()
