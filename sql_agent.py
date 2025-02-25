from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
import sqlalchemy as sa
import pandas as pd
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
from dotenv import load_dotenv
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from math import radians, sin, cos, sqrt, atan2

load_dotenv()

def calculate_distance(lat1, lon1, lat2, lon2):
    # Handle None values from database or geocoder
    if None in (lat1, lon1, lat2, lon2):
        return None
    
    try:
        # Haversine formula to calculate distance in miles
        R = 3958.8  # Earth radius in miles
        
        # Convert all coordinates to floats
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
        
        # Validate coordinate ranges
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90 and
                -180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            return None

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    except (TypeError, ValueError) as e:
        print(f"Distance calculation error: {str(e)}")
        return None

class Geocoder:
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="banner-flip-app/1.0",  # Required by Nominatim terms
            timeout=10  # Add timeout for reliability
        )
    
    def geocode(self, location):
        try:
            result = self.geolocator.geocode(location)
            if result:
                return (result.latitude, result.longitude)
            # Fallback for US cities/states
            if ',' in location:
                city, state = location.split(',', 1)
                result = self.geolocator.geocode(f"{city.strip()}, {state.strip()}, USA")
                return (result.latitude, result.longitude) if result else (None, None)
            return (None, None)
        except Exception as e:
            print(f"Geocoding error: {str(e)}")
            return (None, None)

def load_agent():
    sql_llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini"
    )
    
    engine = sa.create_engine(
        "sqlite:///competitor_stores.db",
        poolclass=sa.pool.SingletonThreadPool
    )

    # Add this distance function to SQL
    @sa.event.listens_for(engine, "connect")
    def create_sqlite_functions(dbapi_connection, connection_record):
        # Add deterministic flag for better query optimization
        dbapi_connection.create_function("distance", 4, calculate_distance, deterministic=True)

    json_file = 'data/competitor_store_19k.json'
    table_name = 'competitor_stores'
    
    df = pd.read_json(json_file)
    
    with engine.connect() as conn:
        if not conn.dialect.has_table(conn, table_name):
            df.to_sql(table_name, conn, index=False, if_exists='replace')
            conn.commit()

    db = SQLDatabase(engine)
    toolkit = SQLDatabaseToolkit(
        db=db, 
        llm=sql_llm
    )

    sql_agent = create_sql_agent(
        llm=sql_llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=SQL_PREFIX,
        suffix=SQL_SUFFIX,
        handle_parsing_errors=True
    )

    tools = [
        Tool(
            name="Geocoder",
            func=Geocoder().geocode,
            description="Convert text addresses to GPS coordinates. Input: 'City, State' or full address. Output: (latitude, longitude)"
        ),
        Tool(
            name="SQLAgent",
            func=lambda q: sql_agent.invoke({"input": q}).get("output", "No results found"),
            description=f"""Useful for answering questions about structured data in SQL databases. 
            Database contains a table called {table_name} with columns: {', '.join(df.columns)}.
            Use the distance(lat1, lon1, lat2, lon2) function to calculate miles between coordinates."""
        ),
    ]

    class State(TypedDict):
        messages: Annotated[list[Any], add_messages]

    agent_llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"
    )
    agent = create_react_agent(agent_llm, tools)
    agent_executor = StateGraph(State)
    agent_executor.add_node("agent", agent)
    agent_executor.set_entry_point("agent")
    agent_executor.add_edge("agent", END)
    return agent_executor.compile()

sql_agent = load_agent()

@tool
def sql_agent_tool(question: str) -> str:
    """Useful for answering questions about structured data in SQL databases. Database contains data about competitor stores.
    Includes distance(lat1, lon1, lat2, lon2) function to calculate miles between coordinates.
    Do not generate profiles.
    """
    response = sql_agent.invoke({"messages": [HumanMessage(content=question)]})
    return response

if __name__ == "__main__":

    print("Banner Flip Engine - Type 'exit' to quit")
    while True:
        query = input("\nAsk about store data: ")
        if query.lower() == 'exit':
            break
            
        print(f"\nProcessing query: {query}")
        try:
            final_response = None
            for event in sql_agent.stream({"messages": [HumanMessage(content=query)]}, stream_mode="updates"):
                for key, value in event.items():
                    print(f"Receiving message from agent: '{key}'")
                    print(value["messages"])
                    if value["messages"]:
                        final_response = value["messages"][-1].content
            
            if final_response:
                print(f"\nFinal Answer: {final_response}")
            else:
                print("No response received")
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
