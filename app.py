from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from typing import Annotated, TypedDict, List
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
# from sql_agent import sql_agent_tool
from rag_agent import rag_agent_tool
from profile_agent import profile_agent_tool
from langgraph.graph.message import add_messages
import chainlit as cl

load_dotenv()

def load_app():

    tivly_tool = TavilySearchResults(max_results=5)
    
    tool_belt = [
        # sql_agent_tool,
        rag_agent_tool,
        profile_agent_tool,
        tivly_tool
    ]

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model = model.bind_tools(tool_belt)

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        context: List[Document]

    tool_node = ToolNode(tool_belt)

    uncompiled_graph = StateGraph(AgentState)

    def call_model(state):
        messages = state["messages"]
        response = model.invoke(messages)
        return {
                "messages": [response],
                "context": state.get("context", [])
            }

    uncompiled_graph.add_node("agent", call_model)
    uncompiled_graph.add_node("action", tool_node)

    uncompiled_graph.set_entry_point("agent")

    def should_continue(state):
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "action"

        return END

    uncompiled_graph.add_conditional_edges(
        "agent",
        should_continue
    )

    uncompiled_graph.add_edge("action", "agent")

    compiled_graph = uncompiled_graph.compile()

    return compiled_graph 

app = load_app()

@cl.on_chat_start
async def start():
  cl.user_session.set("app", app)

@cl.on_message
async def handle(message: cl.Message):
  app = cl.user_session.get("app")
  state = {"messages" : [HumanMessage(content=message.content)]}
  response = await app.ainvoke(state)
  await cl.Message(content=response["messages"][-1].content).send()

# if __name__ == "__main__":
    
#     if os.getenv("OPENAI_API_KEY") is None:
#         print("OPENAI_API_KEY is not set")
#         exit()
    
#     print("Banner Flip Engine - Type 'exit' to quit")
#     while True:
#         query = input("\nAsk about store data: ")
#         if query.lower() == 'exit':
#             break
            
#         print(f"\nProcessing query: {query}")
#         try:
#             final_response = None
#             for event in app.stream({"messages": [HumanMessage(content=query)]}, stream_mode="updates"):
#                 for key, value in event.items():
#                     print(f"Receiving message from agent: '{key}'")
#                     print(value["messages"])
#                     if value["messages"]:
#                         final_response = value["messages"][-1].content
            
#             if final_response:
#                 print(f"\nFinal Answer: {final_response}")
#             else:
#                 print("No response received")
                
#         except Exception as e:
#             print(f"Error processing query: {str(e)}")
