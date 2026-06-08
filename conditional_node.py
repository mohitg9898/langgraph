"""
Conditonal node needs 3:
source node
routing node - decision maker input is state and outout is string
making the edge , 
builder.add_conditional_edges(
    "my_source_node",   # 1. Where does the check happen?
    logic_router,       # 2. What function makes the decision?
    {                   # 3. Mapping: "string_returned": "actual_node_name"
        "go_fast": "premium_processing_node",
        "go_slow": "standard_processing_node"
    }
"""



import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Define the Tool
@tool
def process_refund(username: str, amount: float) -> str:
    """Process a financial refund for a specific user."""
    # Simulated error: The tool fails if the username is formatted poorly
    if "_" not in username:
        raise ValueError("Invalid username format. Must be 'first_last'.")
    return f"Successfully refunded ${amount} to {username}."

tools = [process_refund]
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# 2. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Define Nodes
def agent_node(state: State):
    """The LLM decides whether to call a tool."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def execute_tools_node(state: State):
    """Executes tool calls and handles errors safely."""
    last_message = state["messages"][-1]
    tool_outputs = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "process_refund":
            try:
                # Attempt to execute the tool
                result = process_refund.invoke(tool_args)
                tool_outputs.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            except Exception as e:
                # Catch error and pass it BACK to the LLM as a regular message
                error_msg = f"Tool Error: {str(e)}. Please correct your arguments and try again."
                tool_outputs.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
                
    return {"messages": tool_outputs}

# 4. Define Conditional Routing
def route_after_agent(state: State) -> Literal["human_approval", "execute_tools", "end"]:
    """Routes based on tool calls and security guardrails."""
    last_message = state["messages"][-1]
    
    if not getattr(last_message, "tool_calls", None):
        return "end"
    
    # Guardrail Check: Look at the arguments before running the tool
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "process_refund":
            if tool_call["args"].get("amount", 0) > 500:
                return "human_approval"  # Reroute to a pause/approval state
                
    return "execute_tools"

def human_approval_node(state: State):
    """Placeholder node that halts execution until a human triggers a resume."""
    print("\n[GUARDRAIL TRIGGERED] Refund exceeds $500. Waiting for human verification...")
    # In production, this state waits for API/UI interaction
    return {}

# 5. Assemble the Graph
builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("tools", execute_tools_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "human_approval": "human_approval",
        "execute_tools": "tools",
        "end": END
    }
)

# After tools run (or human approves), go back to the agent to evaluate the results
builder.add_edge("tools", "agent")
builder.add_edge("human_approval", "tools") 

# Memory checkpointer allows the graph to pause and save state
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human_approval"])
