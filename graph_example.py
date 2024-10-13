import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, END, StateGraph
# Define the state
from langgraph.prebuilt import tools_condition
from llm_model import model



llm = model


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    count: Annotated[int, "Number of questions asked"]


# Define the agent node
def agent(state: State):
    messages = state['messages']
    count = state['count']

    last_human_message = messages[-1].content
    response = llm.invoke([SystemMessage("You are a helpful AI assistant. Answer the user's question briefly."), HumanMessage(last_human_message)])

    # Update the stat
    new_state = {
        "messages": [*messages, AIMessage(content=response.content)],
        "count": count + 1
    }

    return new_state


# Define the human node
def human(state: State):

    human_input = input("ask the model a question:")

    new_state = {
        "messages": [*state['messages'], HumanMessage(human_input)],
        "count": state['count']
    }

    return new_state


def should_continue(state: State):
    if state['count'] >= 3:
        return "end"
    else:
        return "human"


workflow = StateGraph(State)

workflow.add_node("agent", agent)
workflow.add_node("human", human)

workflow.add_edge("human", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "human": "human",
        "end": END
    }
)

workflow.set_entry_point("human")

graph = workflow.compile()

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png(output_file_path=os.path.join(os.getcwd(), "graph-png.png"))))

output = graph.invoke({"messages": [], "count": 0})
print(output["messages"])
