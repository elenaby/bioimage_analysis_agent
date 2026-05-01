from langgraph.graph import StateGraph
from typing import TypedDict, Any

from tools.segmentation import segment
from tools.morphology import expand
from tools.colorise import colorize
from llm_node import choose_params


# 1. Define the shared state structure
class GraphState(TypedDict):
    image: Any
    mask: Any
    palette: str
    expand: int
    result: Any
    steps: list        # 👈 NEW: which steps to execute
    message: str       # 👈 NEW: user input


# 2. Define nodes

def segmentation_node(state: GraphState) -> GraphState:
    """Step 1: always needed (produces mask)"""
    state["mask"] = segment(state["image"])
    return state


def llm_node(state: GraphState) -> GraphState:
    """Step 2: decide parameters + which steps to run"""
    return choose_params(state)


def morphology_node(state: GraphState) -> GraphState:
    """Step 3: expand blobs"""
    state["mask"] = expand(state["mask"], state["expand"])
    return state


def color_node(state: GraphState) -> GraphState:
    """Step 4: color instances"""
    state["result"] = colorize(state["mask"], state["palette"])
    return state


# 3. Routing logic (THIS is the key upgrade)

def route_after_llm(state: GraphState) -> str:
    steps = state.get("steps", [])

    if "expand" in steps:
        return "morphology"
    elif "color" in steps:
        return "color"
    else:
        return "__end__"


def route_after_morphology(state: GraphState) -> str:
    if "color" in state.get("steps", []):
        return "color"
    return "__end__"


# 4. Build graph

def build_graph():
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("segmentation", segmentation_node)
    graph.add_node("llm", llm_node)
    graph.add_node("morphology", morphology_node)
    graph.add_node("color", color_node)

    # Entry point
    graph.set_entry_point("segmentation")

    # Always start with segmentation → LLM
    graph.add_edge("segmentation", "llm")

    # 🔀 Conditional routing after LLM
    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "morphology": "morphology",
            "color": "color",
            "__end__": "__end__"
        }
    )

    # 🔀 Conditional routing after morphology
    graph.add_conditional_edges(
        "morphology",
        route_after_morphology,
        {
            "color": "color",
            "__end__": "__end__"
        }
    )

    return graph.compile()