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
    steps: list
    message: str


# 2. Nodes

def segmentation_node(state: GraphState) -> GraphState:
    """Always run first"""
    state["mask"] = segment(state["image"])
    return state


def llm_node(state: GraphState) -> GraphState:
    """Decide steps + parameters"""
    return choose_params(state)


def morphology_node(state: GraphState) -> GraphState:
    """Optional expansion"""
    state["mask"] = expand(state["mask"], state["expand"])
    return state


def color_node(state: GraphState) -> GraphState:
    """Optional coloring"""
    state["result"] = colorize(state["mask"], state["palette"])
    return state


# 3. Routing logic (FIXED + clearer)

def route_after_llm(state: GraphState) -> str:
    steps = state.get("steps", [])

    print("Routing after LLM:", steps)

    if "expand" in steps:
        return "morphology"

    if "color" in steps:
        return "color"

    return "__end__"


def route_after_morphology(state: GraphState) -> str:
    steps = state.get("steps", [])

    print("Routing after morphology:", steps)

    if "color" in steps:
        return "color"

    return "__end__"


# 4. Build graph

def build_graph():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("segmentation", segmentation_node)
    graph.add_node("llm", llm_node)
    graph.add_node("morphology", morphology_node)
    graph.add_node("color", color_node)

    # Entry point
    graph.set_entry_point("segmentation")

    # Linear start
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