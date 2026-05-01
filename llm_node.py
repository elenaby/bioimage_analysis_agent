import json
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")


def choose_params(state):
    """
    Decide:
    - which steps to run
    - color mode
    - expansion size
    """

    user_msg = state.get("message", "")

    prompt = f"""
    You are an image processing assistant.

    User request:
    "{user_msg}"

    Decide:
    1. Steps to execute from this list:
       ["segment", "expand", "color"]

    2. Color mode:
       "random", "pink", or "blue"

    3. Expansion size:
       integer from 1 to 10

    Rules:
    - "only segment" → ["segment"]
    - "segment + expand" → ["segment", "expand"]
    - "just color" → ["segment", "color"]
    - "full pipeline" → ["segment", "expand", "color"]
    - Always include "segment" if color or expand is requested

    Return ONLY valid JSON:
    {{
        "steps": [...],
        "palette": "...",
        "expand": ...
    }}
    """

    response = llm.invoke(prompt)

    # 🧠 Robust parsing
    try:
        parsed = json.loads(response)
    except Exception:
        # fallback if LLM fails
        parsed = {
            "steps": ["segment", "expand", "color"],
            "palette": "random",
            "expand": 3
        }

    # ✅ Safe defaults + validation
    steps = parsed.get("steps", ["segment", "expand", "color"])

    if not isinstance(steps, list):
        steps = ["segment", "expand", "color"]

    # Ensure logical consistency
    if "expand" in steps or "color" in steps:
        if "segment" not in steps:
            steps.insert(0, "segment")

    palette = parsed.get("palette", "random")
    if palette not in ["random", "pink", "blue"]:
        palette = "random"

    try:
        expand_val = int(parsed.get("expand", 3))
    except Exception:
        expand_val = 3

    expand_val = max(1, min(expand_val, 10))

    # 🔁 Update state
    state["steps"] = steps
    state["palette"] = palette
    state["expand"] = expand_val

    return state