import json
import re
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")


def choose_params(state):
    """
    Decide:
    - which steps to run (STRICT RULE-BASED → deterministic)
    - color mode + expansion size (LLM → optional, flexible)
    """

    user_msg = state.get("message", "").lower().strip()

    # -------------------------------------------------
    # 1️⃣ STRONG RULE-BASED STEP SELECTION (FIXED)
    # -------------------------------------------------

    # Normalize text (remove punctuation, simplify spacing)
    msg = re.sub(r"[^a-z\s]", " ", user_msg)
    msg = re.sub(r"\s+", " ", msg).strip()

    # Keywords
    has_segment = "segment" in msg
    has_expand = any(w in msg for w in ["expand", "dilate", "grow"])
    has_color = any(w in msg for w in ["color", "colour", "paint"])

    has_only = "only" in msg or "just" in msg

    # --- RULES (ordered, deterministic) ---

    # 1. Explicit "segment only"
    if has_segment and has_only and not has_expand and not has_color:
        steps = ["segment"]

    # 2. Explicit "color only"
    elif has_color and has_only:
        steps = ["segment", "color"]

    # 3. Explicit "expand only"
    elif has_expand and has_only:
        steps = ["segment", "expand"]

    # 4. Expand requested (no color)
    elif has_expand and not has_color:
        steps = ["segment", "expand"]

    # 5. Color requested (no expand)
    elif has_color and not has_expand:
        steps = ["segment", "color"]

    # 6. Both explicitly requested
    elif has_expand and has_color:
        steps = ["segment", "expand", "color"]

    # 7. Default
    else:
        steps = ["segment", "expand", "color"]

    # -------------------------------------------------
    # 2️⃣ LLM FOR PARAMETER SELECTION ONLY
    # -------------------------------------------------

    prompt = f"""
    You are an image processing assistant.

    User request:
    "{user_msg}"

    Decide:
    - color mode: "random", "pink", or "blue"
    - expansion size: integer 1–10

    Return ONLY JSON:
    {{
        "palette": "...",
        "expand": ...
    }}
    """

    try:
        response = llm.invoke(prompt)
    except Exception:
        response = ""

    # -------------------------------------------------
    # 3️⃣ Robust parsing
    # -------------------------------------------------

    try:
        parsed = json.loads(response)
    except Exception:
        parsed = {
            "palette": "random",
            "expand": 3
        }

    # -------------------------------------------------
    # 4️⃣ Validation
    # -------------------------------------------------

    palette = parsed.get("palette", "random")
    if palette not in ["random", "pink", "blue"]:
        palette = "random"

    try:
        expand_val = int(parsed.get("expand", 3))
    except Exception:
        expand_val = 3

    expand_val = max(1, min(expand_val, 10))

    # -------------------------------------------------
    # 5️⃣ DEBUG (CRITICAL for your case)
    # -------------------------------------------------

    print("\n--- LLM DEBUG ---")
    print("Raw message:", user_msg)
    print("Normalized:", msg)
    print("Flags → segment:", has_segment,
          "expand:", has_expand,
          "color:", has_color,
          "only:", has_only)
    print("FINAL STEPS:", steps)
    print("LLM response:", response)
    print("Parsed params:", parsed)
    print("-----------------\n")

    # -------------------------------------------------
    # 6️⃣ Update state
    # -------------------------------------------------

    state["steps"] = steps
    state["palette"] = palette
    state["expand"] = expand_val

    return state