from smolagents import CodeAgent
from smolagents.models import OpenAIModel

from tools.segmentation import segment_image
from tools.colorize import color_instances

import json
from openai import OpenAI


# ⚠️ smolagents model not used for LLM calls (kept for future)
model = OpenAIModel(
    model_id="llama3",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

agent = CodeAgent(
    tools=[segment_image, color_instances],
    model=model
)


# ✅ Direct Ollama client (this is what actually works)
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)


def run_agent(user_input: str, image_path: str):
    print("\n=== RUN_AGENT START ===")
    print("User input:", user_input)

    # 🧠 LLM extracts structured intent
    decision_prompt = f"""
User request: {user_input}

Extract the task and return JSON ONLY:

{{
  "task": "segment" OR "color",
  "color": "none" OR "random" OR a color name like "pink", "blue", "green"
}}

Rules:
- segmentation only → task = segment
- colorization → task = color
- if user mentions a color → include it
- if no color specified → use "random"

Examples:
- "segment image" → {{"task": "segment", "color": "none"}}
- "color nuclei" → {{"task": "color", "color": "random"}}
- "assign pink shades" → {{"task": "color", "color": "pink"}}

Return ONLY JSON.
"""

    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[{"role": "user", "content": decision_prompt}]
        )

        raw_output = response.choices[0].message.content
        print("LLM raw output:", raw_output)

        parsed = json.loads(raw_output)

        task = parsed.get("task", "segment")
        color_mode = parsed.get("color", "none")

    except Exception as e:
        print("❌ LLM ERROR:", str(e))
        task = "segment"
        color_mode = "none"

    # 🔥 Safety override (important)
    if "color" in user_input.lower():
        task = "color"

    print("Final task:", task)
    print("Color mode:", color_mode)

    # 🔥 EXECUTION (deterministic)
    if task == "color":
        print(">>> Running segmentation + colorization")

        mask_path = segment_image(image_path)

        # Pass color mode to tool
        return color_instances(mask_path, color_mode)

    elif task == "segment":
        print(">>> Running segmentation only")
        return segment_image(image_path)

    print("⚠️ fallback segmentation")
    return segment_image(image_path)