def run_agent(user_input: str):
    user_input = user_input.lower()

    if "segment" in user_input:
        return {"tool": "segment"}

    return {"tool": "none"}