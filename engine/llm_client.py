import requests

def query_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]
    except Exception as e:
        return f"[LLM Error: {e}]"

def generate_npc_response(name, personality):
    prompt = (
        f"You are an NPC named {name}.\n"
        f"Personality: {personality}\n\n"
        "Respond to the player in one or two sentences."
    )

    return query_llm(prompt)

def warmup_model():
    try:
        query_llm("Hello")
    except Exception:
        pass


