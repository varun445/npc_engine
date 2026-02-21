import requests
DEBUG_LLM = True

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
        if DEBUG_LLM:
            print(prompt)
            print("\n")
            print(response)
        return response.json()["response"]
    except Exception as e:
        return f"[LLM Error: {e}]"

def generate_npc_response(name, personality, world_state, memory):
    memory_text = ""
    if memory:
        memory_text = "Recent conversation:\n"
        for msg in memory:
            memory_text += f"{msg['role']}: {msg['content']}\n"

    prompt = (
        f"You are an NPC named {name}.\n"
        f"Personality: {personality}.\n\n"
        f"{memory_text}\n"
        "World Context:\n"
        f"- Time of day: {world_state['time_of_day']}\n"
        f"- Player reputation: {world_state['player_reputation']}\n\n"
        "Respond to the player in one or two sentences."
    )

    return query_llm(prompt)

def warmup_model():
    try:
        query_llm("Hello")
    except Exception:
        pass

import json

def categorize_products(user_query):
    prompt = f"""
    You are a grocery store assistant.

    From the user query below, extract the products mentioned
    and assign each one to a grocery category.

    Valid categories:
    - dairy
    - bakery
    - fruits
    - vegetables
    - beverages
    - snacks

    If you are unsure, use "unknown".

    Return ONLY valid JSON in the following format:

    {{
    "products": [
        {{ "name": "<product_name>", "category": "<category>" }}
    ]
    }}

    User query:
    "{user_query}"
    """

    response = query_llm(prompt)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"products": []}


