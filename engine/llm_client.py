import requests
import json
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
            print(response.json())
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


def detect_customer_intent(user_query):
    """Detect what the customer wants: search, recommendation, price, stock, complaint, etc."""
    prompt = f"""
    Analyze the customer's query and determine their intent. Return ONLY valid JSON.
    
    Valid intents:
    - search (looking for a product)
    - recommendation (asking for product suggestions)
    - price (asking about price)
    - stock (asking if item is available)
    - complaint (expressing dissatisfaction)
    - greeting (hello, hi, etc)
    - checkout (ready to buy)
    - other
    
    Customer query: "{user_query}"
    
    Return ONLY this JSON format:
    {{"intent": "<intent>", "confidence": <0-100>}}
    """
    
    response = query_llm(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"intent": "other", "confidence": 0}


def generate_shop_assistant_response(assistant_name, customer_query, inventory_info, memory):
    """Generate a structured JSON response from the shop assistant.

    Returns a dict with keys:
        dialogue     – the text the NPC says to the customer
        action       – one of: "none", "move"
        target_aisle – aisle number (int) when action is "move", else null
    """
    memory_text = ""
    if memory:
        memory_text = "Previous conversation:\n"
        for msg in memory:
            memory_text += (
                f"Customer: {msg['content']}\n"
                if msg["role"] == "customer"
                else f"You: {msg['content']}\n"
            )

    prompt = f"""
    You are {assistant_name}, a friendly and helpful shop assistant.

    {memory_text}

    Available inventory information:
    {inventory_info}

    Customer just asked: "{customer_query}"

    Respond helpfully and naturally in 1-2 sentences. Be friendly, professional, and knowledgeable about the products.

    You MUST reply with ONLY valid JSON in exactly this format and nothing else:
    {{
      "dialogue": "<your response to the customer>",
      "action": "<one of: none, move>",
      "target_aisle": <aisle number as an integer, or null>
    }}

    Use action "move" and set target_aisle when you are directing the customer to a specific aisle.
    Use action "none" for all other responses.
    """

    response = query_llm(prompt)
    try:
        result = json.loads(response)
        if "dialogue" not in result:
            result["dialogue"] = response
        return result
    except json.JSONDecodeError:
        return {"dialogue": response, "action": "none", "target_aisle": None}


def find_products_in_inventory(user_query, available_products):
    """Find specific products matching user's query from available inventory"""
    prompt = f"""
    The customer is looking for: "{user_query}"
    
    Available products in the store:
    {available_products}
    
    Find the best matching products from the available list. Return ONLY valid JSON.
    
    Return format:
    {{
      "found": true/false,
      "products": [
        {{"name": "Product Name", "price": 9.99, "aisle": 1, "in_stock": true}}
      ],
      "suggestions": ["alternative suggestion 1", "alternative suggestion 2"]
    }}
    """
    
    response = query_llm(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"found": False, "products": [], "suggestions": []}


def generate_product_recommendation(customer_preferences, available_products):
    """Generate product recommendations based on customer preferences"""
    prompt = f"""
    A customer has expressed the following preferences: "{customer_preferences}"
    
    Based on this, recommend 2-3 products from the available inventory:
    {available_products}
    
    Return ONLY valid JSON:
    {{
      "recommendations": [
        {{"name": "Product Name", "reason": "why this product is good for them", "price": 9.99}}
      ]
    }}
    """
    
    response = query_llm(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"recommendations": []}



