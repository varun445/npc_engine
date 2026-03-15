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


def extract_product_terms(customer_query):
    """Extract individual product names to search for from a customer query.

    This is a focused pre-processing step that runs before the main assistant
    call.  It converts any query — including recipe requests — into a flat list
    of individual grocery product names that can be looked up in the inventory.

    Examples:
        "Where can I find milk?" -> ["milk"]
        "I want to make a cake"  -> ["flour", "eggs", "sugar", "butter", "milk"]
        "Hello!"                 -> []

    Args:
        customer_query: The raw text typed by the customer.

    Returns:
        A list of individual product-name strings (may be empty for greetings
        or queries that mention no products).
    """
    prompt = f"""Your job is to extract individual grocery product names from a customer query.

Rules:
- If the customer mentions a recipe or dish (e.g. "cake", "pasta", "salad"), list the common ingredients for that recipe as individual product names.
- If the customer mentions specific products, list each product separately.
- If the query is a greeting or contains no product references, return an empty list.
- Each term must be a single, simple grocery product name (e.g. "milk", "flour", "eggs"). Never return compound phrases like "cake ingredients".

Return ONLY valid JSON in this exact format with no other text:
{{"terms": ["product1", "product2"]}}

Examples:
- "Where is the milk?" -> {{"terms": ["milk"]}}
- "I need ingredients for a cake" -> {{"terms": ["flour", "eggs", "sugar", "butter", "baking powder", "milk"]}}
- "Do you have apples and oranges?" -> {{"terms": ["apple", "orange"]}}
- "Hello there!" -> {{"terms": []}}

Customer query: "{customer_query}"
"""
    response = query_llm(prompt)
    try:
        data = json.loads(response)
        terms = data.get("terms", [])
        if isinstance(terms, list):
            return [str(t).strip() for t in terms if t]
        return []
    except json.JSONDecodeError:
        return []


def generate_shop_assistant_response(
    assistant_name, customer_query, inventory_info, memory, tool_observations=None
):
    """Generate a final Format B response from the shop assistant.

    The inventory has already been searched by the caller (see
    ``_fetch_npc_response``), so *tool_observations* always contains the
    relevant search results for product-related queries.  The LLM's only job
    here is to interpret those results and answer the customer.

    Returns a dict with keys: ``dialogue``, ``action``, ``target_aisles``.
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

    tool_obs_text = ""
    if tool_observations:
        tool_obs_text = "\nInventory Search Results:\n"
        for obs in tool_observations:
            tool_obs_text += f"- {obs}\n"

    prompt = f"""You are {assistant_name}, a friendly and helpful shop assistant in a grocery store.

{memory_text}
{tool_obs_text}
Available inventory (product name, price, aisle):
{inventory_info}

Customer just asked: "{customer_query}"

Instructions:
- Answer the customer honestly based ONLY on the Inventory Search Results above (if provided) or the available inventory list.
- If an item is listed as "not found in store inventory", tell the customer we do not carry it.
- If items are in different aisles, include ALL relevant aisle numbers in target_aisles.
- Keep your response to 1-2 sentences.

Reply with ONLY valid JSON in this exact format and nothing else:
{{
  "dialogue": "<your response to the customer>",
  "action": "<none or move>",
  "target_aisles": [<aisle numbers as integers, or empty []>]
}}

Use action "move" with target_aisles populated when directing the customer to specific aisles.
Use action "none" with empty target_aisles for greetings or when no movement is needed.
"""

    response = query_llm(prompt)
    try:
        result = json.loads(response)

        # Final response – normalise fields
        if "dialogue" not in result:
            result["dialogue"] = response
        # Normalise: ensure target_aisles is always a list
        if "target_aisles" not in result:
            legacy = result.get("target_aisle")
            try:
                result["target_aisles"] = [int(legacy)] if legacy is not None else []
            except (TypeError, ValueError):
                result["target_aisles"] = []
        elif not isinstance(result["target_aisles"], list):
            try:
                result["target_aisles"] = [int(result["target_aisles"])]
            except (TypeError, ValueError):
                result["target_aisles"] = []
        return result
    except json.JSONDecodeError:
        return {"dialogue": response, "action": "none", "target_aisles": []}


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



