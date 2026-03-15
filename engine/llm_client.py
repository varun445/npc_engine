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


def generate_shop_assistant_response(
    assistant_name, customer_query, inventory_info, memory, tool_observations=None
):
    """Generate a structured JSON response from the shop assistant.

    Supports the ReAct (Reasoning + Acting) loop.  When the LLM needs to look
    up items before answering it may return the intermediate action:

        {"action": "search_database", "search_terms": ["item1", "item2"]}

    The caller is responsible for executing the search, appending the result to
    *tool_observations*, and calling this function again.  When the LLM is
    ready to answer the customer it returns:

        {"dialogue": "...", "action": "none"|"move", "target_aisles": [...]}

    Args:
        assistant_name:    Name of the shop-assistant NPC.
        customer_query:    The raw text the customer typed.
        inventory_info:    Pre-built inventory summary string.
        memory:            List of {"role": ..., "content": ...} dicts.
        tool_observations: Optional list of search-result strings accumulated
                           during the current ReAct loop.
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
        tool_obs_text = "\nTool Observations (results from database searches you already ran):\n"
        for obs in tool_observations:
            tool_obs_text += f"- {obs}\n"

    prompt = f"""
    You are {assistant_name}, a friendly and helpful shop assistant in a grocery store.

    STRICT RULES — you MUST follow these exactly:
    1. NEVER guess or assume what is in stock. Use the "search_database" action (Format A) to
       verify items before answering any product-related question.
    2. The "search_terms" list MUST contain ONLY individual product names that a grocery store
       would carry (e.g., "flour", "eggs", "butter", "milk"). NEVER put recipe names, category
       names, or multi-word phrases like "cake ingredients", "baking supplies", or "dairy products"
       into search_terms — those will always return not found.
       If the customer asks about a recipe or "ingredients for X", first think about the individual
       products that recipe requires, then list each one separately in search_terms.
    3. You may only output Format A once per conversation turn. As soon as you have Tool
       Observations, you MUST output Format B. Do NOT issue another search_database action after
       already receiving search results.
    4. After receiving Tool Observations, honestly report what was found and what was not. If an
       item is "not found in store inventory", tell the customer we do not carry it.
    5. If search results confirm items in different aisles, include ALL relevant aisle numbers in
       target_aisles (e.g. [1, 3]).
    6. Keep your dialogue to 1-2 sentences maximum.

    EXAMPLE — Customer asks: "Can you list the ingredients for a cake and where to find them?"
      WRONG Format A:  {{"action": "search_database", "search_terms": ["Cake Ingredients"]}}
      CORRECT Format A: {{"action": "search_database", "search_terms": ["flour", "eggs", "sugar", "butter", "baking powder", "milk"]}}

    {memory_text}
    {tool_obs_text}

    Available inventory (each entry shows: product name, price, and its aisle number):
    {inventory_info}

    Customer just asked: "{customer_query}"

    You MUST reply with ONLY valid JSON in one of the two formats below and nothing else.

    Format A — search the database (use ONLY when you have no Tool Observations yet):
    {{
      "action": "search_database",
      "search_terms": ["<individual product name 1>", "<individual product name 2>"]
    }}

    Format B — final response to the customer (use after receiving Tool Observations, or for
    greetings/general questions that do not require an inventory lookup):
    {{
      "dialogue": "<your response to the customer>",
      "action": "<one of: none, move>",
      "target_aisles": [<aisle number(s) as integers, or empty list []>]
    }}

    Use Format A only when the customer asks about specific products or ingredients AND you have
    not yet searched. Use Format B with action "move" to direct the customer to aisles. Use
    Format B with action "none" for greetings, or when no movement is needed.
    """

    response = query_llm(prompt)
    try:
        result = json.loads(response)

        # Intermediate ReAct step: LLM wants to search the database
        if result.get("action") == "search_database":
            if not isinstance(result.get("search_terms"), list):
                print(
                    f"[ReAct] Warning: LLM returned 'search_database' with invalid "
                    f"search_terms ({result.get('search_terms')!r}). Resetting to []."
                )
                result["search_terms"] = []
            return result

        # Final response – normalise fields
        if "dialogue" not in result:
            result["dialogue"] = response
        # Normalise: ensure target_aisles is always a list
        if "target_aisles" not in result:
            # Fall back to legacy single-value field if present
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



