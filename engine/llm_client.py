import requests
import json
import re
from models.inventory import SEARCH_RESULTS_PREFIX

# Set to True to print a structured workflow trace to the terminal.
# Each LLM call logs the full prompt sent to Ollama and the raw text
# response returned by it, plus per-step summaries (terms, action, aisles,
# dialogue snippet).  The full Ollama API metadata (eval counts, timings,
# etc.) is never printed — only the text "response" field is shown.
DEBUG = True


def _log(msg):
    """Print a debug line to stdout only when DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def _parse_json_flexible(text):
    """Parse JSON from LLM text.

    Args:
        text: Raw model response text.

    Returns:
        Parsed JSON value (dict/list/scalar) when decodable, else ``None``.
    """
    if not isinstance(text, str):
        return None

    # 1) Fast path: strict JSON payload.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Common LLM format: ```json ... ```
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fenced_blocks:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # 3) Fallback: locate first decodable JSON object/array inside mixed text.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[i:])
            return parsed
        except json.JSONDecodeError:
            continue

    return None


def query_llm(prompt):
    """Send a prompt to the local Ollama/Mistral model and return the text response."""
    if DEBUG:
        print(f"[DEBUG] ┌─ PROMPT ───────────────────────────────────────────────────────")
        for line in prompt.splitlines():
            print(f"[DEBUG] │ {line}")
        print(f"[DEBUG] └────────────────────────────────────────────────────────────────")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
            },
        )
        text = response.json()["response"]
        if DEBUG:
            print(f"[DEBUG] ┌─ RESPONSE ─────────────────────────────────────────────────")
            for line in text.splitlines():
                print(f"[DEBUG] │ {line}")
            print(f"[DEBUG] └────────────────────────────────────────────────────────────")
        return text
    except Exception as e:
        return f"[LLM Error: {e}]"


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
        data = _parse_json_flexible(response)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict but got {type(data).__name__}")
        terms = data.get("terms", [])
        if isinstance(terms, list):
            terms = [str(t).strip() for t in terms if t]
        else:
            terms = []
    except (json.JSONDecodeError, TypeError):
        terms = []
    _log(f"  terms extracted  → {terms}")
    return terms


def _format_search_observations(tool_observations):
    """Parse tool observation strings into clearly separated FOUND / NOT FOUND sections.

    Each observation string produced by ``inventory.search_inventory`` has the
    form::

        "Search Results: term1: value1; term2: not found in store inventory; ..."

    This function splits that blob into two explicit lists so the main LLM
    prompt can emphasise the found items and not overlook them.

    Returns a tuple ``(found_lines, not_found_names)`` where:
        found_lines  — list of strings like "milk: Milk (1L) (Aisle 1, $2.50)"
        not_found_names — list of term strings that were not in stock
    """
    found_lines = []
    not_found_names = []

    for obs in tool_observations:
        # Strip the "Search Results: " prefix if present
        body = obs[len(SEARCH_RESULTS_PREFIX):] if obs.startswith(SEARCH_RESULTS_PREFIX) else obs
        for entry in body.split("; "):
            if ": " not in entry:
                continue
            term, value = entry.split(": ", 1)
            term = term.strip()
            value = value.strip()
            if "not found in store inventory" in value:
                not_found_names.append(term)
            else:
                found_lines.append(f"{term}: {value}")

    return found_lines, not_found_names


def generate_shop_assistant_response(
    assistant_name,
    customer_query,
    inventory_info,
    memory,
    tool_observations=None,
    include_inventory_in_no_search=False,
):
    """Generate a final Format B response from the shop assistant.

    The inventory has already been searched by the caller (see
    ``_fetch_npc_response``), so *tool_observations* always contains the
    relevant search results for product-related queries.  The LLM's only job
    here is to interpret those results and answer the customer.

    The full inventory list is NOT included in this prompt — the pre-search
    results are sufficient and sending the entire catalogue adds noise that
    causes small models to overlook found items.

    When *tool_observations* is empty (conversational query — no products were
    identified by the pre-processing step), a lightweight conversational prompt
    is used instead so the assistant can respond naturally without being
    constrained by non-existent search results.

    If *include_inventory_in_no_search* is ``True``, the no-search prompt also
    includes the full inventory summary so evaluation can compare grounded
    responses even without deterministic pre-search results.

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
        memory_text += "\n"

    # Shared preamble used in both prompt branches.
    preamble = f"You are {assistant_name}, a friendly and helpful shop assistant in a grocery store."

    # ------------------------------------------------------------------ #
    # Conversational path — no product search was performed.              #
    # Use a simple prompt that lets the assistant respond naturally.       #
    # ------------------------------------------------------------------ #
    if not tool_observations:
        inventory_context = (
            f"\nAvailable store inventory reference:\n{inventory_info}\n"
            if include_inventory_in_no_search and inventory_info
            else ""
        )
        prompt = f"""{preamble}

{memory_text}{inventory_context}Customer just said: "{customer_query}"

Instructions:
- Respond naturally and helpfully as a shop assistant.
- You help customers find products, answer general questions about the store, and direct them to the right aisles when asked.
- If the customer asks what you can do or how you can help, explain that you can help them find products and add items to their cart.
- If the inventory reference above is provided, use it as the source of truth for product availability and aisle numbers.
- Keep your response to 1-3 sentences.
- Do NOT invent specific products or aisle numbers not present in the inventory reference.

Reply with ONLY valid JSON in this exact format and nothing else:
{{
  "dialogue": "<your response to the customer>",
  "action": "none",
  "target_aisles": []
}}
"""
    else:
        # ------------------------------------------------------------------ #
        # Product-search path — build FOUND / NOT FOUND sections.            #
        # ------------------------------------------------------------------ #
        found_lines, not_found_names = _format_search_observations(tool_observations)

        search_section = ""
        if found_lines:
            search_section += "\n*** ITEMS WE CARRY (you MUST tell the customer about each one with its aisle number) ***\n"
            for line in found_lines:
                search_section += f"  FOUND: {line}\n"

        if not_found_names:
            search_section += "\n*** ITEMS NOT IN STOCK (tell the customer we do not carry these) ***\n"
            for name in not_found_names:
                search_section += f"  NOT FOUND: {name}\n"

        prompt = f"""{preamble}

{memory_text}{search_section}
Customer just asked: "{customer_query}"

Instructions:
- Your answer MUST be based ONLY on the search results listed above.
- For every item in the FOUND section: tell the customer you are adding it to their cart
  and state its exact aisle number (e.g. "I'm adding Milk (1L) to your cart — you'll find
  it in Aisle 1!").
- For every item in the NOT FOUND section: tell the customer we do not carry it.
- If found items are in different aisles, list ALL those aisle numbers in target_aisles.
- Keep your response to 1-2 sentences.

ACTION RULES (follow exactly):
- If the FOUND section above contains any items → set action to "move" and list every
  found item's aisle number in target_aisles.
- If the FOUND section is empty (all items not found, or no search was done) → set
  action to "none" and target_aisles to [].
- NEVER set action to "none" when there are found items with aisle numbers.

Reply with ONLY valid JSON in this exact format and nothing else:
{{
  "dialogue": "<your response to the customer>",
  "action": "<none or move>",
  "target_aisles": [<aisle numbers as integers, or empty []>]
}}
"""

    response = query_llm(prompt)
    try:
        result = _parse_json_flexible(response)
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict but got {type(result).__name__}")

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
        # Safeguard: if aisles are populated the action must be "move".
        # A model may correctly identify aisles but forget to set the action.
        # (The conversational path hardcodes target_aisles=[] so this never
        # fires there — it is only relevant for the product-search path.)
        if result["target_aisles"] and result.get("action") != "move":
            result["action"] = "move"
        _log(f"  assistant result → action={result['action']} | aisles={result['target_aisles']} | \"{result.get('dialogue', '')[:80]}\"")
        return result
    except (json.JSONDecodeError, TypeError):
        _log(f"  assistant result → JSON parse failed; raw={response[:80]}")
        return {"dialogue": response, "action": "none", "target_aisles": []}



def generate_cashier_response(cashier_name, customer_query, cart_items, memory):
    """Generate a response for the Cashier NPC.

    The cashier always receives the full current cart so it never has to
    guess prices or item names.  If the customer asks to checkout / pay, the
    cashier returns ``{"action": "checkout", ...}`` so the game loop can clear
    the cart.  For all other queries the action is ``"none"``.

    Args:
        cashier_name: Name of the cashier NPC.
        customer_query: The raw text typed by the customer.
        cart_items: List of product dicts from ``world_manager.player_cart``.
        memory: List of past conversation turns for this NPC.

    Returns:
        A dict with keys ``dialogue`` and ``action``:
        - ``action`` is ``"checkout"`` when the customer requested payment and the
          cart is non-empty; the game loop should clear the cart and the NPC's
          memory when it receives this value.
        - ``action`` is ``"none"`` in all other cases, including when the customer
          tries to checkout with an empty cart (the cashier politely declines).
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
        memory_text += "\n"

    if cart_items:
        total = sum(item["price"] for item in cart_items)
        cart_lines = "\n".join(
            f"  - {item['name']}: ${item['price']:.2f}" for item in cart_items
        )
        cart_section = f"Customer's cart:\n{cart_lines}\n  TOTAL: ${total:.2f}"
    else:
        cart_section = "Customer's cart is currently empty."

    prompt = f"""You are {cashier_name}, a friendly cashier at a grocery store checkout.

{memory_text}{cart_section}

Customer just said: "{customer_query}"

STRICT ROLE GUARDRAILS — you MUST follow these without exception:
- You ONLY handle payments, cart review, and checkout. Nothing else.
- You do NOT know store layout, aisle numbers, or where any product is located.
  If a customer asks where to find a product or which aisle something is in, politely
  explain that you only work at the register and suggest they ask the shop assistant.
- NEVER mention aisle numbers or product locations in your response.
- NEVER describe, list, or mention any item that is NOT shown in the cart above.
  You MUST only report what is literally in the cart section above.

Instructions:
- If the customer wants to checkout, pay, or buy their items AND the cart is not empty:
  set action to "checkout" and provide a receipt-style dialogue that lists every item
  from the cart and states the final total.
- If the cart is empty and the customer tries to checkout or pay: politely tell them their
  cart is empty and set action to "none".
- If the customer mentions a specific product they want to buy but that product is NOT
  in their cart: explain that their cart does not contain that item and they should ask
  the shop assistant to help them find it first. Set action to "none".
- For all other requests (viewing cart, questions, greetings): describe only the items
  actually in the cart above and set action to "none".
- Keep your response concise (1-4 sentences).

Reply with ONLY valid JSON in this exact format and nothing else:
{{
  "dialogue": "<your response to the customer>",
  "action": "<none or checkout>"
}}
"""

    response = query_llm(prompt)
    try:
        result = _parse_json_flexible(response)
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict but got {type(result).__name__}")
        if "dialogue" not in result:
            result["dialogue"] = response
        if "action" not in result:
            result["action"] = "none"
        # Safety: prevent checkout of an empty cart
        if result.get("action") == "checkout" and not cart_items:
            result["action"] = "none"
        _log(f"  cashier result  → action={result['action']} | \"{result.get('dialogue', '')[:80]}\"")
        return result
    except (json.JSONDecodeError, TypeError):
        _log(f"  cashier result  → JSON parse failed; raw={response[:80]}")
        return {"dialogue": response, "action": "none"}
