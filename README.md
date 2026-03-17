# NPC Engine — AI-Powered 2D Game Simulation

A top-down 2D grocery-store simulation powered by a **local LLM** (Ollama + Mistral).
Shop assistants and cashiers are fully AI-driven agents that reason, search a live inventory, and navigate the world using A* pathfinding — all in real time.

---

## Features

| Feature | Description |
|---|---|
| **ReAct Tool-Calling Loop** | Each NPC follows a _Reason → Act → Observe_ cycle. The assistant extracts product terms from the customer's query, searches the inventory database, and feeds verified results back to the LLM — eliminating hallucination. |
| **A* Pathfinding** | When the shop assistant locates products, it navigates autonomously to every matching aisle using a Manhattan-heuristic A* algorithm. Multi-destination queues let it visit several aisles in sequence. |
| **Aisle Interaction & Menus** | Walk up to any aisle and press **E** to open a context-sensitive popup listing every product in that section with its price. Press keys **1–9** to add items to your cart instantly. |
| **Player Cart & HUD** | A persistent cart tracks everything you've added (manually at an aisle, or automatically by the shop assistant). A HUD in the top-right corner always shows the item count and running total. |
| **Cashier Multi-Agent** | A second independent NPC (Bob) operates the checkout counter with its own LLM prompt and tool set (`view_cart`, `checkout`). It reads the live cart state, generates a receipt, and clears the cart on confirmation — without ever seeing the assistant's context. |
| **Conversation Memory** | Each NPC maintains a rolling conversation history (last 5 exchanges) so follow-up questions feel natural. Memory is cleared after checkout so receipts don't leak into future sessions. |

---

## Architecture

```
+-----------------------------------------------------+
|                    Pygame Frontend                  |
|  +------------+  +-------------+  +--------------+  |
|  | UIManager  |  | InputHandler|  |  Game Loop   |  |
|  | (rendering)|  |  (events)   |  | (60 FPS tick)|  |
|  +------------+  +-------------+  +--------------+  |
|         |                |                |          |
|         +----------------+----------------+          |
|                          |                           |
|                   WorldManager                       |
|         (player pos · NPCs · aisles · cart)          |
+---------------------------+-------------------------+
                            | background thread
+---------------------------v-------------------------+
|               LLM Client (engine/llm_client.py)    |
|                                                     |
|  1. extract_product_terms()  <- first LLM call      |
|  2. inventory.search_inventory()  <- pure Python    |
|  3. generate_shop_assistant_response()  <- main LLM |
|         -- or --                                    |
|     generate_cashier_response()                     |
+---------------------------+-------------------------+
                            | HTTP POST /api/generate
               +------------v------------+
               |   Ollama  (localhost)   |
               |   Model: mistral        |
               +-------------------------+
```

**Key design decisions:**

- **Pre-search before LLM** — inventory is searched with pure Python _before_ the main LLM call. The model only needs to interpret verified results, not decide what to search for. This removes the most common source of hallucination in small local models.
- **Daemon threads** — LLM calls run in daemon background threads so the game loop never blocks. A `queue.Queue` carries the result back to the main thread.
- **Role isolation** — the shop assistant and cashier have entirely separate prompts, tool sets, and memory. Neither can stray into the other's domain.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### 1 — Clone the repository
```bash
git clone https://github.com/varun445/npc_engine.git
cd npc_engine
```

### 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3 — Pull the Mistral model
```bash
ollama pull mistral
```

### 4 — Start Ollama (if not already running)
```bash
ollama serve
```

### 5 — Run the game
```bash
python main.py
```

---

## Controls

| Key | Action |
|---|---|
| **Arrow keys** | Move the player around the store |
| **E** | Interact with the nearest NPC _or_ open the aisle menu when standing next to a shelf |
| **1 – 9** | (Aisle menu open) Add the corresponding item to your cart |
| **ESC** | Close any open dialogue or aisle menu |
| **Enter** | Submit your typed message to the NPC / advance to the next dialogue screen |
| **Up / Down** | Scroll long NPC responses or the text-input box |
| **C** | Clear the player cart at any time |
| **T** | Toggle the store-busy world flag (cosmetic, passed to NPC context) |

### Talking to the Shop Assistant (Alex)
1. Walk near the green NPC and press **E**.
2. Type any natural-language query — product lookup, recipe request, or a general question.
3. Alex will search the inventory and navigate to the relevant aisles. Matched products are automatically added to your cart.

### Checking Out with the Cashier (Bob)
1. Walk near the blue NPC and press **E**.
2. Ask Bob to view your cart or tell him you're ready to pay.
3. Bob will list every item, state the total, and clear your cart on confirmation.

---

## Project Structure

```
npc_engine/
├── main.py                  # Entry point — wires all components together
├── requirements.txt
├── engine/
│   ├── game.py              # Main game loop, LLM result queue dispatch
│   ├── input_handler.py     # Pygame event routing, background LLM workers
│   └── llm_client.py        # Ollama API wrapper, ReAct prompts
├── world/
│   ├── world_manager.py     # Grid layout, aisle definitions, player cart
│   ├── npc.py               # NPC base class, ShopAssistant, Cashier
│   └── pathfinding.py       # A* implementation
├── models/
│   └── inventory.py         # Product database and search logic
└── ui/
    ├── ui_manager.py        # Pygame rendering (world, HUD, dialogue, aisle menu)
    └── utils.py             # Text-wrapping helper
```

---

## How the ReAct Loop Works (Shop Assistant)

```
Player types: "I need ingredients for a cake"
        |
        v
1. extract_product_terms()   ->  ["flour", "eggs", "sugar", "butter", "milk"]
        |
        v
2. inventory.search_inventory()  ->  "flour: not found; eggs: not found;
                                      sugar: not found; butter: not found;
                                      milk: Milk (1L) (Aisle 1, $2.50)"
        |
        v
3. generate_shop_assistant_response()
        |  Prompt includes verified FOUND / NOT FOUND sections
        v
   LLM output (JSON):
   {
     "dialogue": "I found Milk (1L) in Aisle 1 and am adding it to your cart!
                  Unfortunately we don't carry flour, eggs, sugar, or butter.",
     "action": "move",
     "target_aisles": [1]
   }
        |
        v
4. Game loop receives result:
   - Milk (1L) added to player_cart
   - Alex navigates to Aisle 1 via A*
```

---

## Roadmap

- **Phase 3 — Agentic Delegation**: Allow the shop assistant and cashier to transfer the conversation to each other via a `delegate` action, enabling seamless multi-agent handoffs.
- **Phase 4 — Autonomous AI Customers**: Spawn LLM-driven customer NPCs that shop independently, creating AI-to-AI interactions.
- **Phase 5 — Persistent Memory & World Events**: Vector-store conversation history and daily world-event injection so NPCs adapt their behaviour over time.
- **Phase 6 — Manager Mode**: Step back from the shop floor and direct your AI staff with high-level natural-language commands from a manager's console.
