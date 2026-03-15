import pygame
import threading

MAX_MEMORY_TURNS = 5


def _build_inventory_summary(inventory):
    """Build a concise text summary of in-stock products for the LLM prompt.

    Each line explicitly states the product name AND its aisle number so the
    LLM can look up exact names rather than guessing categories from internal
    knowledge.
    """
    lines = []
    for category, products in inventory.products.items():
        aisle_num = inventory.aisles[category]
        for product in products:
            if product["stock"] > 0:
                lines.append(
                    f"{product['name']} (${product['price']:.2f}) — category: {category}, Aisle {aisle_num}"
                )
    return "Store inventory:\n" + "\n".join(lines)


def _fetch_npc_response(npc, query, inventory, result_queue):
    """Background worker: calls the LLM and puts the parsed JSON result in result_queue."""
    from engine.llm_client import generate_shop_assistant_response

    inventory_summary = _build_inventory_summary(inventory)
    result = generate_shop_assistant_response(
        npc.name, query, inventory_summary, npc.memory
    )
    result_queue.put(result)


class InputHandler:
    """Routes pygame events to the correct handler based on the current game state."""

    def __init__(self, world_manager, inventory, result_queue):
        self.world = world_manager
        self.inventory = inventory
        self.result_queue = result_queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_events(self, events, ui_state, closest_npc):
        """Process a list of pygame events. Returns False when the game should quit."""
        for event in events:
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event, ui_state, closest_npc)
        return True

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _handle_keydown(self, event, ui_state, closest_npc):
        if ui_state.in_text_input:
            self._handle_text_input(event, ui_state)
        elif ui_state.in_dialogue:
            self._handle_dialogue_keys(event, ui_state)
        else:
            self._handle_free_roam(event, ui_state, closest_npc)

    # ------------------------------------------------------------------
    # State-specific handlers
    # ------------------------------------------------------------------

    def _handle_text_input(self, event, ui_state):
        if event.key == pygame.K_RETURN:
            if ui_state.player_input_text.strip():
                query = ui_state.player_input_text
                ui_state.customer_query = query
                ui_state.in_text_input = False
                ui_state.npc_response = None
                ui_state.npc_action = None
                ui_state.show_npc_response = False
                ui_state.response_scroll_offset = 0
                ui_state.input_scroll_offset = 0
                ui_state.is_waiting_for_llm = True

                if ui_state.active_npc:
                    ui_state.active_npc.memory.append({"role": "customer", "content": query})

                threading.Thread(
                    target=_fetch_npc_response,
                    args=(ui_state.active_npc, query, self.inventory, self.result_queue),
                    daemon=True,
                ).start()

        elif event.key == pygame.K_ESCAPE:
            ui_state.reset_dialogue()

        elif event.key == pygame.K_BACKSPACE:
            ui_state.player_input_text = ui_state.player_input_text[:-1]
            ui_state.input_scroll_offset = 0

        elif event.key == pygame.K_UP:
            ui_state.input_scroll_offset += 1

        elif event.key == pygame.K_DOWN:
            ui_state.input_scroll_offset = max(0, ui_state.input_scroll_offset - 1)

        elif event.unicode.isprintable():
            ui_state.player_input_text += event.unicode
            ui_state.input_scroll_offset = 0

    def _handle_dialogue_keys(self, event, ui_state):
        if event.key == pygame.K_RETURN:
            ui_state.reset_for_next_message()
        elif event.key == pygame.K_ESCAPE:
            ui_state.reset_dialogue()
        elif event.key == pygame.K_UP:
            ui_state.response_scroll_offset = max(0, ui_state.response_scroll_offset - 1)
        elif event.key == pygame.K_DOWN:
            ui_state.response_scroll_offset += 1

    def _handle_free_roam(self, event, ui_state, closest_npc):
        if event.key == pygame.K_UP:
            self.world.move_player("up")
        elif event.key == pygame.K_DOWN:
            self.world.move_player("down")
        elif event.key == pygame.K_LEFT:
            self.world.move_player("left")
        elif event.key == pygame.K_RIGHT:
            self.world.move_player("right")
        elif event.key == pygame.K_e and not ui_state.is_waiting_for_llm:
            if closest_npc is not None:
                ui_state.in_dialogue = True
                ui_state.in_text_input = True
                ui_state.active_npc = closest_npc
                ui_state.player_input_text = ""
                ui_state.npc_response = None
                ui_state.npc_action = None
                ui_state.customer_query = None
                ui_state.show_npc_response = False
                ui_state.response_scroll_offset = 0
                ui_state.input_scroll_offset = 0
        elif event.key == pygame.K_t:
            self.world.world_state["store_busy"] = not self.world.world_state["store_busy"]
        elif event.key == pygame.K_c:
            if ui_state.active_npc:
                ui_state.active_npc.clear_cart()
