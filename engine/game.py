import queue
import pygame

from engine.input_handler import MAX_MEMORY_TURNS
from ui.ui_manager import UIState
from world.pathfinding import astar


class Game:
    """Owns the pygame screen, clock, result queue, and the primary game loop."""

    def __init__(self, world_manager, ui_manager, input_handler, width=800, height=800):
        self.world = world_manager
        self.ui = ui_manager
        self.input_handler = input_handler
        self.result_queue = input_handler.result_queue
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Shop Assistant - NPC Engine")
        self.clock = pygame.time.Clock()
        self.ui_state = UIState()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            self._check_llm_queue()

            # Advance NPC path-following one step per frame
            for npc in self.world.npcs:
                npc.update()

            interactable_npcs, closest_npc = self.world.get_interactable_npcs()

            events = pygame.event.get()
            running = self.input_handler.handle_events(events, self.ui_state, closest_npc)

            self.screen.fill((30, 30, 30))
            self.ui.draw(self.screen, self.world, interactable_npcs, self.ui_state)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    # ------------------------------------------------------------------
    # LLM queue polling
    # ------------------------------------------------------------------

    def _check_llm_queue(self):
        """Non-blocking check of the result queue for completed LLM responses."""
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            return

        self.ui_state.is_waiting_for_llm = False
        self.ui_state.npc_response = result.get("dialogue", "")
        self.ui_state.npc_action = result.get("action")
        self.ui_state.show_npc_response = True

        npc = self.ui_state.active_npc
        if npc and self.ui_state.npc_response:
            last = npc.memory[-1] if npc.memory else None
            if last != {"role": "assistant", "content": self.ui_state.npc_response}:
                npc.memory.append({"role": "assistant", "content": self.ui_state.npc_response})
                if len(npc.memory) > MAX_MEMORY_TURNS * 2:
                    npc.memory = npc.memory[-MAX_MEMORY_TURNS * 2 :]

        # Dispatch NPC action (e.g., move to one or more target aisles, or checkout)
        if self.ui_state.npc_action == "checkout":
            self.world.clear_player_cart()
            # Clear cashier memory so the receipt doesn't leak into future conversations.
            if npc:
                npc.memory.clear()
        elif self.ui_state.npc_action == "move" and npc:
            # Add database-verified products to the player's cart (no hallucination —
            # these come from a pure-Python inventory lookup, not from the LLM).
            for product in result.get("add_to_cart", []):
                self.world.add_to_player_cart(product)

            target_aisles = result.get("target_aisles") or []
            # Backwards-compat: fall back to legacy single-value field
            if not target_aisles:
                legacy = result.get("target_aisle")
                if legacy is not None:
                    target_aisles = [int(legacy)]

            if target_aisles:
                destinations = []
                for aisle_id in target_aisles:
                    dest = self.world.get_aisle_destination(int(aisle_id))
                    if dest:
                        destinations.append(dest)

                if destinations:
                    # Attach world geometry to the NPC so its update() loop can
                    # call astar autonomously for subsequent queue entries.
                    npc._obstacles = self.world.obstacles
                    npc._grid_size = (self.world.rows, self.world.cols)

                    # Kick off pathfinding to the first destination immediately;
                    # the rest are held in the queue.
                    npc.destination_queue = list(destinations[1:])
                    first_dest = destinations[0]
                    path = astar(
                        (npc.row, npc.col),
                        first_dest,
                        self.world.obstacles,
                        self.world.rows,
                        self.world.cols,
                    )
                    npc.path = path
                    npc.path_timer = 0
