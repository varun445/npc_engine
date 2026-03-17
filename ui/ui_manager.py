import pygame
from ui.utils import wrap_text


class UIState:
    """Holds all mutable UI and dialogue state for a single frame."""

    def __init__(self):
        self.in_dialogue = False
        self.active_npc = None
        self.in_text_input = False
        self.player_input_text = ""
        self.customer_query = None
        self.npc_response = None
        self.npc_action = None
        self.is_waiting_for_llm = False
        self.show_npc_response = False
        self.response_scroll_offset = 0
        self.input_scroll_offset = 0
        # Aisle browsing menu
        self.aisle_menu_open = False
        self.active_aisle = None
        self.aisle_menu_items = []

    def reset_dialogue(self):
        """Fully exit dialogue mode and clear all related state."""
        self.in_dialogue = False
        self.active_npc = None
        self.in_text_input = False
        self.player_input_text = ""
        self.customer_query = None
        self.npc_response = None
        self.npc_action = None
        self.show_npc_response = False
        self.response_scroll_offset = 0
        self.input_scroll_offset = 0
        self.aisle_menu_open = False
        self.active_aisle = None
        self.aisle_menu_items = []

    def reset_for_next_message(self):
        """Keep dialogue open but clear the current exchange for a new query."""
        self.player_input_text = ""
        self.in_text_input = True
        self.npc_response = None
        self.npc_action = None
        self.customer_query = None
        self.show_npc_response = False
        self.response_scroll_offset = 0
        self.input_scroll_offset = 0


class UIManager:
    """Handles all pygame drawing: world entities and the dialogue panel."""

    PANEL_HEIGHT = 300
    PADDING = 20
    LINE_SPACING = 3

    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.font = pygame.font.SysFont(None, 24)
        self.aisle_font = pygame.font.SysFont(None, 18)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, screen, world_manager, interactable_npcs, ui_state):
        """Draw the full frame: world entities and, if open, the dialogue panel."""
        self._draw_world(screen, world_manager, interactable_npcs)
        self._draw_cart_hud(screen, world_manager)
        if ui_state.aisle_menu_open:
            self._draw_aisle_menu(screen, ui_state)
        elif ui_state.in_dialogue and ui_state.active_npc is not None:
            self._draw_dialogue_panel(screen, ui_state)

    # ------------------------------------------------------------------
    # World rendering
    # ------------------------------------------------------------------

    def _draw_world(self, screen, world_manager, interactable_npcs):
        self._draw_aisles(screen, world_manager)

        for npc in world_manager.npcs:
            npc_x = npc.col * self.cell_size
            npc_y = npc.row * self.cell_size
            pygame.draw.rect(screen, npc.color, (npc_x, npc_y, self.cell_size, self.cell_size))
            if npc in interactable_npcs:
                pygame.draw.rect(
                    screen,
                    (255, 255, 0),
                    (npc_x, npc_y, self.cell_size, self.cell_size),
                    3,
                )

        player_x = world_manager.player_col * self.cell_size
        player_y = world_manager.player_row * self.cell_size
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (player_x, player_y, self.cell_size, self.cell_size),
        )

    # ------------------------------------------------------------------
    # Aisle rendering
    # ------------------------------------------------------------------

    def _draw_aisles(self, screen, world_manager):
        """Draw each aisle as a coloured rectangle with a rotated text label."""
        for aisle in world_manager.aisles:
            start_col, start_row, end_col, end_row = aisle["grid_rect"]
            x = start_col * self.cell_size
            y = start_row * self.cell_size
            width = (end_col - start_col + 1) * self.cell_size
            height = (end_row - start_row + 1) * self.cell_size

            # Fill and border
            pygame.draw.rect(screen, aisle["color"], (x, y, width, height))
            pygame.draw.rect(screen, (220, 220, 220), (x, y, width, height), 2)

            # Render label rotated 90° so it fits inside the tall, narrow aisle
            label_surf = self.aisle_font.render(aisle["name"], True, (255, 255, 255))
            rotated = pygame.transform.rotate(label_surf, 90)
            label_x = x + width // 2 - rotated.get_width() // 2
            label_y = y + height // 2 - rotated.get_height() // 2
            screen.blit(rotated, (label_x, label_y))

    # ------------------------------------------------------------------
    # Dialogue panel
    # ------------------------------------------------------------------

    def _draw_dialogue_panel(self, screen, ui_state):
        panel_x = 0
        panel_y = self.height - self.PANEL_HEIGHT
        panel_rect = pygame.Rect(panel_x, panel_y, self.width, self.PANEL_HEIGHT)
        text_max_width = self.width - 2 * self.PADDING

        pygame.draw.rect(screen, (20, 20, 20), panel_rect)
        pygame.draw.rect(screen, (200, 200, 200), panel_rect, 2)

        npc = ui_state.active_npc
        role = getattr(npc, "role", "NPC")
        name_color = getattr(npc, "name_color", (200, 200, 200))
        name_surface = self.font.render(
            f"{npc.name} ({role})", True, name_color
        )
        screen.blit(name_surface, (self.PADDING, panel_y + 10))

        y_offset = panel_y + self.PADDING + 30

        if ui_state.in_text_input:
            self._draw_input_box(screen, ui_state, panel_x, panel_y, y_offset, text_max_width)
        elif ui_state.customer_query is not None:
            self._draw_response(screen, ui_state, panel_x, panel_y, y_offset, text_max_width)

    def _draw_input_box(self, screen, ui_state, panel_x, panel_y, y_offset, text_max_width):
        screen.blit(
            self.font.render("You: ", True, (200, 200, 100)),
            (panel_x + self.PADDING, y_offset),
        )

        input_box_height = self.PANEL_HEIGHT - 110
        input_box_rect = pygame.Rect(
            panel_x + self.PADDING + 50,
            y_offset,
            text_max_width - 50,
            input_box_height,
        )
        pygame.draw.rect(screen, (255, 255, 255), input_box_rect, 2)

        line_height = self.font.get_height() + self.LINE_SPACING
        wrapped_lines = wrap_text(
            ui_state.player_input_text if ui_state.player_input_text else " ",
            self.font,
            input_box_rect.width - 10,
        )
        max_visible = max(1, (input_box_rect.height - 10) // line_height)
        max_scroll = max(0, len(wrapped_lines) - max_visible)
        ui_state.input_scroll_offset = min(ui_state.input_scroll_offset, max_scroll)

        visible_start = max_scroll - ui_state.input_scroll_offset
        visible_lines = wrapped_lines[visible_start : visible_start + max_visible]

        input_y = input_box_rect.y + 5
        for line in visible_lines:
            screen.blit(
                self.font.render(line, True, (255, 255, 255)),
                (input_box_rect.x + 5, input_y),
            )
            input_y += line_height

        # Blinking cursor on the last visible line
        cursor_line_text = visible_lines[-1] if visible_lines else ""
        cursor_x = input_box_rect.x + 5 + self.font.size(cursor_line_text)[0]
        cursor_y = input_box_rect.y + 5 + (len(visible_lines) - 1) * line_height
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (cursor_x, cursor_y),
            (cursor_x, cursor_y + self.font.get_height()),
        )

        instruction = "Press ENTER to submit, ESC to cancel"
        if max_scroll > 0:
            instruction += ", UP/DOWN to scroll"
        screen.blit(
            self.font.render(instruction, True, (150, 150, 150)),
            (panel_x + self.PADDING, panel_y + self.PANEL_HEIGHT - 25),
        )

    def _draw_response(self, screen, ui_state, panel_x, panel_y, y_offset, text_max_width):
        query_lines = wrap_text(f"You: {ui_state.customer_query}", self.font, text_max_width)
        for line in query_lines[:2]:
            screen.blit(
                self.font.render(line, True, (200, 200, 100)),
                (panel_x + self.PADDING, y_offset),
            )
            y_offset += self.font.get_height() + self.LINE_SPACING

        y_offset += 5

        if ui_state.is_waiting_for_llm:
            dialogue_lines = ["Helping you find products..."]
        elif ui_state.npc_response is not None:
            dialogue_lines = wrap_text(ui_state.npc_response, self.font, text_max_width)
        else:
            dialogue_lines = []

        instructions_reserved = 25
        panel_y_abs = self.height - self.PANEL_HEIGHT
        available_height = (
            self.PANEL_HEIGHT - (y_offset - panel_y_abs) - instructions_reserved
        )
        line_height = self.font.get_height() + self.LINE_SPACING
        max_visible = max(1, available_height // line_height)
        max_scroll = max(0, len(dialogue_lines) - max_visible)
        ui_state.response_scroll_offset = min(ui_state.response_scroll_offset, max_scroll)

        visible_lines = dialogue_lines[
            ui_state.response_scroll_offset : ui_state.response_scroll_offset + max_visible
        ]
        for line in visible_lines:
            screen.blit(
                self.font.render(line, True, (255, 255, 255)),
                (panel_x + self.PADDING, y_offset),
            )
            y_offset += line_height

        if ui_state.show_npc_response and not ui_state.is_waiting_for_llm:
            instruction = "Press ENTER to continue, ESC to exit"
            if max_scroll > 0:
                instruction += ", UP/DOWN to scroll"
            screen.blit(
                self.font.render(instruction, True, (150, 150, 150)),
                (panel_x + self.PADDING, self.height - 25),
            )

    # ------------------------------------------------------------------
    # Cart HUD
    # ------------------------------------------------------------------

    def _draw_cart_hud(self, screen, world_manager):
        """Draw a small HUD in the top-right corner showing cart item count and total."""
        item_count = len(world_manager.player_cart)
        total = world_manager.get_cart_total()

        hud_w = 185
        hud_h = 52
        hud_x = self.width - hud_w - 8
        hud_y = 8

        pygame.draw.rect(screen, (40, 40, 40), (hud_x, hud_y, hud_w, hud_h))
        pygame.draw.rect(screen, (200, 200, 200), (hud_x, hud_y, hud_w, hud_h), 2)

        count_surf = self.font.render(f"Cart: {item_count} item(s)", True, (255, 255, 255))
        total_surf = self.font.render(f"Total: ${total:.2f}", True, (0, 220, 130))
        screen.blit(count_surf, (hud_x + 10, hud_y + 8))
        screen.blit(total_surf, (hud_x + 10, hud_y + 28))

    # ------------------------------------------------------------------
    # Aisle menu
    # ------------------------------------------------------------------

    def _draw_aisle_menu(self, screen, ui_state):
        """Draw the interactive aisle browsing menu."""
        items = ui_state.aisle_menu_items
        aisle = ui_state.active_aisle

        row_height = 28
        menu_w = 420
        menu_h = 80 + len(items) * row_height
        menu_x = (self.width - menu_w) // 2
        menu_y = (self.height - menu_h) // 2

        pygame.draw.rect(screen, (20, 20, 20), (menu_x, menu_y, menu_w, menu_h))
        pygame.draw.rect(screen, (200, 200, 200), (menu_x, menu_y, menu_w, menu_h), 2)

        title_text = f"Aisle {aisle['id']}: {aisle['name']}"
        title_surf = self.font.render(title_text, True, (255, 220, 60))
        screen.blit(title_surf, (menu_x + 15, menu_y + 14))

        sep_y = menu_y + 40
        pygame.draw.line(screen, (120, 120, 120), (menu_x + 10, sep_y), (menu_x + menu_w - 10, sep_y))

        for i, item in enumerate(items):
            label = f"{i + 1}. {item['name']}  —  ${item['price']:.2f}"
            item_surf = self.font.render(label, True, (255, 255, 255))
            screen.blit(item_surf, (menu_x + 15, menu_y + 50 + i * row_height))

        num_items = len(items)
        key_range = f"1-{num_items}" if num_items > 1 else "1"
        hint_surf = self.font.render(f"Press {key_range} to add to cart  |  ESC to close", True, (150, 150, 150))
        screen.blit(hint_surf, (menu_x + 15, menu_y + menu_h - 24))
