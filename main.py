import pygame
from world.npc import ShopAssistant
from engine.llm_client import (
    generate_shop_assistant_response, 
    detect_customer_intent,
    find_products_in_inventory,
    warmup_model
)
from models.inventory import Inventory
import threading
import random
from ui.utils import wrap_text

pygame.init()

WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shop Assistant - NPC Engine")

CELL_SIZE = 40
ROWS = HEIGHT // CELL_SIZE
COLS = WIDTH // CELL_SIZE

player_row = ROWS // 2
player_col = COLS // 2
running = True
clock = pygame.time.Clock()

# Initialize inventory
inventory = Inventory()

world_state = {
    "store_busy": False,
    "current_time": "day"
}

# Shop Assistant instantiation
shop_assistant = ShopAssistant(
    name="Alex",
    row=0,
    col=0,
    color=(0, 200, 100),
    interaction_range=5,
    inventory=inventory
)
npcs = [shop_assistant]  # Keep NPCs list for compatibility with movement logic


# NPC Interactions
MAX_MEMORY_TURNS = 5
in_dialogue = False
active_npc = None
font = pygame.font.SysFont(None, 24)
dialogue_index = 0
npc_response = None
is_waiting_for_llm = False
customer_query = None
in_text_input = False
player_input_text = ""
show_npc_response = False
response_scroll_offset = 0
input_scroll_offset = 0

def fetch_npc_response(npc, query):
    global npc_response, is_waiting_for_llm
    
    # Prepare inventory info for the LLM
    inventory_summary = "Available products: "
    product_list = []
    for category, products in inventory.products.items():
        for product in products:
            if product["stock"] > 0:
                product_list.append(f"{product['name']} (${product['price']:.2f}, Aisle {inventory.aisles[category]})")
    inventory_summary += " | ".join(product_list[:10])  # Limit for context
    
    # Generate response using shop assistant prompt
    npc_response = generate_shop_assistant_response(
        npc.name, 
        query, 
        inventory_summary,
        npc.memory
    )
    is_waiting_for_llm = False


def calculate_visible_line_count(total_height, line_height):
    return max(1, total_height // line_height)

# Pre-Loading the model so the responses are quicker
# threading.Thread(
#     target=warmup_model,
#     daemon=True
# ).start()

while running:

    screen.fill((30, 30, 30))

    interactable_npcs = []
    closest_npc = None
    min_distance = float("inf")
    for npc in npcs:
        dx = abs(player_row - npc.row)
        dy = abs(player_col - npc.col)
        distance = dx + dy

        if distance <= npc.interaction_range:
            interactable_npcs.append(npc)
            if distance < min_distance:
                closest_npc = npc

    for npc in npcs:
        npc_x = npc.col * CELL_SIZE
        npc_y = npc.row * CELL_SIZE
        pygame.draw.rect(screen, npc.color, (npc_x,npc_y,CELL_SIZE,CELL_SIZE))

        if npc in interactable_npcs:
            pygame.draw.rect(
            screen,
            (255, 255, 0),  # yellow highlight
            (npc_x, npc_y, CELL_SIZE, CELL_SIZE),
            3  # border thickness
        )

    player_x = player_col * CELL_SIZE
    player_y = player_row * CELL_SIZE
    pygame.draw.rect(screen, (255, 255, 255), (player_x, player_y, CELL_SIZE, CELL_SIZE))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if in_text_input:
                # Text input mode
                if event.key == pygame.K_RETURN:
                    if player_input_text.strip():
                        customer_query = player_input_text
                        in_text_input = False
                        npc_response = None
                        show_npc_response = False
                        response_scroll_offset = 0
                        input_scroll_offset = 0
                        is_waiting_for_llm = True

                        threading.Thread(
                            target=fetch_npc_response,
                            args=(active_npc, customer_query),
                            daemon=True
                        ).start()
                        
                        # Add to memory
                        if active_npc:
                            active_npc.memory.append({
                                "role": "customer",
                                "content": customer_query
                            })
                
                elif event.key == pygame.K_ESCAPE:
                    in_text_input = False
                    player_input_text = ""
                    in_dialogue = False
                    active_npc = None
                    customer_query = None
                    response_scroll_offset = 0
                    input_scroll_offset = 0
                
                elif event.key == pygame.K_BACKSPACE:
                    player_input_text = player_input_text[:-1]
                    input_scroll_offset = 0
                
                elif event.unicode.isprintable():
                    player_input_text += event.unicode
                    input_scroll_offset = 0

                elif event.key == pygame.K_UP:
                    input_scroll_offset += 1

                elif event.key == pygame.K_DOWN:
                    input_scroll_offset = max(0, input_scroll_offset - 1)
            
            elif in_dialogue and not in_text_input:
                # In dialogue but not typing - waiting for assistance or continuing
                if event.key == pygame.K_RETURN:
                    # Reset for next message
                    player_input_text = ""
                    in_text_input = True
                    npc_response = None
                    customer_query = None
                    show_npc_response = False
                    response_scroll_offset = 0
                    input_scroll_offset = 0
                
                elif event.key == pygame.K_ESCAPE:
                    in_dialogue = False
                    in_text_input = False
                    player_input_text = ""
                    active_npc = None
                    customer_query = None
                    npc_response = None
                    response_scroll_offset = 0
                    input_scroll_offset = 0

                elif event.key == pygame.K_UP:
                    response_scroll_offset = max(0, response_scroll_offset - 1)

                elif event.key == pygame.K_DOWN:
                    response_scroll_offset += 1
            
            elif not in_dialogue:
                if event.key == pygame.K_UP and player_row > 0:
                    player_row -= 1
                elif event.key == pygame.K_DOWN and player_row < ROWS-1:
                    player_row += 1
                elif event.key == pygame.K_LEFT and player_col > 0:
                    player_col -= 1
                elif event.key == pygame.K_RIGHT and player_col < COLS -1:
                    player_col += 1
                elif event.key == pygame.K_e and not is_waiting_for_llm:
                    if closest_npc is not None:
                        in_dialogue = True
                        in_text_input = True
                        active_npc = closest_npc
                        player_input_text = ""
                        dialogue_index = 0
                        npc_response = None
                        customer_query = None
                        show_npc_response = False
                        response_scroll_offset = 0
                        input_scroll_offset = 0

                elif event.key == pygame.K_t:
                    world_state["store_busy"] = not world_state["store_busy"]

                elif event.key == pygame.K_c:
                    # Clear customer's cart
                    active_npc.clear_cart() if active_npc else None

        
    if in_dialogue and active_npc is not None:

        # Dialogue Box Dimensions
        panel_height = 300
        panel_width = WIDTH
        panel_x = 0
        panel_y = HEIGHT - panel_height
        
        padding = 20
        text_max_width = panel_width - 2 * padding

        # Drawing the dialogue box
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        pygame.draw.rect(screen, (20, 20, 20), panel_rect)
        pygame.draw.rect(screen, (200, 200, 200), panel_rect, 2)

        name_text = font.render(active_npc.name + " (Shop Assistant)", True, (0, 200, 100))
        screen.blit(name_text, (20, panel_y + 10))

        y_offset = panel_y + padding + 30

        # If in text input mode, show input field
        if in_text_input:
            input_label = font.render("You: ", True, (200, 200, 100))
            screen.blit(input_label, (panel_x + padding, y_offset))
            
            # Input box
            input_box_height = panel_height - 110
            input_box_rect = pygame.Rect(panel_x + padding + 50, y_offset, text_max_width - 50, input_box_height)
            pygame.draw.rect(screen, (255, 255, 255), input_box_rect, 2)
            
            input_line_height = font.get_height() + 3
            wrapped_input_lines = wrap_text(player_input_text if player_input_text else " ", font, input_box_rect.width - 10)
            max_visible_input_lines = calculate_visible_line_count(input_box_rect.height - 10, input_line_height)
            max_input_scroll = max(0, len(wrapped_input_lines) - max_visible_input_lines)
            input_scroll_offset = min(input_scroll_offset, max_input_scroll)
            visible_input_start = max_input_scroll - input_scroll_offset

            visible_input_lines = wrapped_input_lines[visible_input_start:visible_input_start + max_visible_input_lines]
            input_y = input_box_rect.y + 5
            for line in visible_input_lines:
                line_surface = font.render(line, True, (255, 255, 255))
                screen.blit(line_surface, (input_box_rect.x + 5, input_y))
                input_y += input_line_height

            # Cursor on the final visible line.
            cursor_line_text = visible_input_lines[-1] if visible_input_lines else ""
            cursor_x = input_box_rect.x + 5 + font.size(cursor_line_text)[0]
            cursor_y = input_box_rect.y + 5 + (len(visible_input_lines) - 1) * input_line_height
            pygame.draw.line(screen, (255, 255, 255), (cursor_x, cursor_y), (cursor_x, cursor_y + font.get_height()))
            
            # Instructions
            input_instruction_text = "Press ENTER to submit, ESC to cancel"
            if max_input_scroll > 0:
                input_instruction_text += ", UP/DOWN to scroll"
            instructions_text = font.render(input_instruction_text, True, (150, 150, 150))
            screen.blit(instructions_text, (panel_x + padding, panel_y + panel_height - 25))
        
        # Show assistant response after query is submitted
        elif not in_text_input and customer_query is not None:
            # Display customer's query
            query_lines = wrap_text(f"You: {customer_query}", font, text_max_width)
            for line in query_lines[:2]:  # Limit to 2 lines for query
                line_surface = font.render(line, True, (200, 200, 100))
                screen.blit(line_surface, (panel_x + padding, y_offset))
                y_offset += font.get_height() + 3
            
            y_offset += 5
            
            # Display assistant response
            if is_waiting_for_llm:
                dialogue_lines = ["Helping you find products..."]
            elif npc_response is not None:
                show_npc_response = True
                dialogue_lines = wrap_text(npc_response, font, text_max_width)
                
                # Add to memory when response is first shown
                if len(active_npc.memory) > 0 and active_npc.memory[-1]["role"] == "customer":
                    if len(active_npc.memory) == 0 or active_npc.memory[-1] != {"role": "assistant", "content": npc_response}:
                        active_npc.memory.append({
                            "role": "assistant",
                            "content": npc_response
                        })
                        
                        # Trim memory
                        if len(active_npc.memory) > MAX_MEMORY_TURNS * 2:
                            active_npc.memory = active_npc.memory[-MAX_MEMORY_TURNS * 2:]
            else:
                dialogue_lines = []
            
            instructions_reserved_height = 25
            available_height = panel_height - (y_offset - panel_y) - instructions_reserved_height
            max_visible_lines = max(1, available_height // (font.get_height() + 3))
            max_scroll = max(0, len(dialogue_lines) - max_visible_lines)
            response_scroll_offset = min(response_scroll_offset, max_scroll)

            visible_lines = dialogue_lines[response_scroll_offset:response_scroll_offset + max_visible_lines]
            for line in visible_lines:
                line_surface = font.render(line, True, (255, 255, 255))
                screen.blit(line_surface, (panel_x + padding, y_offset))
                y_offset += font.get_height() + 3
            
            # Instructions for next step
            if show_npc_response and not is_waiting_for_llm:
                instructions_text = "Press ENTER to continue, ESC to exit"
                if max_scroll > 0:
                    instructions_text += ", UP/DOWN to scroll"
                instructions_surface = font.render(instructions_text, True, (150, 150, 150))
                screen.blit(instructions_surface, (panel_x + padding, HEIGHT - 25))


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
