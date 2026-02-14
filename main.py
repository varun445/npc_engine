import pygame
from world.npc import NPC

pygame.init()

WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI NPC Engine")

CELL_SIZE = 40
ROWS = HEIGHT // CELL_SIZE
COLS = WIDTH // CELL_SIZE

player_row = ROWS // 2
player_col = COLS // 2
running = True
clock = pygame.time.Clock()

# NPC instantiation

npcs = [NPC("Gundalf", 0, 0, (0,255,0), 5, ["Greetings, Traveller.","Stay out of trouble.", "The night is dangerous."]), 
        NPC("Harvey",19,19, (255,0,0), 5,["I make my own luck.", "Life is like this and i like this.","Mikee..!!"]), 
        NPC("Mike",19,0,(0,0,255),5,["I have photographic memory", "Harveyyyy...!!!"])]

# Interactions

in_dialogue = False
active_npc = None
font = pygame.font.SysFont(None, 24)
dialogue_index = 0


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
            if not in_dialogue:
                if event.key == pygame.K_UP and player_row > 0:
                    player_row -= 1
                elif event.key == pygame.K_DOWN and player_row < ROWS-1:
                    player_row += 1
                elif event.key == pygame.K_LEFT and player_col > 0:
                    player_col -= 1
                elif event.key == pygame.K_RIGHT and player_col < COLS -1:
                    player_col += 1
                elif event.key == pygame.K_e:
                    if closest_npc is not None:
                        in_dialogue = True
                        active_npc = closest_npc
                        dialogue_index = (dialogue_index + 1) % len(active_npc.static_dialogue)
            elif event.key == pygame.K_ESCAPE:
                in_dialogue = False
                active_npc = None
        
    if in_dialogue and active_npc is not None:

        panel_height = 150
        panel_rect = pygame.Rect(
            0,
            HEIGHT - panel_height,
            WIDTH,
            panel_height
        )
        pygame.draw.rect(screen, (20, 20, 20), panel_rect)
        pygame.draw.rect(screen, (200, 200, 200), panel_rect, 2)

        name_text = font.render(active_npc.name, True, (255, 255, 255))
        dialogue_line = active_npc.static_dialogue[dialogue_index]
        dialogue_surface = font.render(dialogue_line, True, (255, 255, 255))
        screen.blit(name_text, (20, HEIGHT - 140))
        screen.blit(dialogue_surface, (20, HEIGHT - 110))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
