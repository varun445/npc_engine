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

npcs = [NPC("gundalf", 0, 0, (0,255,0), 5), NPC("bob",19,19, (255,0,0), 5), NPC("alex",19,0,(0,0,255),5)]

while running:

    screen.fill((30, 30, 30))

    interactable_npcs = []
    for npc in npcs:
        dx = abs(player_row - npc.row)
        dy = abs(player_col - npc.col)
        distance = dx + dy

        if distance <= npc.interaction_range:
            interactable_npcs.append(npc)

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
            if event.key == pygame.K_UP and player_row > 0:
                player_row -= 1
            elif event.key == pygame.K_DOWN and player_row < ROWS-1:
                player_row += 1
            elif event.key == pygame.K_LEFT and player_col > 0:
                player_col -= 1
            elif event.key == pygame.K_RIGHT and player_col < COLS -1:
                player_col += 1
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
