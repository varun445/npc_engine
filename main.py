import pygame

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

while running:
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
    


    screen.fill((30, 30, 30))

    player_x = player_col * CELL_SIZE
    player_y = player_row * CELL_SIZE
    pygame.draw.rect(screen, (255, 255, 255), (player_x, player_y, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
