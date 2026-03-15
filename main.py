import queue

import pygame

from engine.game import Game
from engine.input_handler import InputHandler
from models.inventory import Inventory
from ui.ui_manager import UIManager
from world.npc import ShopAssistant
from world.world_manager import WorldManager


def main():
    pygame.init()

    WIDTH, HEIGHT = 800, 800
    CELL_SIZE = 40
    ROWS = HEIGHT // CELL_SIZE
    COLS = WIDTH // CELL_SIZE

    inventory = Inventory()
    result_queue = queue.Queue()

    world = WorldManager(rows=ROWS, cols=COLS, cell_size=CELL_SIZE)
    shop_assistant = ShopAssistant(
        name="Alex",
        row=16,
        col=9,
        color=(0, 200, 100),
        interaction_range=5,
        inventory=inventory,
    )
    world.add_npc(shop_assistant)

    ui = UIManager(width=WIDTH, height=HEIGHT, cell_size=CELL_SIZE)
    input_handler = InputHandler(
        world_manager=world, inventory=inventory, result_queue=result_queue
    )
    game = Game(
        world_manager=world,
        ui_manager=ui,
        input_handler=input_handler,
        width=WIDTH,
        height=HEIGHT,
    )
    game.run()


if __name__ == "__main__":
    main()
