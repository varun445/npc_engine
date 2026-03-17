from world.pathfinding import astar


class NPC:
    MOVE_EVERY = 15  # frames between each path step (~4 cells/sec at 60 fps)

    def __init__(self, name, row, col, color, interaction_range, static_dialogue, personality):
        self.name = name
        self.row = row
        self.col = col
        self.color = color
        self.interaction_range = interaction_range
        self.static_dialogue = static_dialogue
        self.personality = personality
        self.memory = []
        self.role = "NPC"
        self.name_color = (200, 200, 200)
        # Path-following state
        self.path = []                # list of (row, col) steps remaining
        self.path_timer = 0           # counts frames since last step
        self.move_every = self.MOVE_EVERY
        self.destination_queue = []   # queue of (row, col) destinations to visit in order

    def update(self):
        """Advance the NPC one step along its current path when the timer fires.

        When the current path is exhausted, automatically begin pathfinding to
        the next destination in destination_queue (if any).
        """
        if not self.path:
            if self.destination_queue:
                next_dest = self.destination_queue.pop(0)
                # _obstacles and grid dimensions are stored on the NPC so the
                # update loop can call astar without referencing the world manager.
                if hasattr(self, "_obstacles") and hasattr(self, "_grid_size"):
                    rows, cols = self._grid_size
                    path = astar(
                        (self.row, self.col),
                        next_dest,
                        self._obstacles,
                        rows,
                        cols,
                    )
                    self.path = path
                    self.path_timer = 0
            return
        self.path_timer += 1
        if self.path_timer >= self.move_every:
            self.path_timer = 0
            next_cell = self.path.pop(0)
            self.row, self.col = next_cell


class ShopAssistant(NPC):
    """A specialized NPC for shop assistance."""

    def __init__(self, name, row, col, color, interaction_range, inventory):
        personality = (
            "A helpful, friendly shop assistant who knows the store layout and inventory "
            "well. Speaks clearly and politely, helps customers find products, answers "
            "questions about prices and stock, and provides recommendations. Professional, "
            "patient, and focused on customer satisfaction."
        )
        static_dialogue = [
            "Welcome to our store! How can I help?",
            "Looking for anything in particular?",
            "Let me help you find what you need.",
        ]
        super().__init__(name, row, col, color, interaction_range, static_dialogue, personality)
        self.role = "Shop Assistant"
        self.name_color = (0, 200, 100)
        self.inventory = inventory


class Cashier(NPC):
    """A stationary cashier NPC that handles checkout."""

    def __init__(self, name, row, col, color, interaction_range):
        personality = "A professional, efficient cashier who processes payments and helps customers checkout. Friendly, accurate with prices, and focused on completing transactions smoothly."
        static_dialogue = [
            "Ready to checkout when you are!",
            "Did you find everything you needed?",
            "I can help you with your purchase.",
        ]
        super().__init__(name, row, col, color, interaction_range, static_dialogue, personality)
        self.role = "Cashier"
        self.name_color = (100, 150, 255)
