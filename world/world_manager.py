# ---------------------------------------------------------------------------
# Aisle definitions
# Each aisle entry describes a rectangular block on the grid that acts as
# both a visual landmark and a collidable obstacle.
#
# grid_rect: (start_col, start_row, end_col, end_row)  — all inclusive
# dest:      (row, col) where the NPC should stand when "at" this aisle
# color:     RGB fill colour for the rendered rectangle
# ---------------------------------------------------------------------------
AISLES = [
    {
        "id": 1,
        "name": "Dairy",
        "categories": ["dairy"],
        "grid_rect": (1, 2, 2, 9),
        "dest": (10, 1),
        "color": (70, 130, 180),
    },
    {
        "id": 2,
        "name": "Bakery",
        "categories": ["bakery"],
        "grid_rect": (5, 2, 6, 9),
        "dest": (10, 5),
        "color": (210, 160, 80),
    },
    {
        "id": 3,
        "name": "Produce",
        "categories": ["fruits", "vegetables"],
        "grid_rect": (9, 2, 10, 9),
        "dest": (10, 9),
        "color": (60, 180, 80),
    },
    {
        "id": 4,
        "name": "Beverages",
        "categories": ["beverages"],
        "grid_rect": (13, 2, 14, 9),
        "dest": (10, 13),
        "color": (150, 80, 200),
    },
    {
        "id": 5,
        "name": "Snacks",
        "categories": ["snacks"],
        "grid_rect": (17, 2, 18, 9),
        "dest": (10, 17),
        "color": (200, 80, 80),
    },
]


class WorldManager:
    """Manages player position, NPC list, world state, and spatial queries."""

    def __init__(self, rows, cols, cell_size):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.player_row = rows // 2
        self.player_col = cols // 2
        self.npcs = []
        self.aisles = AISLES
        self.obstacles = self._build_obstacles()
        self.world_state = {
            "store_busy": False,
            "current_time": "day",
        }
        self.player_cart = []

    # ------------------------------------------------------------------
    # Obstacle helpers
    # ------------------------------------------------------------------

    def _build_obstacles(self):
        """Return the set of (row, col) grid cells occupied by aisles."""
        cells = set()
        for aisle in self.aisles:
            start_col, start_row, end_col, end_row = aisle["grid_rect"]
            for r in range(start_row, end_row + 1):
                for c in range(start_col, end_col + 1):
                    cells.add((r, c))
        return cells

    def get_aisle_destination(self, aisle_id):
        """Return the (row, col) destination cell for the given aisle id."""
        for aisle in self.aisles:
            if aisle["id"] == aisle_id:
                return aisle["dest"]
        return None

    # ------------------------------------------------------------------
    # NPC management
    # ------------------------------------------------------------------

    def add_npc(self, npc):
        """Add an NPC to the world."""
        self.npcs.append(npc)

    def get_interactable_npcs(self):
        """Return (interactable_npcs, closest_npc) based on player position."""
        interactable = []
        closest = None
        min_dist = float("inf")
        for npc in self.npcs:
            dx = abs(self.player_row - npc.row)
            dy = abs(self.player_col - npc.col)
            distance = dx + dy
            if distance <= npc.interaction_range:
                interactable.append(npc)
                if distance < min_dist:
                    min_dist = distance
                    closest = npc
        return interactable, closest

    def get_nearby_aisle(self):
        """Return the nearest aisle within 1 step of the player, or None."""
        for aisle in self.aisles:
            start_col, start_row, end_col, end_row = aisle["grid_rect"]
            for r in range(start_row, end_row + 1):
                for c in range(start_col, end_col + 1):
                    if abs(self.player_row - r) + abs(self.player_col - c) <= 1:
                        return aisle
        return None

    # ------------------------------------------------------------------
    # Player cart management
    # ------------------------------------------------------------------

    def add_to_player_cart(self, item):
        """Add a product dict to the player's cart."""
        self.player_cart.append(item)

    def get_cart_total(self):
        """Return the total price of all items in the player's cart."""
        return sum(item["price"] for item in self.player_cart)

    def clear_player_cart(self):
        """Clear the player's cart."""
        self.player_cart = []

    # ------------------------------------------------------------------
    # Player movement (obstacle-aware)
    # ------------------------------------------------------------------

    def move_player(self, direction):
        """Move the player one cell in the given direction, respecting obstacles."""
        new_row, new_col = self.player_row, self.player_col
        if direction == "up":
            new_row -= 1
        elif direction == "down":
            new_row += 1
        elif direction == "left":
            new_col -= 1
        elif direction == "right":
            new_col += 1

        if (
            0 <= new_row < self.rows
            and 0 <= new_col < self.cols
            and (new_row, new_col) not in self.obstacles
        ):
            self.player_row = new_row
            self.player_col = new_col
