class WorldManager:
    """Manages player position, NPC list, world state, and spatial queries."""

    def __init__(self, rows, cols, cell_size):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.player_row = rows // 2
        self.player_col = cols // 2
        self.npcs = []
        self.world_state = {
            "store_busy": False,
            "current_time": "day",
        }

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

    def move_player(self, direction):
        """Move the player one cell in the given direction."""
        if direction == "up" and self.player_row > 0:
            self.player_row -= 1
        elif direction == "down" and self.player_row < self.rows - 1:
            self.player_row += 1
        elif direction == "left" and self.player_col > 0:
            self.player_col -= 1
        elif direction == "right" and self.player_col < self.cols - 1:
            self.player_col += 1
