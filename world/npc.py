class NPC:
    def __init__(self, name, row, col, color, interaction_range, static_dialogue):
        self.name = name
        self.row = row
        self.col = col
        self.color = color
        self.interaction_range = interaction_range
        self.static_dialogue = static_dialogue
