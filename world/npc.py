class NPC:
    def __init__(self, name, row, col, color, interaction_range, static_dialogue, personality):
        self.name = name
        self.row = row
        self.col = col
        self.color = color
        self.interaction_range = interaction_range
        self.static_dialogue = static_dialogue
        self.personality = personality
        self.memory = []


class ShopAssistant(NPC):
    """A specialized NPC for shop assistance"""
    def __init__(self, name, row, col, color, interaction_range, inventory):
        personality = "A helpful, friendly shop assistant who knows the store layout and inventory well. Speaks clearly and politely, helps customers find products, answers questions about prices and stock, and provides recommendations. Professional, patient, and focused on customer satisfaction."
        static_dialogue = [
            "Welcome to our store! How can I help?",
            "Looking for anything in particular?",
            "Let me help you find what you need."
        ]
        super().__init__(name, row, col, color, interaction_range, static_dialogue, personality)
        self.inventory = inventory
        self.customer_cart = []
        self.conversation_count = 0
    
    def add_to_cart(self, product_id):
        """Add a product to the customer's cart"""
        self.customer_cart.append(product_id)
    
    def clear_cart(self):
        """Clear the customer's cart"""
        self.customer_cart = []
    
    def get_cart_total(self):
        """Calculate the total price of items in cart"""
        total = 0
        for product_id in self.customer_cart:
            product = self.inventory.get_product_by_id(product_id)
            if product:
                total += product["price"]
        return total
