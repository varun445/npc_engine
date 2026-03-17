# Product Inventory System for Shop Assistant

SEARCH_RESULTS_PREFIX = "Search Results: "

PRODUCTS = {
    "dairy": [
        {"id": "milk_1l", "name": "Milk (1L)", "price": 2.50, "stock": 15},
        {"id": "milk_2l", "name": "Milk (2L)", "price": 4.50, "stock": 12},
        {"id": "cheese", "name": "Cheddar Cheese", "price": 6.00, "stock": 8},
        {"id": "yogurt", "name": "Plain Yogurt", "price": 3.50, "stock": 20},
    ],
    "bakery": [
        {"id": "bread_white", "name": "White Bread", "price": 2.00, "stock": 10},
        {"id": "bread_wheat", "name": "Wheat Bread", "price": 2.50, "stock": 8},
        {"id": "croissant", "name": "Croissant", "price": 3.00, "stock": 12},
        {"id": "donut", "name": "Donut (Glazed)", "price": 1.50, "stock": 16},
    ],
    "fruits": [
        {"id": "apple", "name": "Apple", "price": 1.00, "stock": 30},
        {"id": "banana", "name": "Banana", "price": 0.60, "stock": 25},
        {"id": "orange", "name": "Orange", "price": 1.50, "stock": 18},
        {"id": "grape", "name": "Grapes (1lb)", "price": 3.00, "stock": 12},
    ],
    "vegetables": [
        {"id": "carrot", "name": "Carrot", "price": 0.80, "stock": 20},
        {"id": "broccoli", "name": "Broccoli", "price": 2.50, "stock": 10},
        {"id": "lettuce", "name": "Lettuce", "price": 2.00, "stock": 14},
        {"id": "tomato", "name": "Tomato", "price": 1.50, "stock": 18},
    ],
    "beverages": [
        {"id": "water", "name": "Bottled Water", "price": 1.50, "stock": 40},
        {"id": "juice_orange", "name": "Orange Juice", "price": 3.50, "stock": 15},
        {"id": "juice_apple", "name": "Apple Juice", "price": 3.50, "stock": 12},
        {"id": "soda", "name": "Cola (2L)", "price": 2.50, "stock": 20},
        {"id": "coffee", "name": "Coffee Beans", "price": 8.00, "stock": 8},
    ],
    "snacks": [
        {"id": "chips", "name": "Potato Chips", "price": 2.50, "stock": 25},
        {"id": "cookies", "name": "Chocolate Cookies", "price": 3.00, "stock": 18},
        {"id": "nuts", "name": "Mixed Nuts", "price": 5.00, "stock": 10},
        {"id": "granola", "name": "Granola Bar", "price": 1.50, "stock": 30},
        {"id": "popcorn", "name": "Popcorn", "price": 2.00, "stock": 22},
    ]
}

AISLE_LOCATIONS = {
    "dairy": 1,
    "bakery": 2,
    "fruits": 3,
    "vegetables": 3,
    "beverages": 4,
    "snacks": 5
}

class Inventory:
    def __init__(self):
        self.products = PRODUCTS
        self.aisles = AISLE_LOCATIONS

    def find_product(self, product_name):
        """Find a product by name across all categories"""
        product_name_lower = product_name.lower()
        for category, products in self.products.items():
            for product in products:
                if product_name_lower in product["name"].lower() or product_name_lower == product["id"]:
                    return {**product, "category": category, "aisle": self.aisles[category]}
        return None
    
    def get_category_products(self, category):
        """Get all products in a category"""
        if category.lower() in self.products:
            return self.products[category.lower()]
        return []
    
    def is_in_stock(self, product_id):
        """Check if a product is in stock"""
        for products in self.products.values():
            for product in products:
                if product["id"] == product_id:
                    return product["stock"] > 0
        return False
    
    def get_product_by_id(self, product_id):
        """Get product details by ID"""
        for products in self.products.values():
            for product in products:
                if product["id"] == product_id:
                    return product
        return None

    def find_products_by_terms(self, search_terms):
        """Return a de-duplicated list of in-stock product dicts that match any of the given terms.

        Matching is the same substring/id logic used by ``search_inventory``.
        This is a pure-Python, zero-LLM call that guarantees no hallucination.

        Args:
            search_terms: list of strings (e.g. ["milk", "cheese"])

        Returns:
            List of product dicts from the database (each has ``id``, ``name``,
            ``price``, ``stock`` keys).
        """
        matched = []
        seen_ids = set()
        for term in search_terms:
            term_lower = term.lower().strip()
            for products in self.products.values():
                for product in products:
                    if product["id"] in seen_ids:
                        continue
                    if (
                        term_lower in product["name"].lower()
                        or term_lower in product["id"].lower()
                    ) and product["stock"] > 0:
                        matched.append(product)
                        seen_ids.add(product["id"])
        return matched

    def search_inventory(self, search_terms):
        """Search inventory for a list of search terms.

        Checks each term against product names and IDs and returns a plain-text
        summary that can be fed back to the LLM as a tool observation.

        Args:
            search_terms: list of strings to look up (e.g. ["sugar", "eggs"])

        Returns:
            A string such as:
            "Search Results: sugar: Sugar (Aisle 3, $1.00, 20 in stock);
             eggs: not found in store inventory"
        """
        results = []
        for term in search_terms:
            term_lower = term.lower().strip()
            matches = []
            for category, products in self.products.items():
                aisle_num = self.aisles[category]
                for product in products:
                    if (
                        term_lower in product["name"].lower()
                        or term_lower in product["id"].lower()
                    ):
                        if product["stock"] > 0:
                            matches.append(
                                f"{product['name']} (Aisle {aisle_num},"
                                f" ${product['price']:.2f}, {product['stock']} in stock)"
                            )
                        else:
                            matches.append(f"{product['name']} (out of stock)")
            if matches:
                results.append(f"{term}: {', '.join(matches)}")
            else:
                results.append(f"{term}: not found in store inventory")
        return SEARCH_RESULTS_PREFIX + "; ".join(results)
