# Product Inventory System for Shop Assistant

import math
import os
import re
from urllib.parse import urlparse

import requests

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
    _SEMANTIC_STOPWORDS = {
        "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "and", "or",
        "to", "for", "of", "in", "on", "with", "some", "any", "something", "want",
        "need", "find", "get", "give", "show", "where", "is", "are", "can", "please",
    }
    _MAX_SCORE_WEIGHT = 0.65
    _AVG_SCORE_WEIGHT = 0.35

    def __init__(self):
        self.products = PRODUCTS
        self.aisles = AISLE_LOCATIONS
        self._semantic_embedding_cache = {}
        self._embedding_endpoint = os.getenv(
            "OLLAMA_EMBEDDING_URL",
            os.getenv("OLLAMA_EMBEDDINGS_URL", "http://localhost:11434/api/embeddings"),
        )
        self._embedding_endpoint_fallback = os.getenv(
            "OLLAMA_EMBEDDING_FALLBACK_URL",
            os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
        )
        self._embedding_config_warning = ""
        try:
            self._embedding_timeout_s = float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "30"))
        except ValueError:
            self._embedding_timeout_s = 30.0
            self._embedding_config_warning = (
                "Invalid OLLAMA_EMBEDDING_TIMEOUT value; using default timeout of 30s."
            )
        self._last_embedding_error = ""
        self._semantic_vector_db = {}

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

    def _format_match(self, product, aisle_num):
        return (
            f"{product['name']} (Aisle {aisle_num},"
            f" ${product['price']:.2f}, {product['stock']} in stock)"
        )

    def _semantic_text_for_product(self, product, category):
        return (
            f"name: {product['name']}. id: {product['id']}. "
            f"category: {category}. aisle: {self.aisles[category]}"
        )

    def extract_semantic_query_terms(self, query, max_terms=8):
        """Extract important terms from a query for semantic embedding."""
        query_lower = (query or "").lower().strip()
        if not query_lower:
            return []

        terms = []

        # Keep full product phrases if directly present in the query.
        for products in self.products.values():
            for product in products:
                name = product["name"].lower()
                clean_name = re.sub(r"[^a-z0-9\s]", " ", name)
                if clean_name and clean_name in query_lower:
                    terms.append(clean_name.strip())

                for token in re.findall(r"[a-z0-9]+", product["id"].lower()):
                    if token and token in query_lower:
                        terms.append(token)

        tokens = re.findall(r"[a-z0-9]+", query_lower)
        for token in tokens:
            if len(token) <= 2 or token in self._SEMANTIC_STOPWORDS:
                continue
            terms.append(token)

        deduped = []
        seen = set()
        for term in terms:
            key = term.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
            if len(deduped) >= max_terms:
                break
        return deduped

    def _ollama_embed(self, text, model):
        self._last_embedding_error = ""

        def _candidate_payloads(endpoint_url):
            path = (urlparse(endpoint_url).path or "").lower()
            if path.endswith("/embed"):
                return [
                    {"model": model, "input": text},
                    {"model": model, "prompt": text},
                ]
            return [
                {"model": model, "prompt": text},
                {"model": model, "input": text},
            ]

        def _is_numeric_vector(value):
            def _is_finite_number(x):
                if not isinstance(x, (int, float)):
                    return False
                numeric = float(x)
                return not math.isnan(numeric) and not math.isinf(numeric)

            return (
                isinstance(value, list)
                and len(value) > 0
                and all(_is_finite_number(x) for x in value)
            )

        def _extract_embedding(data):
            embedding = data.get("embedding")
            if _is_numeric_vector(embedding):
                return embedding

            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                if isinstance(embeddings[0], list):
                    if _is_numeric_vector(embeddings[0]):
                        return embeddings[0]
                if _is_numeric_vector(embeddings):
                    return embeddings

            rows = data.get("data")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                emb = rows[0].get("embedding")
                if _is_numeric_vector(emb):
                    return emb
            return None

        for endpoint in (self._embedding_endpoint, self._embedding_endpoint_fallback):
            if not endpoint:
                continue
            for payload in _candidate_payloads(endpoint):
                try:
                    response = requests.post(endpoint, json=payload, timeout=self._embedding_timeout_s)
                    response.raise_for_status()
                    data = response.json()
                    embedding = _extract_embedding(data)
                    if embedding:
                        return embedding
                    self._last_embedding_error = (
                        f"Unexpected response format from {endpoint}: keys={list(data.keys())}"
                    )
                except (requests.RequestException, ValueError, TypeError) as exc:
                    self._last_embedding_error = f"{endpoint} ({type(exc).__name__}): {exc}"
                    continue
        return None

    def get_last_embedding_error(self):
        """Return last embedding-request error detail, if any."""
        if self._embedding_config_warning and self._last_embedding_error:
            return f"{self._embedding_config_warning} {self._last_embedding_error}"
        return self._last_embedding_error or self._embedding_config_warning

    def get_embedding_endpoints(self):
        """Return configured primary/fallback embedding endpoints."""
        return (self._embedding_endpoint, self._embedding_endpoint_fallback)

    def probe_embedding_model(self, model="nomic-embed-text", text="embedding probe"):
        """Return one embedding vector for a probe text, or None if unavailable."""
        return self._ollama_embed(text, model)

    def _cosine_similarity(self, vec_a, vec_b):
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def ensure_semantic_vector_db(self, model="nomic-embed-text"):
        """Build and cache inventory embeddings for semantic lookup."""
        if model in self._semantic_vector_db and self._semantic_vector_db[model]:
            return True

        vector_rows = []
        for category, products in self.products.items():
            aisle_num = self.aisles[category]
            for product in products:
                if product["stock"] <= 0:
                    continue
                cache_key = (model, product["id"])
                if cache_key not in self._semantic_embedding_cache:
                    embedded = self._ollama_embed(
                        self._semantic_text_for_product(product, category),
                        model,
                    )
                    if embedded:
                        self._semantic_embedding_cache[cache_key] = embedded
                embedding = self._semantic_embedding_cache.get(cache_key)
                if not embedding:
                    continue
                vector_rows.append(
                    {
                        "product": product,
                        "category": category,
                        "aisle": aisle_num,
                        "embedding": embedding,
                    }
                )
        self._semantic_vector_db[model] = vector_rows
        return len(vector_rows) > 0

    def semantic_vector_count(self, model="nomic-embed-text"):
        """Return the number of embedded inventory items stored for *model*."""
        return len(self._semantic_vector_db.get(model, []))

    def semantic_search(self, query, query_terms=None, top_k=5, model="nomic-embed-text", min_score=0.15):
        """Return top semantic matches from inventory for a free-form query.

        Args:
            query: Raw customer query text to embed.
            top_k: Maximum number of matches to return.
            model: Ollama embedding model name.
            min_score: Minimum cosine-similarity score to keep a match.
            query_terms: Optional pre-extracted terms to embed instead of raw query.
        """
        query = (query or "").strip()
        if not query:
            return []
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        if top_k <= 0:
            return []

        terms_to_embed = query_terms if query_terms else self.extract_semantic_query_terms(query)
        if not terms_to_embed:
            terms_to_embed = [query]

        query_embeddings = []
        for term in terms_to_embed:
            term_embedding = self._ollama_embed(term, model)
            if term_embedding:
                query_embeddings.append(term_embedding)

        if not query_embeddings:
            return []

        if not self.ensure_semantic_vector_db(model=model):
            return []

        scored = []
        for row in self._semantic_vector_db.get(model, []):
            per_term_scores = [
                self._cosine_similarity(query_vec, row["embedding"])
                for query_vec in query_embeddings
            ]
            if not per_term_scores:
                continue
            max_score = max(per_term_scores)
            avg_score = sum(per_term_scores) / len(per_term_scores)
            # Blend peak term hit and overall term alignment so exact product words
            # score strongly while still rewarding multi-term query consistency.
            score = (self._MAX_SCORE_WEIGHT * max_score) + (self._AVG_SCORE_WEIGHT * avg_score)
            if score >= min_score:
                scored.append(
                    {
                        "product": row["product"],
                        "category": row["category"],
                        "aisle": row["aisle"],
                        "score": score,
                    }
                )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def semantic_search_inventory(
        self, query, query_terms=None, top_k=5, model="nomic-embed-text", min_score=0.15
    ):
        """Search inventory semantically and format as a tool observation string."""
        matches = self.semantic_search(
            query=query,
            query_terms=query_terms,
            top_k=top_k,
            model=model,
            min_score=min_score,
        )
        if not matches:
            label = ", ".join(query_terms) if query_terms else query
            return SEARCH_RESULTS_PREFIX + f"{label}: not found in store inventory"

        formatted_matches = [
            self._format_match(item["product"], item["aisle"])
            for item in matches
        ]
        label = ", ".join(query_terms) if query_terms else query
        return SEARCH_RESULTS_PREFIX + f"{label}: {', '.join(formatted_matches)}"

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
                            matches.append(self._format_match(product, aisle_num))
                        else:
                            matches.append(f"{product['name']} (out of stock)")
            if matches:
                results.append(f"{term}: {', '.join(matches)}")
            else:
                results.append(f"{term}: not found in store inventory")
        return SEARCH_RESULTS_PREFIX + "; ".join(results)
