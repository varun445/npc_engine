import argparse
import sys

from models.inventory import Inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick semantic-only check: Ollama embedding + inventory vector DB + top-k matching."
    )
    parser.add_argument(
        "--query",
        default="I need cola and some mixed nuts",
        help="Query to embed and match against inventory vectors.",
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Ollama embedding model name.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of semantic matches to print.",
    )
    args = parser.parse_args()

    inventory = Inventory()

    print(f"[1/3] Checking embedding model '{args.model}'...")
    probe = inventory._ollama_embed("embedding probe", args.model)
    if not probe:
        print(
            "❌ Embedding model check failed. Ensure Ollama is running and model is available "
            f"(e.g., `ollama pull {args.model}`)."
        )
        return 1
    print(f"✅ Embedding model is working (vector dim: {len(probe)})")

    print("[2/3] Building inventory vector DB...")
    ok = inventory.ensure_semantic_vector_db(model=args.model)
    count = inventory.semantic_vector_count(model=args.model)
    if not ok or count == 0:
        print("❌ Vector DB setup failed: no inventory embeddings were stored.")
        return 1
    print(f"✅ Vector DB ready with {count} in-stock inventory vectors")

    print("[3/3] Running semantic retrieval...")
    terms = inventory.extract_semantic_query_terms(args.query)
    matches = inventory.semantic_search(
        query=args.query,
        top_k=args.top_k,
        model=args.model,
        query_terms=terms,
    )

    print(f"Query: {args.query}")
    print(f"Extracted terms: {terms}")
    if not matches:
        print("❌ No semantic matches found.")
        return 1

    print("Top matches:")
    for idx, m in enumerate(matches, start=1):
        product = m["product"]
        print(
            f"  {idx}. {product['name']} | aisle={m['aisle']} | "
            f"score={m['score']:.4f} | id={product['id']}"
        )
    print("✅ Semantic pipeline check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
