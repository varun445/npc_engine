"""
evaluate.py — Automated Evaluation Framework for the NPC Engine LLM Pipeline
==============================================================================
Runs 50 natural-language queries through the shop-assistant pipeline and
measures four research metrics:

    1. Task Success Rate  — agent correctly handles each query type
                           (moves to valid aisles when items are found;
                            returns action "none" when no items match).
    2. Hallucination Rate — agent references an invalid aisle number or
                            an aisle not listed in the store's AISLE_LOCATIONS
                            map.
    3. Latency           — wall-clock time (seconds) for the full end-to-end
                           pipeline per query (term extraction + inventory
                           search + response generation).
    4. JSON Adherence    — the model reliably outputs a parseable JSON object
                           containing exactly the keys: "dialogue", "action",
                           and "target_aisles".

Usage
-----
    # Run against a live Ollama server (default):
    python evaluate.py

    # Dry-run with a mock LLM (no Ollama needed — for CI / unit testing):
    python evaluate.py --mock

    # Specify a custom dataset path or output directory:
    python evaluate.py --queries evaluation/queries.json --output evaluation/results

    # Limit to the first N queries (useful for quick smoke-tests):
    python evaluate.py --limit 10

Output
------
  evaluation/results/results.csv  — per-query detail rows
  evaluation/results/summary.txt  — human-readable metric summary
  Console                         — live progress + final table
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — allow running from the repository root without installing the
# package so that  `python evaluate.py`  works out of the box.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.inventory import Inventory, AISLE_LOCATIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_AISLES = set(AISLE_LOCATIONS.values())          # derived from AISLE_LOCATIONS
AISLE_REFERENCE_RE = re.compile(r"[Aa]isle\s+(\d+)")  # matches "Aisle N" etc.
REQUIRED_JSON_KEYS = {"dialogue", "action", "target_aisles"}
MAX_DIALOGUE_SNIPPET_LENGTH = 120
ASSISTANT_NAME = "Alex"

DEFAULT_QUERIES_PATH = os.path.join(REPO_ROOT, "evaluation", "queries.json")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "evaluation", "results")


# ---------------------------------------------------------------------------
# Mock LLM helpers (used when --mock flag is set so CI can run without Ollama)
# ---------------------------------------------------------------------------

def _mock_extract_product_terms(query: str) -> list[str]:
    """Simple keyword extraction that doesn't need an LLM."""
    inventory = Inventory()
    all_names = []
    for products in inventory.products.values():
        for p in products:
            all_names.append(p["name"].lower())
            all_names.append(p["id"].lower())
    found = []
    query_lower = query.lower()
    # Prefer longer names first to avoid sub-string false-matches
    for name in sorted(all_names, key=len, reverse=True):
        if name in query_lower and name not in found:
            found.append(name)
    return found


def _mock_generate_response(
    assistant_name: str,
    customer_query: str,
    inventory_info: str,
    memory: list,
    tool_observations: list | None,
) -> dict:
    """Return a minimal valid JSON response without calling the LLM."""
    if not tool_observations:
        return {"dialogue": "How can I help you?", "action": "none", "target_aisles": []}

    from models.inventory import SEARCH_RESULTS_PREFIX

    found_aisles = []
    for obs in tool_observations:
        body = (
            obs[len(SEARCH_RESULTS_PREFIX):]
            if obs.startswith(SEARCH_RESULTS_PREFIX)
            else obs
        )
        for entry in body.split("; "):
            if ": " not in entry:
                continue
            _, value = entry.split(": ", 1)
            value = value.strip()
            if "not found in store inventory" in value:
                continue
            m = re.search(r"Aisle (\d+)", value)
            if m:
                found_aisles.append(int(m.group(1)))

    found_aisles = sorted(set(found_aisles))
    if found_aisles:
        return {
            "dialogue": f"I found those items. They are in Aisle(s) {', '.join(str(a) for a in found_aisles)}.",
            "action": "move",
            "target_aisles": found_aisles,
        }
    return {
        "dialogue": "Sorry, we don't carry those items.",
        "action": "none",
        "target_aisles": [],
    }


# ---------------------------------------------------------------------------
# Hallucination detection helpers
# ---------------------------------------------------------------------------

def _find_hallucinated_aisles(response: dict) -> list[int]:
    """Return a list of aisle numbers referenced that are not valid store aisles.

    Checks both ``target_aisles`` (structured field) and any "Aisle N"
    mentions inside the ``dialogue`` string.
    """
    invalid: list[int] = []

    # 1. Structured field
    for aisle in response.get("target_aisles", []):
        try:
            if int(aisle) not in VALID_AISLES:
                invalid.append(int(aisle))
        except (TypeError, ValueError):
            pass

    # 2. Inline dialogue mentions
    for m in AISLE_REFERENCE_RE.finditer(response.get("dialogue", "")):
        aisle_num = int(m.group(1))
        if aisle_num not in VALID_AISLES and aisle_num not in invalid:
            invalid.append(aisle_num)

    return invalid


# ---------------------------------------------------------------------------
# Task success helpers
# ---------------------------------------------------------------------------

def _is_task_success(response: dict, expected_items: list[str], inventory: Inventory) -> bool:
    """Determine whether the agent handled the query correctly.

    Rules:
    - If expected_items is non-empty (items should be found):
        The agent must have set action to "move" and populated target_aisles
        with at least one valid aisle.
    - If expected_items is empty (conversational / invalid query):
        The agent must have set action to "none".
    """
    action = response.get("action", "none")
    target_aisles = response.get("target_aisles", [])

    # Determine which of the expected items are actually in stock
    stocked_terms = []
    if expected_items:
        for term in expected_items:
            matches = inventory.find_products_by_terms([term])
            if matches:
                stocked_terms.append(term)

    if stocked_terms:
        # At least one expected in-stock item → agent should navigate
        return action == "move" and len(target_aisles) > 0
    else:
        # No in-stock items expected → agent should stay put
        return action == "none"


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    queries: list[dict],
    mock: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Run all queries through the pipeline and return a list of result dicts."""

    inventory = Inventory()
    inventory_summary = _build_inventory_summary(inventory)

    if mock:
        extract_fn = _mock_extract_product_terms
        generate_fn = _mock_generate_response
    else:
        from engine.llm_client import extract_product_terms, generate_shop_assistant_response
        extract_fn = extract_product_terms
        generate_fn = generate_shop_assistant_response

    results: list[dict] = []

    for idx, item in enumerate(queries, start=1):
        qid = item["id"]
        query = item["query"]
        query_type = item.get("query_type", "unknown")
        expected_items = item.get("expected_items", [])

        if verbose:
            print(f"  [{idx:02d}/{len(queries)}] Q{qid} ({query_type:14s}): {query[:60]!r}")

        # ── Pipeline timing start ─────────────────────────────────────────
        t_start = time.perf_counter()

        # Step 1 — extract product terms
        product_terms = extract_fn(query)

        # Step 2 — deterministic inventory search
        tool_observations: list[str] = []
        if product_terms:
            observation = inventory.search_inventory(product_terms)
            tool_observations.append(observation)

        # Step 3 — generate final response
        raw_response = generate_fn(
            ASSISTANT_NAME,
            query,
            inventory_summary,
            [],          # empty memory for independent evaluation
            tool_observations,
        )

        t_end = time.perf_counter()
        latency = round(t_end - t_start, 3)
        # ── Pipeline timing end ───────────────────────────────────────────

        # ── Metric 1: JSON Adherence ──────────────────────────────────────
        json_adherent = isinstance(raw_response, dict) and REQUIRED_JSON_KEYS.issubset(
            raw_response.keys()
        )

        # ── Metric 2: Task Success ────────────────────────────────────────
        task_success = _is_task_success(raw_response, expected_items, inventory)

        # ── Metric 3: Hallucination ───────────────────────────────────────
        hallucinated_aisles = _find_hallucinated_aisles(raw_response)
        hallucinated = len(hallucinated_aisles) > 0

        # ── Collect result ────────────────────────────────────────────────
        result_row = {
            "id": qid,
            "query_type": query_type,
            "query": query,
            "expected_items": "; ".join(expected_items),
            "product_terms_extracted": "; ".join(product_terms),
            "action": raw_response.get("action", ""),
            "target_aisles": str(raw_response.get("target_aisles", [])),
            "dialogue_snippet": raw_response.get("dialogue", "")[:MAX_DIALOGUE_SNIPPET_LENGTH],
            "json_adherent": json_adherent,
            "task_success": task_success,
            "hallucinated": hallucinated,
            "hallucinated_aisles": str(hallucinated_aisles),
            "latency_s": latency,
        }
        results.append(result_row)

        if verbose:
            status_icon = "✓" if task_success else "✗"
            hall_icon = "H" if hallucinated else " "
            json_icon = "J" if json_adherent else "!"
            print(
                f"         [{status_icon}][{hall_icon}][{json_icon}]"
                f"  action={result_row['action']:5s}"
                f"  aisles={result_row['target_aisles']:10s}"
                f"  latency={latency:.2f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Inventory summary helper (mirrors engine/input_handler._build_inventory_summary)
# ---------------------------------------------------------------------------

def _build_inventory_summary(inventory: Inventory) -> str:
    lines = []
    for category, products in inventory.products.items():
        aisle_num = inventory.aisles[category]
        for product in products:
            if product["stock"] > 0:
                lines.append(
                    f"{product['name']} (${product['price']:.2f})"
                    f" — category: {category}, Aisle {aisle_num}"
                )
    return "Store inventory:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _compute_summary(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    successes = sum(1 for r in results if r["task_success"])
    hallucinations = sum(1 for r in results if r["hallucinated"])
    json_ok = sum(1 for r in results if r["json_adherent"])
    latencies = [r["latency_s"] for r in results]

    by_type: dict[str, dict] = {}
    for r in results:
        qt = r["query_type"]
        if qt not in by_type:
            by_type[qt] = {"total": 0, "success": 0, "hallucinated": 0}
        by_type[qt]["total"] += 1
        if r["task_success"]:
            by_type[qt]["success"] += 1
        if r["hallucinated"]:
            by_type[qt]["hallucinated"] += 1

    return {
        "total_queries": n,
        "task_success_rate": round(successes / n * 100, 1),
        "hallucination_rate": round(hallucinations / n * 100, 1),
        "json_adherence_rate": round(json_ok / n * 100, 1),
        "latency_mean_s": round(sum(latencies) / n, 3),
        "latency_min_s": round(min(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
        "latency_p50_s": round(sorted(latencies)[n // 2], 3),
        "by_query_type": by_type,
    }


def _print_summary(summary: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  EVALUATION SUMMARY")
    print(sep)
    print(f"  Total queries        : {summary['total_queries']}")
    print(f"  Task Success Rate    : {summary['task_success_rate']:.1f}%")
    print(f"  Hallucination Rate   : {summary['hallucination_rate']:.1f}%")
    print(f"  JSON Adherence Rate  : {summary['json_adherence_rate']:.1f}%")
    print(f"  Latency (mean)       : {summary['latency_mean_s']:.3f}s")
    print(f"  Latency (min/p50/max): {summary['latency_min_s']:.3f}s"
          f" / {summary['latency_p50_s']:.3f}s"
          f" / {summary['latency_max_s']:.3f}s")
    print()
    print("  By query type:")
    header = f"  {'Type':<16} {'Queries':>7} {'Success':>9} {'Hallucinated':>14}"
    print(header)
    print("  " + "─" * 50)
    for qt, stats in summary["by_query_type"].items():
        success_pct = stats["success"] / stats["total"] * 100 if stats["total"] else 0
        hall_pct = stats["hallucinated"] / stats["total"] * 100 if stats["total"] else 0
        print(
            f"  {qt:<16} {stats['total']:>7}"
            f" {success_pct:>8.1f}%"
            f" {hall_pct:>13.1f}%"
        )
    print(sep)


def _save_csv(results: list[dict], output_dir: str, timestamp: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    return csv_path


def _save_summary(summary: dict, output_dir: str, timestamp: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
    lines = [
        "NPC Engine — LLM Pipeline Evaluation Summary",
        f"Generated : {datetime.now().isoformat()}",
        "",
        f"Total queries        : {summary['total_queries']}",
        f"Task Success Rate    : {summary['task_success_rate']:.1f}%",
        f"Hallucination Rate   : {summary['hallucination_rate']:.1f}%",
        f"JSON Adherence Rate  : {summary['json_adherence_rate']:.1f}%",
        f"Latency mean (s)     : {summary['latency_mean_s']:.3f}",
        f"Latency min  (s)     : {summary['latency_min_s']:.3f}",
        f"Latency p50  (s)     : {summary['latency_p50_s']:.3f}",
        f"Latency max  (s)     : {summary['latency_max_s']:.3f}",
        "",
        "By query type:",
        f"  {'Type':<16} {'Queries':>7} {'Success':>9} {'Hallucinated':>14}",
        "  " + "─" * 50,
    ]
    for qt, stats in summary["by_query_type"].items():
        success_pct = stats["success"] / stats["total"] * 100 if stats["total"] else 0
        hall_pct = stats["hallucinated"] / stats["total"] * 100 if stats["total"] else 0
        lines.append(
            f"  {qt:<16} {stats['total']:>7}"
            f" {success_pct:>8.1f}%"
            f" {hall_pct:>13.1f}%"
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the NPC Engine LLM pipeline on a query dataset."
    )
    parser.add_argument(
        "--queries",
        default=DEFAULT_QUERIES_PATH,
        help="Path to the JSON query dataset (default: evaluation/queries.json)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write CSV and summary files (default: evaluation/results)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a mock LLM instead of calling Ollama (useful for CI / offline testing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N queries (useful for quick smoke-tests)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-query progress output",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Load queries
    if not os.path.isfile(args.queries):
        print(f"[ERROR] Query file not found: {args.queries}", file=sys.stderr)
        return 1
    with open(args.queries, encoding="utf-8") as f:
        queries: list[dict] = json.load(f)

    if args.limit:
        queries = queries[: args.limit]

    mode_label = "MOCK (no Ollama)" if args.mock else "LIVE (Ollama/Mistral)"
    print(f"\nNPC Engine — LLM Pipeline Evaluation")
    print(f"Mode    : {mode_label}")
    print(f"Queries : {len(queries)}")
    print(f"Output  : {args.output}")
    print()

    # Run evaluation
    results = run_evaluation(queries, mock=args.mock, verbose=not args.quiet)

    # Compute and display summary
    summary = _compute_summary(results)
    _print_summary(summary)

    # Persist results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = _save_csv(results, args.output, timestamp)
    summary_path = _save_summary(summary, args.output, timestamp)
    print(f"\n  Results CSV  → {csv_path}")
    print(f"  Summary file → {summary_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
