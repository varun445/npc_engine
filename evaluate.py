"""
evaluate.py — Automated Evaluation Framework for the NPC Engine LLM Pipeline
==============================================================================
Runs natural-language queries through the shop-assistant pipeline and measures
four research metrics:

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

Evaluation Modes (--mode)
--------------------------
    presearch  (default)
        Full pipeline: term extraction → inventory search (search_inventory
        tool hook) → response generation.  This is the grounded, anti-
        hallucination architecture described in the project thesis.

    semantic
        Semantic retrieval pipeline: embed the full user query and retrieve
        top-k nearest in-stock inventory items, then generate the response
        from those grounded tool observations.

    llm_only
        The query is sent directly to the response-generation step with no
        deterministic inventory pre-search, but with full inventory context in
        the prompt.

    all
        Runs every query through all modes (presearch, semantic, llm_only)
        and saves three labelled rows per query for side-by-side comparison.

Checkpointing / Resume Support
--------------------------------
Results for completed queries are written immediately to a rolling file
``<output_dir>/results.csv``.  If the run is interrupted, re-launching with
the same ``--output`` directory will automatically detect how many rows are
already present and continue from the next query index.

Use ``--start_from N`` to explicitly override the resume point (0-based query
index).

Usage
-----
    # Run against a live Ollama server (default mode: presearch):
    python evaluate.py

    # Dry-run with a mock LLM (no Ollama needed — for CI / unit testing):
    python evaluate.py --mock

    # Experimental comparison across all pipelines:
    python evaluate.py --mode all

    # Baseline only (vanilla LLM, no search):
    python evaluate.py --mode llm_only

    # Specify a custom dataset path or output directory:
    python evaluate.py --queries evaluation/queries.json --output evaluation/results

    # Limit to the first N queries (useful for quick smoke-tests):
    python evaluate.py --limit 10

    # Explicitly resume from a specific query index (0-based):
    python evaluate.py --start_from 20

Output
------
  <output_dir>/results.csv              — rolling per-query detail rows
                                          (appended on resume; never overwritten)
  <output_dir>/results_<timestamp>.csv  — snapshot of this run's rows
  <output_dir>/summary_<timestamp>.txt  — human-readable metric summary
  Console                               — live progress + final table

Design Notes (for thesis appendix)
------------------------------------
The evaluation script implements *incremental, resumable benchmarking*:

  1. After each query (or per-query mode bundle in comparison modes) the result row(s)
     are appended to ``results.csv`` so that a crash or keyboard interrupt only
     loses the current in-flight query.

  2. On startup, ``_count_existing_results`` reads the row count from the
     rolling CSV.  For a single-mode run this equals the number of completed
      queries; for an ``all``-mode run it equals ``completed_queries × 3``, so
      the resume index is ``row_count // 3``.

  3. The ``--mode all`` comparison is implemented by running each query for
     each comparison mode through ``_run_single_query`` and saving all rows before
     moving to the next query.  This keeps the per-query checkpoint atomic:
     either all rows for a query are saved or none are (the checkpoint is
     written after all mode calls complete).

  4. The ``mode`` column in the CSV makes it straightforward to load the data
     in a notebook and split rows by mode for metric comparison:
         df_pre  = df[df['mode'] == 'presearch']
         df_llm  = df[df['mode'] == 'llm_only']
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

# Rolling results file used for checkpointing / resume support.
# This filename is fixed (no timestamp) so re-runs can detect existing rows.
ROLLING_RESULTS_FILENAME = "results.csv"


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
    include_inventory_in_no_search: bool = False,
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


def _mock_semantic_search(query: str, inventory: Inventory, top_k: int = 5) -> str:
    """Deterministic semantic-like retrieval for --mock runs."""
    query_lower = (query or "").lower()
    semantic_terms = []

    def _terms_from_category(category_name: str) -> list[str]:
        return [p["id"] for p in inventory.products.get(category_name, []) if p.get("stock", 0) > 0]

    if "healthy" in query_lower or "health" in query_lower:
        semantic_terms.extend(_terms_from_category("fruits"))
        semantic_terms.extend(_terms_from_category("vegetables"))
        semantic_terms.extend(
            [
                p["id"]
                for products in inventory.products.values()
                for p in products
                if p.get("stock", 0) > 0
                and ("yogurt" in p["id"].lower() or "water" in p["id"].lower())
            ]
        )
    if "drink" in query_lower or "thirst" in query_lower or "beverage" in query_lower:
        semantic_terms.extend(_terms_from_category("beverages"))
    if "snack" in query_lower:
        semantic_terms.extend(_terms_from_category("snacks"))

    keyword_terms = _mock_extract_product_terms(query)
    semantic_terms.extend(keyword_terms)

    # Preserve order while deduplicating.
    deduped_terms = []
    seen = set()
    for term in semantic_terms:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped_terms.append(term)

    if not deduped_terms:
        deduped_terms = query_lower.split()

    # Reuse deterministic exact matcher but cap to top_k output products.
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    if top_k <= 0:
        return f"Search Results: {query}: not found in store inventory"

    matched = inventory.find_products_by_terms(deduped_terms)[:top_k]
    if not matched:
        return f"Search Results: {query}: not found in store inventory"

    matches_with_aisle = []
    for product in matched:
        category = next(
            (
                cat
                for cat, products in inventory.products.items()
                if any(p["id"] == product["id"] for p in products)
            ),
            None,
        )
        aisle_num = inventory.aisles.get(category) if category else None
        if aisle_num is None:
            continue
        matches_with_aisle.append(
            f"{product['name']} (Aisle {aisle_num}, ${product['price']:.2f}, {product['stock']} in stock)"
        )

    if not matches_with_aisle:
        return f"Search Results: {query}: not found in store inventory"
    return f"Search Results: {query}: {', '.join(matches_with_aisle)}"


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
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _count_existing_results(csv_path: str) -> int:
    """Return the number of *data* rows already present in *csv_path*.

    Returns 0 if the file does not exist or contains only a header.
    """
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _append_csv(rows: list[dict], csv_path: str) -> None:
    """Append *rows* to the rolling CSV at *csv_path*.

    If the file does not yet exist it is created with a header row first.
    All rows in *rows* must share the same set of keys.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def _run_single_query(
    item: dict,
    idx: int,
    total: int,
    inventory: "Inventory",
    inventory_summary: str,
    extract_fn,
    semantic_search_fn,
    generate_fn,
    mode: str,
    verbose: bool,
) -> dict:
    """Execute the pipeline for a single query in the specified *mode*.

    Parameters
    ----------
    item:
        A single query dict from the JSON dataset.
    idx:
        1-based position of this query in the current run (for progress display).
    total:
        Total number of queries in the current run.
    inventory:
        Shared :class:`~models.inventory.Inventory` instance.
    inventory_summary:
        Pre-built inventory summary string passed to the response generator.
    extract_fn:
        Callable that extracts product terms from a query string.
    generate_fn:
        Callable that generates the shop-assistant JSON response.
    mode:
        ``"presearch"`` — full pipeline (extraction + search + generation).
        ``"semantic"``  — semantic retrieval (embedding similarity + generation).
        ``"llm_only"``  — generation without deterministic lookup; includes
        full inventory context in the prompt.
    verbose:
        Print per-query progress lines to stdout.

    Returns
    -------
    dict
        A result row ready to be appended to the results CSV.
    """
    qid = item["id"]
    query = item["query"]
    query_type = item.get("query_type", "unknown")
    expected_items = item.get("expected_items", [])

    if verbose:
        print(
            f"  [{idx:02d}/{total}] Q{qid} ({query_type:14s})"
            f" [{mode:9s}]: {query[:60]!r}"
        )

    # ── Pipeline timing start ─────────────────────────────────────────────
    t_start = time.perf_counter()

    if mode == "presearch":
        # Step 1 — extract product terms from the query
        product_terms = extract_fn(query)

        # Step 2 — deterministic inventory search (the "pre-search hook")
        tool_observations: list[str] = []
        if product_terms:
            observation = inventory.search_inventory(product_terms)
            tool_observations.append(observation)
    elif mode == "semantic":
        # Semantic mode — embed query and retrieve nearest inventory items.
        product_terms = []
        observation = semantic_search_fn(query, inventory, top_k=5)
        tool_observations = [observation]
    else:
        # llm_only — skip extraction and pre-search; send the query directly
        product_terms = []
        tool_observations = []

    # Step 3 — generate final response (grounded or vanilla depending on mode)
    raw_response = generate_fn(
        ASSISTANT_NAME,
        query,
        inventory_summary,
        [],              # empty memory for independent per-query evaluation
        tool_observations,
        include_inventory_in_no_search=(mode == "llm_only"),
    )

    t_end = time.perf_counter()
    latency = round(t_end - t_start, 3)
    # ── Pipeline timing end ───────────────────────────────────────────────

    # ── Metric 1: JSON Adherence ──────────────────────────────────────────
    json_adherent = isinstance(raw_response, dict) and REQUIRED_JSON_KEYS.issubset(
        raw_response.keys()
    )

    # ── Metric 2: Task Success ────────────────────────────────────────────
    task_success = _is_task_success(raw_response, expected_items, inventory)

    # ── Metric 3: Hallucination ───────────────────────────────────────────
    hallucinated_aisles = _find_hallucinated_aisles(raw_response)
    hallucinated = len(hallucinated_aisles) > 0

    # ── Collect result ────────────────────────────────────────────────────
    result_row = {
        "id": qid,
        "mode": mode,
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

    return result_row


def _build_pipeline_callables(mock: bool):
    """Return ``(extract_fn, semantic_search_fn, generate_fn)`` for the requested backend."""
    if mock:
        return _mock_extract_product_terms, _mock_semantic_search, _mock_generate_response
    from engine.llm_client import extract_product_terms, generate_shop_assistant_response
    return (
        extract_product_terms,
        lambda query, inventory, top_k=5: inventory.semantic_search_inventory(query, top_k=top_k),
        generate_shop_assistant_response,
    )


def run_evaluation(
    queries: list[dict],
    mock: bool = False,
    verbose: bool = True,
    mode: str = "presearch",
    checkpoint_path: str | None = None,
) -> list[dict]:
    """Run all queries through the pipeline and return a list of result dicts.

    Parameters
    ----------
    queries:
        List of query dicts loaded from the JSON dataset.
    mock:
        When True, use deterministic mock functions instead of calling Ollama.
    verbose:
        Print per-query progress to stdout.
    mode:
        ``"presearch"`` — full pipeline (term extraction + inventory search +
        generation).
        ``"semantic"``  — semantic retrieval (embedding similarity + generation).
        ``"llm_only"``  — generation without inventory pre-search, but with
        full inventory context in the prompt.
    checkpoint_path:
        If provided, each result row is appended to this CSV immediately after
        it is computed, enabling crash-resume without losing completed work.
    """
    inventory = Inventory()
    inventory_summary = _build_inventory_summary(inventory)
    extract_fn, semantic_search_fn, generate_fn = _build_pipeline_callables(mock)

    results: list[dict] = []
    total = len(queries)

    for idx, item in enumerate(queries, start=1):
        result_row = _run_single_query(
            item, idx, total, inventory, inventory_summary,
            extract_fn, semantic_search_fn, generate_fn, mode, verbose,
        )
        results.append(result_row)

        if checkpoint_path:
            _append_csv([result_row], checkpoint_path)

    return results


def _run_comparison_modes(
    queries: list[dict],
    mock: bool = False,
    verbose: bool = True,
    modes: list[str] | None = None,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """Run each query in each mode from *modes* with atomic per-query checkpointing."""
    if modes is None:
        modes = ["presearch", "semantic", "llm_only"]
    inventory = Inventory()
    inventory_summary = _build_inventory_summary(inventory)
    extract_fn, semantic_search_fn, generate_fn = _build_pipeline_callables(mock)

    results: list[dict] = []
    total = len(queries)

    for idx, item in enumerate(queries, start=1):
        if verbose:
            print(f"\n  ── Query {idx}/{total} ({', '.join(modes)}) ──")

        rows_for_query = []
        for mode in modes:
            row = _run_single_query(
                item, idx, total, inventory, inventory_summary,
                extract_fn, semantic_search_fn, generate_fn, mode, verbose,
            )
            rows_for_query.append(row)
            results.append(row)

        # Checkpoint all rows atomically so we never write a partial mode set
        if checkpoint_path:
            _append_csv(rows_for_query, checkpoint_path)

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

def _pct(count: int, total: int) -> float:
    """Return *count* as a percentage of *total*, or 0.0 when *total* is zero."""
    return count / total * 100 if total else 0.0


def _mean(values: list) -> float:
    """Return the arithmetic mean of *values*, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


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

    # Per-mode breakdown — only populated when multiple modes are present
    modes_present = {r.get("mode", "presearch") for r in results}
    by_mode: dict[str, dict] = {}
    if len(modes_present) > 1:
        for r in results:
            m = r.get("mode", "presearch")
            if m not in by_mode:
                by_mode[m] = {
                    "total": 0, "success": 0, "hallucinated": 0, "latencies": [],
                }
            by_mode[m]["total"] += 1
            if r["task_success"]:
                by_mode[m]["success"] += 1
            if r["hallucinated"]:
                by_mode[m]["hallucinated"] += 1
            by_mode[m]["latencies"].append(r["latency_s"])

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
        "by_mode": by_mode,
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

    # Per-mode comparison table (shown when multiple modes are present)
    if summary.get("by_mode"):
        print()
        print("  Mode comparison:")
        m_header = f"  {'Mode':<12} {'Queries':>7} {'Success':>9} {'Hallucinated':>14} {'Latency mean':>14}"
        print(m_header)
        print("  " + "─" * 58)
        for mode_name, stats in summary["by_mode"].items():
            t = stats["total"]
            print(
                f"  {mode_name:<12} {t:>7}"
                f" {_pct(stats['success'], t):>8.1f}%"
                f" {_pct(stats['hallucinated'], t):>13.1f}%"
                f" {_mean(stats['latencies']):>13.3f}s"
            )

    print()
    print("  By query type:")
    header = f"  {'Type':<16} {'Queries':>7} {'Success':>9} {'Hallucinated':>14}"
    print(header)
    print("  " + "─" * 50)
    for qt, stats in summary["by_query_type"].items():
        t = stats["total"]
        print(
            f"  {qt:<16} {t:>7}"
            f" {_pct(stats['success'], t):>8.1f}%"
            f" {_pct(stats['hallucinated'], t):>13.1f}%"
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
    ]

    if summary.get("by_mode"):
        lines += [
            "",
            "Mode comparison:",
            f"  {'Mode':<12} {'Queries':>7} {'Success':>9} {'Hallucinated':>14} {'Latency mean':>14}",
            "  " + "─" * 58,
        ]
        for mode_name, stats in summary["by_mode"].items():
            t = stats["total"]
            lines.append(
                f"  {mode_name:<12} {t:>7}"
                f" {_pct(stats['success'], t):>8.1f}%"
                f" {_pct(stats['hallucinated'], t):>13.1f}%"
                f" {_mean(stats['latencies']):>13.3f}s"
            )

    lines += [
        "",
        "By query type:",
        f"  {'Type':<16} {'Queries':>7} {'Success':>9} {'Hallucinated':>14}",
        "  " + "─" * 50,
    ]
    for qt, stats in summary["by_query_type"].items():
        t = stats["total"]
        lines.append(
            f"  {qt:<16} {t:>7}"
            f" {_pct(stats['success'], t):>8.1f}%"
            f" {_pct(stats['hallucinated'], t):>13.1f}%"
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the NPC Engine LLM pipeline on a query dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes\n"
            "-----\n"
            "  presearch  Full pipeline: term extraction → inventory search → generation.\n"
            "  semantic   Semantic retrieval over inventory embeddings → generation.\n"
            "  llm_only   No inventory pre-search; prompt includes full inventory context.\n"
            "  all        Run every query in all three modes and save three labelled rows each.\n\n"
            "Resume support\n"
            "--------------\n"
            "  Rows are written to <output>/results.csv after each query.  Re-running\n"
            "  with the same --output dir automatically resumes from the last saved row."
        ),
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
        "--mode",
        choices=["llm_only", "presearch", "semantic", "all", "both"],
        default="presearch",
        help=(
            "Evaluation mode: 'presearch' (full pipeline, default), "
            "'semantic' (embedding retrieval + generation), "
            "'llm_only' (no pre-search, but includes inventory context), or "
            "'all' (compare all three per query). "
            "'both' is kept as a legacy alias for presearch + llm_only."
        ),
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Start evaluation from query index N (0-based).  "
            "Auto-detected from the existing results.csv when not specified."
        ),
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

    # ── Load query dataset ────────────────────────────────────────────────
    if not os.path.isfile(args.queries):
        print(f"[ERROR] Query file not found: {args.queries}", file=sys.stderr)
        return 1
    with open(args.queries, encoding="utf-8") as f:
        queries: list[dict] = json.load(f)

    if args.limit:
        queries = queries[: args.limit]

    # ── Determine starting query index (resume support) ───────────────────
    resume_csv = os.path.join(args.output, ROLLING_RESULTS_FILENAME)

    if args.start_from is not None:
        start_idx = args.start_from
        print(f"[Resume] --start_from override: starting at query index {start_idx}")
    else:
        existing_rows = _count_existing_results(resume_csv)
        if args.mode == "all":
            # Each query produces 3 rows (presearch + semantic + llm_only)
            start_idx = existing_rows // 3
        elif args.mode == "both":
            # Legacy alias: each query produces 2 rows (presearch + llm_only)
            start_idx = existing_rows // 2
        else:
            start_idx = existing_rows
        if start_idx > 0:
            print(f"[Resume] Found {existing_rows} existing row(s) in {resume_csv}")
            print(f"         Resuming from query index {start_idx} "
                  f"(skipping {start_idx} already-completed queries)")

    if start_idx >= len(queries):
        print(
            f"[INFO] All {len(queries)} queries already completed "
            f"(found {_count_existing_results(resume_csv)} rows in {resume_csv}). "
            "Nothing to do."
        )
        return 0

    queries_to_run = queries[start_idx:]

    # ── Print run header ──────────────────────────────────────────────────
    llm_label = "MOCK (no Ollama)" if args.mock else "LIVE (Ollama/Mistral)"
    if args.mode == "both":
        print("[INFO] '--mode both' is a legacy alias. Prefer '--mode all' for full comparison.")
    print(f"\nNPC Engine — LLM Pipeline Evaluation")
    print(f"LLM     : {llm_label}")
    print(f"Mode    : {args.mode}")
    print(
        f"Queries : {len(queries_to_run)}"
        + (f" (of {len(queries)} total, resuming from index {start_idx})"
           if start_idx > 0 else f" (of {len(queries)} total)")
    )
    print(f"Output  : {args.output}")
    print()

    # ── Run evaluation ────────────────────────────────────────────────────
    if args.mode == "all":
        results = _run_comparison_modes(
            queries_to_run,
            mock=args.mock,
            verbose=not args.quiet,
            modes=["presearch", "semantic", "llm_only"],
            checkpoint_path=resume_csv,
        )
    elif args.mode == "both":
        results = _run_comparison_modes(
            queries_to_run,
            mock=args.mock,
            verbose=not args.quiet,
            modes=["presearch", "llm_only"],
            checkpoint_path=resume_csv,
        )
    else:
        results = run_evaluation(
            queries_to_run,
            mock=args.mock,
            verbose=not args.quiet,
            mode=args.mode,
            checkpoint_path=resume_csv,
        )

    # ── Compute and display summary ───────────────────────────────────────
    summary = _compute_summary(results)
    _print_summary(summary)

    # ── Persist timestamped snapshot of this run ──────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = _save_csv(results, args.output, timestamp)
    summary_path = _save_summary(summary, args.output, timestamp)
    print(f"\n  Results CSV  → {csv_path}")
    print(f"  Rolling CSV  → {resume_csv}")
    print(f"  Summary file → {summary_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
