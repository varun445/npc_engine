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
# Captures text after "aisle"/"aisles" so we can extract all referenced numbers.
AISLE_CLAUSE_RE = re.compile(r"(?i)\baisles?\b([^.;\n]*)")
AISLE_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:(?:are|is)\s+)?(?:in\s+)?(\d+(?:\s*(?:,|and|or|&|/)\s*\d+)*)\b",
    re.IGNORECASE,
)
REQUIRED_JSON_KEYS = {"dialogue", "action", "target_aisles"}
MAX_DIALOGUE_SNIPPET_LENGTH = 120
ASSISTANT_NAME = "Alex"
MODE_ALIAS_FOR_PLOTTING = {
    "direct": "llm_only",
}

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
    query_lower = (query or "").lower()

    # Keep parity with recipe-style extraction in the real LLM pipeline.
    recipe_map = {
        "cake": ["milk", "cheese", "bread_white", "yogurt"],
        "salad": ["lettuce", "tomato", "carrot", "broccoli"],
        "smoothie": ["banana", "apple", "orange", "yogurt"],
        "breakfast": ["milk", "bread_white", "banana", "coffee"],
        "sandwich": ["bread_white", "cheese", "tomato", "lettuce"],
    }
    for key, ingredients in recipe_map.items():
        if key in query_lower:
            return ingredients

    inventory = Inventory()
    all_names = []
    for products in inventory.products.values():
        for p in products:
            all_names.append(p["name"].lower())
            all_names.append(p["id"].lower())
    found = []
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
    retrieval_mode: str = "presearch",
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


def _mock_semantic_search(
    query: str,
    inventory: Inventory,
    top_k: int = 5,
    query_terms: list[str] | None = None,
    min_score: float = 0.15,
) -> str:
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

    _ = min_score  # parity with live semantic_search_inventory signature
    keyword_terms = query_terms if query_terms else _mock_extract_product_terms(query)
    semantic_terms.extend(keyword_terms)

    # Preserve order while deduplicating.
    deduplicated_terms = []
    seen = set()
    for term in semantic_terms:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduplicated_terms.append(term)

    if not deduplicated_terms:
        deduplicated_terms = query_lower.split()

    # Reuse deterministic exact matcher but cap to top_k output products.
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    if top_k <= 0:
        return f"Search Results: {query}: not found in store inventory"

    matched = inventory.find_products_by_terms(deduplicated_terms)[:top_k]
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


def _extract_deterministic_terms(query: str, inventory: Inventory, max_terms: int = 8) -> list[str]:
    """Extract lexical product terms from query text without an LLM call."""
    query_lower = (query or "").lower().strip()
    if not query_lower:
        return []

    terms: list[str] = []
    seen = set()
    stopwords = set(getattr(Inventory, "_SEMANTIC_STOPWORDS", set()))

    def _add(term: str) -> None:
        key = (term or "").strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        terms.append(key)

    # 1) Exact product-name phrase match from inventory surface form.
    for products in inventory.products.values():
        for product in products:
            clean_name = re.sub(r"[^a-z0-9\s]", " ", product["name"].lower())
            clean_name = re.sub(r"\s+", " ", clean_name).strip()
            if clean_name and clean_name in query_lower:
                _add(clean_name)
            pid = str(product["id"]).strip().lower()
            if pid and pid in query_lower:
                _add(pid)

    # 2) Token overlap against inventory id/name tokens.
    inventory_tokens = set()
    for products in inventory.products.values():
        for product in products:
            inventory_tokens.update(re.findall(r"[a-z0-9]+", str(product["id"]).lower()))
            inventory_tokens.update(re.findall(r"[a-z0-9]+", str(product["name"]).lower()))
    for token in re.findall(r"[a-z0-9]+", query_lower):
        if len(token) <= 2 or token in stopwords:
            continue
        if token in inventory_tokens:
            _add(token)
        if len(terms) >= max_terms:
            break

    return terms[:max_terms]


# ---------------------------------------------------------------------------
# Hallucination detection helpers
# ---------------------------------------------------------------------------

def _normalize_aisle_values(value) -> list[int]:
    """Normalize aisle payloads into a list of integer aisle IDs."""
    if value is None:
        values = []
    elif isinstance(value, (str, bytes, int, float)):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]

    normalized = []
    for aisle in values:
        try:
            normalized.append(int(aisle))
        except (TypeError, ValueError):
            continue
    return normalized


def _expected_stocked_aisles(expected_items: list[str], inventory: Inventory) -> set[int]:
    """Return aisles of in-stock products matched by expected query terms."""
    if not expected_items:
        return set()

    aisles = set()
    for term in expected_items:
        for product in inventory.find_products_by_terms([term]):
            category = product.get("category")
            if not category:
                continue
            aisle_num = inventory.aisles.get(category)
            if aisle_num is not None:
                aisles.add(int(aisle_num))
    return aisles


def _find_hallucinated_aisles(response: dict, expected_aisles: set[int] | None = None) -> list[int]:
    """Return a list of aisle numbers referenced that are not valid store aisles,
    that do not map to expected in-stock aisles, or that were claimed by the
    LLM but removed by the grounding filter.

    Checks:
    1. ``target_aisles`` (structured field, after grounding)
    2. Mismatch against ``expected_aisles`` when provided.
    3. ``_raw_target_aisles`` (LLM-claimed aisles *before* grounding) — aisles
       that were removed by grounding indicate the model hallucinated navigation
       targets not backed by any search result.
    4. Any "Aisle N" mentions inside the ``dialogue`` string.
    """
    invalid: list[int] = []

    # 1. Structured field (post-grounding)
    target_aisles = _normalize_aisle_values(response.get("target_aisles", []))
    for aisle_int in target_aisles:
        if aisle_int not in VALID_AISLES:
            invalid.append(aisle_int)

    # If this query has known expected aisle(s), any target aisle outside that
    # set is an aisle hallucination (e.g., milk reported in aisle 4 instead of 1).
    if expected_aisles:
        for aisle_int in target_aisles:
            if aisle_int not in expected_aisles and aisle_int not in invalid:
                invalid.append(aisle_int)

    # 2. Raw aisles claimed by the LLM before the grounding filter was applied.
    #    Any aisle present in _raw_target_aisles but NOT in the post-grounding
    #    target_aisles was removed because it had no inventory backing — i.e.
    #    the model hallucinated it.
    grounded_set = set(target_aisles)
    raw_aisles = response.get("_raw_target_aisles", [])
    if raw_aisles is None:
        raw_aisles = []
    for aisle_int in _normalize_aisle_values(raw_aisles):
        if aisle_int not in grounded_set and aisle_int not in invalid:
            # This aisle was claimed by the LLM but removed by grounding
            # (i.e. not backed by any found inventory item).
            invalid.append(aisle_int)

    # 3. Inline dialogue mentions — check for invalid store aisles
    dialogue = response.get("dialogue", "")
    for m in AISLE_REFERENCE_RE.finditer(dialogue):
        aisle_num = int(m.group(1))
        if aisle_num not in VALID_AISLES and aisle_num not in invalid:
            invalid.append(aisle_num)
        elif expected_aisles and aisle_num not in expected_aisles and aisle_num not in invalid:
            invalid.append(aisle_num)

    # 4. Plural/compound aisle mentions, e.g. "Aisles 1 and 9"
    for m in AISLE_CLAUSE_RE.finditer(dialogue):
        clause = m.group(1) or ""
        list_match = AISLE_LIST_PREFIX_RE.search(clause)
        if not list_match:
            continue
        list_segment = list_match.group(1)
        for num_str in re.findall(r"\d+", list_segment):
            aisle_num = int(num_str)
            if aisle_num not in VALID_AISLES and aisle_num not in invalid:
                invalid.append(aisle_num)
            elif expected_aisles and aisle_num not in expected_aisles and aisle_num not in invalid:
                invalid.append(aisle_num)

    return invalid


# ---------------------------------------------------------------------------
# Task success helpers
# ---------------------------------------------------------------------------

def _is_task_success(
    response: dict,
    expected_items: list[str],
    inventory: Inventory,
    hallucinated_aisles: list[int] | None = None,
) -> bool:
    """Determine whether the agent handled the query correctly.

    Rules:
    - If expected_items is non-empty (items should be found):
        The agent must have set action to "move" and populated target_aisles
        with at least one valid aisle.
    - If expected_items is empty (conversational / invalid query):
        The agent must have set action to "none".
    """
    if hallucinated_aisles:
        return False

    action = str(response.get("action", "none")).strip().lower()
    target_aisles = _normalize_aisle_values(response.get("target_aisles", []))

    # Determine which of the expected items are actually in stock
    stocked_terms = []
    if expected_items:
        for term in expected_items:
            matches = inventory.find_products_by_terms([term])
            if matches:
                stocked_terms.append(term)

    expected_aisles = _expected_stocked_aisles(expected_items, inventory)

    if stocked_terms:
        # At least one expected in-stock item → agent should navigate
        if action != "move" or len(target_aisles) == 0:
            return False
        if any(aisle not in VALID_AISLES for aisle in target_aisles):
            return False
        if expected_aisles and not any(aisle in expected_aisles for aisle in target_aisles):
            return False
        return True
    else:
        # No in-stock items expected → agent should stay put
        return action == "none" and len(target_aisles) == 0


def _is_json_adherent(response: dict) -> bool:
    """Return True when required response fields are present and parseable.

    This is intentionally tolerant of extra keys (e.g. internal metadata like
    ``_raw_target_aisles``) as long as the three required fields can be
    extracted in usable form.
    """
    if not isinstance(response, dict):
        return False
    if not REQUIRED_JSON_KEYS.issubset(response.keys()):
        return False

    dialogue = response.get("dialogue")
    action = response.get("action")
    target_aisles = response.get("target_aisles")

    if isinstance(dialogue, (dict, list)) or dialogue is None:
        return False
    if isinstance(action, (dict, list)) or action is None:
        return False

    parsed_aisles = target_aisles
    if isinstance(target_aisles, str):
        text = target_aisles.strip()
        if text == "":
            parsed_aisles = []
        else:
            try:
                parsed_aisles = json.loads(text)
            except json.JSONDecodeError:
                return False
    elif isinstance(target_aisles, (int, float)):
        parsed_aisles = [target_aisles]
    elif isinstance(target_aisles, tuple):
        parsed_aisles = list(target_aisles)

    if not isinstance(parsed_aisles, list):
        return False

    for aisle in parsed_aisles:
        try:
            int(aisle)
        except (TypeError, ValueError):
            return False
    return True


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


def _to_int_bool(value) -> int:
    """Convert bool-like values to 0/1 for plot-friendly CSV fields."""
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "yes", "y"} else 0


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
    query_type_key = str(query_type).strip().lower()
    expected_items = item.get("expected_items", [])

    if verbose:
        print(
            f"  [{idx:02d}/{total}] Q{qid} ({query_type:14s})"
            f" [{mode:9s}]: {query[:60]!r}"
        )

    # ── Write query header to structured log ─────────────────────────────
    try:
        from engine.llm_client import log_step
        log_step(
            f"\n{'═'*72}\n"
            f"QUERY [{idx}/{total}] id={qid}  mode={mode}  type={query_type}\n"
            f"  query : {query}\n"
            f"{'═'*72}\n"
        )
    except Exception:
        pass

    # ── Pipeline timing start ─────────────────────────────────────────────
    t_start = time.perf_counter()

    extraction_source = "none"
    llm_extraction_empty = False
    semantic_profile_top_k = ""
    semantic_profile_min_score = ""

    if mode == "presearch":
        # Step 1 — extract product terms from the query
        product_terms = extract_fn(query)
        if product_terms:
            extraction_source = "llm_extract"
        else:
            llm_extraction_empty = True

        # Fallback A — deterministic lexical term extraction (no LLM).
        if not product_terms:
            product_terms = _extract_deterministic_terms(query, inventory, max_terms=8)
            if product_terms:
                extraction_source = "deterministic_terms"

        # Fallback B — broader semantic query-term extraction.
        if not product_terms:
            product_terms = inventory.extract_semantic_query_terms(query, max_terms=12)
            if product_terms:
                extraction_source = "semantic_terms_fallback"

        # Step 2 — deterministic inventory search (the "pre-search hook")
        tool_observations: list[str] = []
        if product_terms:
            observation = inventory.search_inventory(product_terms)
            tool_observations.append(observation)
    elif mode == "semantic":
        semantic_profiles = {
            "direct": {"top_k": 3, "min_score": 0.30},
            "not_in_store": {"top_k": 4, "min_score": 0.30},
            "recipe": {"top_k": 6, "min_score": 0.18},
            "associative": {"top_k": 6, "min_score": 0.20},
            "conversational": {"top_k": 0, "min_score": 1.00},
        }
        profile = semantic_profiles.get(query_type_key, {"top_k": 5, "min_score": 0.24})
        semantic_profile_top_k = int(profile["top_k"])
        semantic_profile_min_score = float(profile["min_score"])

        # Semantic mode (2-step):
        # 1) run product/recipe extraction first (same intent extraction as presearch),
        # 2) run semantic retrieval over extracted terms; fall back to semantic query terms
        #    for associative requests (e.g. "healthy", "snack", "drink").
        extracted_terms = extract_fn(query)
        if extracted_terms:
            product_terms = extracted_terms
            extraction_source = "llm_extract"
        else:
            llm_extraction_empty = True
            product_terms = inventory.extract_semantic_query_terms(query)
            if product_terms:
                extraction_source = "semantic_terms_fallback"

        if semantic_profile_top_k <= 0:
            tool_observations = []
        else:
            observation = semantic_search_fn(
                query,
                inventory,
                top_k=semantic_profile_top_k,
                query_terms=product_terms,
                min_score=semantic_profile_min_score,
            )
            tool_observations = [observation]
        # Log intermediate semantic search result
        try:
            from engine.llm_client import log_step
            log_step(
                f"[SEMANTIC PROFILE] type={query_type_key} top_k={semantic_profile_top_k} min_score={semantic_profile_min_score:.2f}\n"
                f"[SEMANTIC TERMS  ] extracted={product_terms}\n"
                f"[SEMANTIC OBS    ] {tool_observations[0] if tool_observations else '(no retrieval: conversational profile)'}\n"
            )
        except Exception:
            pass
    else:
        # llm_only — skip extraction and pre-search; send the query directly
        product_terms = []
        tool_observations = []

    used_inventory_context_fallback = (
        (mode == "llm_only") or (mode in {"presearch", "semantic"} and not tool_observations)
    )

    # Step 3 — generate final response (grounded or vanilla depending on mode)
    raw_response = generate_fn(
        ASSISTANT_NAME,
        query,
        inventory_summary,
        [],              # empty memory for independent per-query evaluation
        tool_observations,
        include_inventory_in_no_search=used_inventory_context_fallback,
        retrieval_mode=mode,
    )

    t_end = time.perf_counter()
    latency = round(t_end - t_start, 3)
    # ── Pipeline timing end ───────────────────────────────────────────────

    # ── Metric 1: JSON Adherence ──────────────────────────────────────────
    json_adherent = _is_json_adherent(raw_response)

    # ── Metric 2: Hallucination ───────────────────────────────────────────
    expected_aisles = _expected_stocked_aisles(expected_items, inventory)
    hallucinated_aisles = _find_hallucinated_aisles(
        raw_response,
        expected_aisles=expected_aisles,
    )
    hallucinated = len(hallucinated_aisles) > 0
    raw_target_aisles = _normalize_aisle_values(raw_response.get("_raw_target_aisles", []))
    grounded_target_aisles = _normalize_aisle_values(raw_response.get("target_aisles", []))
    grounding_drop_count = max(0, len(raw_target_aisles) - len(grounded_target_aisles))
    retrieval_no_results = (
        len(tool_observations) == 0
        or all("not found in store inventory" in (obs or "") for obs in tool_observations)
    )
    if hallucinated:
        raw_removed = any(aisle not in set(grounded_target_aisles) for aisle in raw_target_aisles)
        hallucination_source = "raw_removed_by_grounding" if raw_removed else "dialogue_or_expected_mismatch"
    else:
        hallucination_source = ""

    # ── Metric 3: Task Success ────────────────────────────────────────────
    task_success = _is_task_success(
        raw_response,
        expected_items,
        inventory,
        hallucinated_aisles=hallucinated_aisles,
    )

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
        "json_adherent_int": _to_int_bool(json_adherent),
        "task_success": task_success,
        "task_success_int": _to_int_bool(task_success),
        "hallucinated": hallucinated,
        "hallucinated_int": _to_int_bool(hallucinated),
        "hallucinated_aisles": str(hallucinated_aisles),
        "extraction_source": extraction_source,
        "llm_extraction_empty": llm_extraction_empty,
        "retrieval_no_results": retrieval_no_results,
        "used_inventory_context_fallback": used_inventory_context_fallback,
        "semantic_top_k": semantic_profile_top_k if mode == "semantic" else "",
        "semantic_min_score": semantic_profile_min_score if mode == "semantic" else "",
        "raw_target_aisles": str(raw_target_aisles),
        "grounded_target_aisles": str(grounded_target_aisles),
        "grounding_drop_count": grounding_drop_count,
        "hallucination_source": hallucination_source,
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
        lambda query, inventory, top_k=5, query_terms=None, min_score=0.15: inventory.semantic_search_inventory(
            query, top_k=top_k, query_terms=query_terms, min_score=min_score
        ),
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


def _canonical_mode_for_plot(mode: str, available_modes: set[str]) -> str:
    """Resolve display mode aliases (e.g., 'direct' -> 'llm_only') for plotting."""
    mode = (mode or "").strip()
    if mode in available_modes:
        return mode
    alias = MODE_ALIAS_FOR_PLOTTING.get(mode, mode)
    if alias in available_modes:
        return alias
    return mode


def _float_or_zero(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def plot_mode_comparison(
    results_csv_path: str,
    output_image_path: str,
    modes: list[str] | None = None,
) -> str | None:
    """Create a mode-comparison bar plot from a results CSV."""
    if modes is None:
        modes = ["direct", "presearch", "semantic"]
    if not os.path.isfile(results_csv_path):
        return None

    rows = []
    with open(results_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    available_modes = {r.get("mode", "").strip() for r in rows if r.get("mode")}
    if not available_modes:
        return None

    selected_modes = []
    seen_canonical_modes = set()
    for mode in modes:
        canonical = _canonical_mode_for_plot(mode, available_modes)
        if canonical in available_modes and canonical not in seen_canonical_modes:
            selected_modes.append((mode, canonical))
            seen_canonical_modes.add(canonical)

    if not selected_modes:
        return None

    metrics_by_mode = []
    labels = []
    for display_mode, actual_mode in selected_modes:
        mode_rows = [r for r in rows if r.get("mode", "").strip() == actual_mode]
        if not mode_rows:
            continue

        success_vals = []
        hallucinated_vals = []
        json_vals = []
        latencies = []
        for r in mode_rows:
            success_raw = r.get("task_success_int", r.get("task_success", "0"))
            hall_raw = r.get("hallucinated_int", r.get("hallucinated", "0"))
            json_raw = r.get("json_adherent_int", r.get("json_adherent", "0"))
            success_vals.append(_to_int_bool(success_raw))
            hallucinated_vals.append(_to_int_bool(hall_raw))
            json_vals.append(_to_int_bool(json_raw))
            latencies.append(_float_or_zero(r.get("latency_s")))

        total = len(mode_rows)
        metrics_by_mode.append(
            {
                "success_rate": (_mean(success_vals) * 100) if total else 0.0,
                "hallucination_rate": (_mean(hallucinated_vals) * 100) if total else 0.0,
                "json_rate": (_mean(json_vals) * 100) if total else 0.0,
                "latency_mean": _mean(latencies),
            }
        )
        labels.append(display_mode)

    if not metrics_by_mode:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_dir = os.path.dirname(output_image_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    x = list(range(len(labels)))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    success_rates = [m["success_rate"] for m in metrics_by_mode]
    hallucination_rates = [m["hallucination_rate"] for m in metrics_by_mode]
    json_rates = [m["json_rate"] for m in metrics_by_mode]
    latency_means = [m["latency_mean"] for m in metrics_by_mode]

    axes[0].bar([i - width for i in x], success_rates, width=width, label="Success %")
    axes[0].bar(x, hallucination_rates, width=width, label="Hallucination %")
    axes[0].bar([i + width for i in x], json_rates, width=width, label="JSON %")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Evaluation Rates by Mode")
    axes[0].legend()

    axes[1].bar(x, latency_means, width=0.55, color="#4C78A8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean latency (s)")
    axes[1].set_title("Latency by Mode")

    fig.suptitle(f"Mode Comparison ({os.path.basename(results_csv_path)})")
    fig.tight_layout()
    fig.savefig(output_image_path, dpi=150)
    plt.close(fig)
    return output_image_path


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
            "'both' is a deprecated alias for presearch + llm_only."
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
    parser.add_argument(
        "--log_file",
        default=None,
        metavar="PATH",
        help=(
            "Path for the structured run log (.txt).  "
            "Defaults to <output>/run_<timestamp>.log when not specified.  "
            "Pass 'none' to disable file logging entirely."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a mode-comparison plot from a results CSV after evaluation.",
    )
    parser.add_argument(
        "--plot_csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to results CSV used for plotting. Defaults to <output>/results.csv "
            "when --plot is set."
        ),
    )
    parser.add_argument(
        "--plot_modes",
        default="direct,presearch,semantic",
        help=(
            "Comma-separated mode labels to compare in the plot. "
            "Supports aliases like 'direct' (mapped to llm_only in existing CSVs)."
        ),
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

    # ── Set up structured run log ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(args, "log_file", None) and args.log_file.lower() == "none":
        log_path = None
    else:
        log_path = getattr(args, "log_file", None) or os.path.join(
            args.output, f"run_{timestamp}.log"
        )
    if not args.mock:
        # Only set up file logging when using a live LLM; skip for mock runs
        # so CI output stays clean (mock runs produce no interesting LLM text).
        try:
            from engine.llm_client import setup_log_file
            setup_log_file(log_path)
            if log_path:
                print(f"Log file : {log_path}")
        except ImportError:
            pass
    else:
        log_path = None

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
    csv_path = _save_csv(results, args.output, timestamp)
    summary_path = _save_summary(summary, args.output, timestamp)
    print(f"\n  Results CSV  → {csv_path}")
    print(f"  Rolling CSV  → {resume_csv}")
    print(f"  Summary file → {summary_path}")

    if args.plot:
        plot_source_csv = args.plot_csv or resume_csv
        plot_modes = [m.strip() for m in (args.plot_modes or "").split(",") if m.strip()]
        plot_path = os.path.join(args.output, f"mode_comparison_{timestamp}.png")
        generated = plot_mode_comparison(
            results_csv_path=plot_source_csv,
            output_image_path=plot_path,
            modes=plot_modes,
        )
        if generated:
            print(f"  Plot file    → {generated}")
        else:
            print("  Plot file    → [not generated: missing CSV data or matplotlib]")

    if log_path:
        print(f"  Run log      → {log_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
