#!/usr/bin/env python3
"""
01_sample_queries.py — Sample evaluation queries from valid_ids + citation graph.

PURPOSE:
    Build a reproducible evaluation query set of 500 rulings per language
    (1500 total) where every ground-truth citation is guaranteed to be
    present in both Qdrant and the citation graph, and is temporally valid.

    No external HuggingFace dataset is required. All inputs live locally:
      - valid_ids.json   — intersection of Qdrant ∩ citation graph (131,125)
      - citation_graph.pkl — provides cited_rulings (case_to_case edges) and
                             cited_laws (case_to_law edges) for every ruling
      - Qdrant (TEI/aiserver01) — provides query_text, language, year, date
                                  for every decision_id we sample

    HF is only needed if you want to re-compute valid_ids or citation_graph
    from scratch (via build_citation_graph.py). For routine sampling we work
    entirely off the derived artefacts, so Stage 1 is a pure graph+vector
    operation with no network dataset pull.

OUTPUT:
    data/eval/eval_queries.jsonl   (env-var EVAL_DIR)
    Each line is a JSON object with keys:
        query_id            — ruling UUID (matches Qdrant decision_id)
        query_text          — first 512 whitespace-tokens of the HF `facts`
                              field (Sachverhalt only, leak-free)
        ground_truth_cases  — cited rulings filtered to valid_ids + temporal
        ground_truth_laws   — cited law UUIDs (legislation nodes in graph)
        language            — "de" | "fr" | "it"
        year                — int (derived from date_ms)
        date_ms             — int, Unix epoch milliseconds

USAGE:
    python3 01_sample_queries.py              # N = 500/lang, 1500 total
    python3 01_sample_queries.py --n 100      # smaller sample
    python3 01_sample_queries.py --dry-run    # resource smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from tqdm import tqdm


# ── Config (env-var driven) ───────────────────────────────────────────────────

EVAL_DIR = Path(os.environ.get("EVAL_DIR", "data/eval"))
OUTPUT_FILE = EVAL_DIR / "eval_queries.jsonl"
VALID_IDS_PATH = Path(os.environ.get("VALID_IDS_PATH", str(EVAL_DIR / "valid_ids.json")))
DATE_INDEX_CACHE = Path(os.environ.get("DATE_INDEX_CACHE", str(EVAL_DIR / "date_index.json")))
GRAPH_PATH = Path(os.environ.get("GRAPH_PATH", str(EVAL_DIR / "citation_graph.pkl")))
METADATA_CACHE = Path(os.environ.get("METADATA_CACHE", str(EVAL_DIR / "decision_metadata.json")))
FACTS_INDEX_PATH = Path(os.environ.get("FACTS_INDEX_PATH", str(EVAL_DIR / "facts_index.jsonl")))

QDRANT_HOST = os.environ.get("QDRANT_HOST", "aiserver01")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "bger"
RULINGS_SOURCE = "swiss_rulings_chunked"
LEADING_SOURCE = "swiss_leading_decisions_chunked"

# Sampling parameters
PER_LANGUAGE_N = 4226
LANGUAGES = ["de", "fr", "it"]
RANDOM_SEED = 42
FIRST_N_TOKENS = 4096
QUERY_TEXT_CHUNKS = 3   # number of leading chunks to concatenate as query_text

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cap text at `max_tokens` whitespace-separated tokens."""
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def year_from_date_ms(date_ms: int) -> int:
    """Extract calendar year from Unix epoch milliseconds."""
    if not date_ms:
        return 0
    try:
        return datetime.fromtimestamp(int(date_ms) / 1000, tz=timezone.utc).year
    except (ValueError, OSError):
        return 0


def build_decision_metadata(
    client: QdrantClient, sources: list, restrict_to: set | None = None
) -> dict:
    """Scroll Qdrant (chunk_index=0 only) to build decision_id → metadata.

    Returns:
        dict[decision_id -> {"date_ms": int, "language": str, "court": str}]
    """
    log.info("Building decision metadata from Qdrant (chunk_index=0 scroll) ...")
    index: dict = {}
    offset = None
    must_filters = [
        FieldCondition(key="chunk_index", match=MatchValue(value=0)),
        Filter(should=[FieldCondition(key="source", match=MatchValue(value=s))
                       for s in sources]),
    ]
    with tqdm(desc="Scrolling Qdrant", unit="pts") as pbar:
        while True:
            batch, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=Filter(must=must_filters),
            )
            for pt in batch:
                did = pt.payload.get("decision_id")
                if not did:
                    continue
                if restrict_to is not None and did not in restrict_to:
                    continue
                if did in index:
                    continue
                raw_date = pt.payload.get("date", "0")
                try:
                    date_ms = int(raw_date)
                except (TypeError, ValueError):
                    date_ms = 0
                index[did] = {
                    "date_ms": date_ms,
                    "language": (pt.payload.get("language") or "?").lower(),
                    "court": pt.payload.get("court", "") or "",
                }
            pbar.update(len(batch))
            if offset is None:
                break
    log.info("Metadata built: %d decisions", len(index))
    return index


def load_facts_index(path: Path) -> dict:
    """Load `decision_id → facts` from the JSONL artefact built by
    `build_facts_index.py`.

    Using the HF dataset's structured `facts` column for query text is the
    leak-free alternative to scrolling the first chunks of the full
    `swiss_rulings_chunked` collection. The Qdrant chunks contain the
    decision's `considerations` section as well, which in BGer rulings is
    where explicit BGE citations to precedent cases appear. Putting that
    text into the query embedding leaks the ground-truth labels.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"facts index not found at {path}. Run scripts/eval/build_facts_index.py first."
        )
    log.info("Loading facts index from %s ...", path)
    index: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["decision_id"]] = row.get("facts") or ""
    log.info("Facts index loaded: %d decisions", len(index))
    return index


def build_ground_truth(
    graph, decision_id: str, valid_ids: set, query_date_ms: int,
    metadata: dict,
) -> tuple[list, list]:
    """Return (cited_rulings, cited_laws) for one decision with filters applied.

    - cited_rulings: graph successors via `case_to_case` edges, only kept if
                     the target is in valid_ids AND its date < query_date_ms.
    - cited_laws:    graph successors via `case_to_law` edges. Laws aren't in
                     Qdrant, but they are graph nodes and are returned as-is.
    """
    if decision_id not in graph:
        return [], []
    cited_rulings = []
    cited_laws = []
    for target, attrs in graph[decision_id].items():
        edge_type = attrs.get("type")
        if edge_type == "case_to_case":
            if target not in valid_ids:
                continue
            cited_date = metadata.get(target, {}).get("date_ms", 0)
            # Strict temporal filter, consistent with the candidate-pool gate
            # in `run()`: targets with unknown date are dropped rather than
            # silently kept as untested GT. Without this the strict-GT
            # contract (every GT temporally pre-Query) is violated for ~17%
            # of GT entries whose cited_date is missing in metadata.
            if not cited_date or cited_date >= query_date_ms:
                continue
            cited_rulings.append(target)
        elif edge_type == "case_to_law":
            cited_laws.append(target)
    return cited_rulings, cited_laws


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    per_language_n: int = PER_LANGUAGE_N,
    dry_run: bool = False,
    on_event=None,
    cancel_check=None,
) -> None:
    """Sample N queries per language directly from valid_ids + graph + Qdrant.

    No HF dataset is loaded. All inputs are local artefacts produced by the
    graph-building step and the Qdrant-embedding step that run upstream.
    """
    def emit(event):
        if on_event:
            on_event(event)

    emit({"type": "stage_started", "stage": "sample"})
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load valid_ids (sampling pool) ─────────────────────────────────
    log.info("Loading valid_ids from %s ...", VALID_IDS_PATH)
    with open(VALID_IDS_PATH) as f:
        valid_ids: set = set(json.load(f))
    log.info("valid_ids loaded: %d", len(valid_ids))

    # ── 2. Load citation graph ────────────────────────────────────────────
    log.info("Loading citation graph from %s ...", GRAPH_PATH)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    log.info(
        "Graph: %d nodes, %d edges",
        graph.number_of_nodes(), graph.number_of_edges(),
    )

    # ── 3. Decision metadata (date + language + court) from Qdrant ────────
    # Cache-first: if `decision_metadata.json` exists we trust it. Falls back
    # to a live Qdrant scroll filtered to chunk_index=0. Restricted to
    # valid_ids so we don't pull metadata for decisions we'll never sample.
    if METADATA_CACHE.exists():
        log.info("Loading decision metadata from cache %s ...", METADATA_CACHE)
        with open(METADATA_CACHE) as f:
            metadata: dict = json.load(f)
        log.info("Metadata cache hit: %d entries", len(metadata))
    else:
        log.info("Connecting to Qdrant at %s:%d ...", QDRANT_HOST, QDRANT_PORT)
        client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT,
            check_compatibility=False, timeout=300,
        )
        metadata = build_decision_metadata(
            client, sources=[RULINGS_SOURCE, LEADING_SOURCE],
            restrict_to=valid_ids,
        )
        with open(METADATA_CACHE, "w") as f:
            json.dump(metadata, f)
        log.info("Metadata cached to %s", METADATA_CACHE)

    # Dry-run exits here: we verified the three sources load and intersect.
    if dry_run:
        sample_ids = list(valid_ids)[:5]
        log.info("DRY RUN — showing 5 sample decisions:")
        for did in sample_ids:
            md = metadata.get(did, {})
            succ_case = [t for t, a in graph[did].items() if a.get("type") == "case_to_case"] \
                if did in graph else []
            succ_law = [t for t, a in graph[did].items() if a.get("type") == "case_to_law"] \
                if did in graph else []
            print(f"  [{did}] lang={md.get('language','?')} "
                  f"date_ms={md.get('date_ms',0)} "
                  f"cited_rulings={len(succ_case)} cited_laws={len(succ_law)}")
        print("Dry run OK — graph + metadata + valid_ids all consistent")
        return

    # ── 3b. Load facts-index up front so candidates without facts are
    # excluded BEFORE the language-stratified sample. Loading later (after
    # sampling) caused per-language imbalance: Italian had a 28% facts-miss
    # rate vs French 5%, so a balanced N/lang sample silently became
    # unbalanced after the post-sample facts-miss drops.
    facts_index = load_facts_index(FACTS_INDEX_PATH)

    # ── 4. Build candidate pool per language ──────────────────────────────
    # A valid candidate is a ruling in valid_ids with:
    #   - language ∈ LANGUAGES
    #   - metadata present (date_ms > 0)
    #   - facts text available in facts_index
    #   - at least one case_to_case successor that is also in valid_ids AND
    #     has an earlier date (strict GT filter — identical to the one
    #     documented in thesis/03-methodik.md)
    log.info("Building strict-GT candidate pool ...")
    candidates_by_lang: dict = {lang: [] for lang in LANGUAGES}
    skipped_no_metadata = 0
    skipped_no_valid_gt = 0
    skipped_wrong_lang = 0
    skipped_no_facts = 0

    # cancel-check hook for UI
    cancel_after = 50000 if cancel_check else None

    # Sort valid_ids deterministically before iterating: set-iteration over
    # UUID strings is randomised per-process via PYTHONHASHSEED, which makes
    # the candidate-pool order non-reproducible. The downstream
    # `rng.shuffle(pool)` with a fixed seed then samples a different subset
    # on each fresh process.
    for i, did in enumerate(tqdm(sorted(valid_ids), desc="Filtering", unit="ids")):
        if cancel_after and i % cancel_after == 0 and cancel_check and cancel_check():
            log.warning("Cancel requested — stopping candidate build")
            return
        md = metadata.get(did)
        if not md:
            skipped_no_metadata += 1
            continue
        lang = md.get("language", "?")
        if lang not in candidates_by_lang:
            skipped_wrong_lang += 1
            continue
        query_date_ms = md.get("date_ms", 0)
        if not query_date_ms:
            skipped_no_metadata += 1
            continue
        if did not in graph:
            skipped_no_valid_gt += 1
            continue
        if not (facts_index.get(did) or "").strip():
            skipped_no_facts += 1
            continue
        # At least one valid successor
        has_valid = False
        for target, attrs in graph[did].items():
            if attrs.get("type") != "case_to_case":
                continue
            if target not in valid_ids:
                continue
            cited_date = metadata.get(target, {}).get("date_ms", 0)
            if cited_date and cited_date < query_date_ms:
                has_valid = True
                break
        if not has_valid:
            skipped_no_valid_gt += 1
            continue
        candidates_by_lang[lang].append(did)

    log.info(
        "Candidate pool: %s (skipped: %d no-metadata, %d wrong-lang, %d no-facts, %d no-valid-gt)",
        {lang: len(ids) for lang, ids in candidates_by_lang.items()},
        skipped_no_metadata, skipped_wrong_lang, skipped_no_facts, skipped_no_valid_gt,
    )

    # ── 5. Stratified sample (equal per language) ─────────────────────────
    rng = random.Random(RANDOM_SEED)
    sampled: list = []
    for lang in LANGUAGES:
        pool = candidates_by_lang[lang]
        rng.shuffle(pool)
        take = min(per_language_n, len(pool))
        sampled.extend(pool[:take])
        if take < per_language_n:
            log.warning("lang=%s: only %d available, wanted %d", lang, take, per_language_n)
        else:
            log.info("lang=%s: %d candidates → sampled %d", lang, len(pool), take)
    rng.shuffle(sampled)
    log.info("Sampled %d rulings total", len(sampled))

    # ── 6. Materialise each sampled ruling into a query record ────────────
    # Query text comes from the HF `facts` field (Sachverhalt only), already
    # loaded above in step 3b and pre-filtered out of the candidate pool, so
    # this loop should never hit an empty-facts entry.
    log.info("Building records ...")
    records: list = []
    for idx, did in enumerate(tqdm(sampled, desc="Building records", unit="q")):
        if cancel_check is not None and cancel_check():
            log.warning("Cancel requested — stopping record build at %d/%d",
                        idx, len(sampled))
            break
        md = metadata[did]
        query_date_ms = md["date_ms"]
        cited_rulings, cited_laws = build_ground_truth(
            graph, did, valid_ids, query_date_ms, metadata,
        )
        if not cited_rulings:
            continue  # defensive — should be guaranteed by step 4
        query_text = facts_index.get(did, "")
        if not query_text.strip():
            # Defensive: candidate-pool gate already filters facts-miss
            # entries, but a stale facts_index could still drop one here.
            continue
        records.append({
            "query_id": did,
            "query_text": truncate_to_tokens(query_text, FIRST_N_TOKENS),
            "ground_truth_cases": cited_rulings,
            "ground_truth_laws": cited_laws,
            "language": md["language"],
            "year": year_from_date_ms(query_date_ms),
            "date_ms": query_date_ms,
        })

        if on_event and ((idx + 1) % 25 == 0 or (idx + 1) == len(sampled)):
            on_event({"type": "progress", "stage": "sample",
                      "current": idx + 1, "total": len(sampled)})

    # ── 7. Write output ───────────────────────────────────────────────────
    log.info("Writing %d records to %s ...", len(records), OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Done.")

    # ── 8. Summary ────────────────────────────────────────────────────────
    lang_counts: dict = {}
    total_cited = 0
    total_laws = 0
    years: list = []
    for rec in records:
        lang_counts[rec["language"]] = lang_counts.get(rec["language"], 0) + 1
        total_cited += len(rec["ground_truth_cases"])
        total_laws += len(rec["ground_truth_laws"])
        if rec["year"]:
            years.append(rec["year"])

    print("\n" + "=" * 60)
    print("EVALUATION QUERY SUMMARY")
    print("=" * 60)
    print(f"Total queries:           {len(records)}")
    for lang, cnt in sorted(lang_counts.items()):
        share = 100 * cnt / len(records) if records else 0
        print(f"  {lang:4s}: {cnt:5d}  ({share:.1f}%)")
    if records:
        print(f"Avg cited rulings/query: {total_cited / len(records):.1f}")
        print(f"Avg cited laws/query:    {total_laws / len(records):.1f}")
    if years:
        print(f"Year range:              {min(years)}–{max(years)}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    emit({"type": "stage_done", "stage": "sample",
          "output": str(OUTPUT_FILE),
          "total": len(records),
          "languages": lang_counts})


# Back-compat alias for older callers
main = run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample evaluation queries (HF-free)")
    parser.add_argument("--n", type=int, default=PER_LANGUAGE_N,
                        help=f"Queries per language (default: {PER_LANGUAGE_N})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify inputs load and intersect; skip sampling")
    args = parser.parse_args()
    run(per_language_n=args.n, dry_run=args.dry_run)
