#!/usr/bin/env python3
"""
build_valid_ids.py
==================
Builds `data/eval/valid_ids.json`, the precomputed intersection of decision
IDs that exist in BOTH the Qdrant vector index AND the citation graph as
ruling nodes (V ∩ G in the thesis notation).

This artefact is the operational implementation of the strict GT consistency
filter described in chapter 3 of the thesis. Stage 1 (`01_sample_queries.py`)
samples query candidates exclusively from this list and filters every
ground-truth target through it again, so that every cited_ruling we evaluate
against is guaranteed to be retrievable.

Inputs:
  - Qdrant collection `bger` (sources `swiss_rulings_chunked` and
    `swiss_leading_decisions_chunked`), scrolled with `chunk_index=0` so
    each decision contributes exactly one point.
  - `data/graph/citation_graph.pkl` (NetworkX DiGraph from
    `build_citation_graph.py`).

Output:
  - `data/eval/valid_ids.json`, a sorted JSON list of UUIDv4 strings.
  - Stats printed to stdout (Qdrant ruling count, graph ruling count,
    intersection count).

Usage:
    QDRANT_HOST=localhost QDRANT_PORT=6333 \\
        .venv/bin/python scripts/eval/build_valid_ids.py

Historical note on the committed `data/eval/valid_ids.json` (131'125 IDs):
    A recompute audit run on 2026-05-18 showed a symmetric drift of roughly
    4-5K IDs against this script's output. Root cause was an earlier build
    that used a THREE-source whitelist including the now-unused
    `missing_bge_chunked` source, which provided ~4'300 decisions that are
    indexed ONLY under that source. The current Stage 2 retrieval
    (`02_run_retrieval.py`) filters to `swiss_rulings_chunked` and
    `swiss_leading_decisions_chunked` only, which is why this script
    matches that whitelist. Decisions present only under
    `missing_bge_chunked` therefore remain in the committed `valid_ids.json`
    but are not retrievable, the resulting recall under-estimation is
    documented in chapter 6 of the thesis. To preserve the canonical Run
    #66 (10 May 2026), the committed artefact is NOT overwritten by this
    script's output, run with a custom `VALID_IDS_PATH` if you want to
    inspect a fresh build.
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "bger"

RULINGS_SOURCE = "swiss_rulings_chunked"
LEADING_SOURCE = "swiss_leading_decisions_chunked"

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = Path(os.environ.get(
    "CITATION_GRAPH_PATH", str(REPO_ROOT / "data" / "eval" / "citation_graph.pkl")
))
OUTPUT_PATH = Path(os.environ.get(
    "VALID_IDS_PATH", str(REPO_ROOT / "data" / "eval" / "valid_ids.json")
))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("build_valid_ids")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def scroll_qdrant_decision_ids(client: QdrantClient) -> set:
    """Scroll all chunk_index=0 points of the two ruling sources, collect
    distinct decision_ids. One point per decision is enough to enumerate IDs.

    Two micro-optimisations keep the scroll fast and disconnect-robust on a
    656k-point collection. `with_payload=["decision_id"]` fetches only the
    one field we read, which cuts response size roughly 10× compared to the
    full payload. `limit=10000` reduces round-trip count to a few dozen
    requests. On a `ResponseHandlingException` (typically a server-side
    disconnect during a slow batch) the loop retries the current offset up
    to 5 times with exponential back-off before giving up.
    """
    sources = [RULINGS_SOURCE, LEADING_SOURCE]
    must_filters = [
        FieldCondition(key="chunk_index", match=MatchValue(value=0)),
        Filter(should=[
            FieldCondition(key="source", match=MatchValue(value=s))
            for s in sources
        ]),
    ]
    ids: set = set()
    offset = None
    with tqdm(desc="Qdrant scroll", unit="pts") as pbar:
        while True:
            for attempt in range(5):
                try:
                    batch, offset = client.scroll(
                        collection_name=QDRANT_COLLECTION,
                        limit=10000,
                        offset=offset,
                        with_payload=["decision_id"],
                        with_vectors=False,
                        scroll_filter=Filter(must=must_filters),
                    )
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    log.warning("scroll failed (attempt %d): %s — retry in %ds",
                                attempt + 1, e, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError("Qdrant scroll failed after 5 retries")
            for pt in batch:
                did = pt.payload.get("decision_id")
                if did:
                    ids.add(did)
            pbar.update(len(batch))
            if offset is None:
                break
    return ids


def load_graph_ruling_ids(graph_path: Path) -> set:
    """Return the set of node IDs with attribute source='ruling'."""
    log.info("Loading citation graph from %s ...", graph_path)
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    return {n for n, a in graph.nodes(data=True) if a.get("source") == "ruling"}


def main() -> int:
    if not GRAPH_PATH.exists():
        log.error("Citation graph not found at %s", GRAPH_PATH)
        log.error("Run scripts/eval/build_citation_graph.py first.")
        return 2

    log.info("Connecting to Qdrant at %s:%d ...", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=600)

    t0 = time.time()
    qdrant_ids = scroll_qdrant_decision_ids(client)
    log.info("Qdrant ruling decisions: %d (%.1fs)", len(qdrant_ids), time.time() - t0)

    t0 = time.time()
    graph_ids = load_graph_ruling_ids(GRAPH_PATH)
    log.info("Graph ruling nodes:      %d (%.1fs)", len(graph_ids), time.time() - t0)

    valid_ids = qdrant_ids & graph_ids
    log.info("Intersection V ∩ G:      %d", len(valid_ids))
    log.info("  only in Qdrant:        %d", len(qdrant_ids - graph_ids))
    log.info("  only in graph:         %d", len(graph_ids - qdrant_ids))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(valid_ids)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sorted_ids, f)
    log.info("Wrote %s (%d IDs)", OUTPUT_PATH, len(sorted_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
