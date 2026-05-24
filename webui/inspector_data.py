"""Lazy data sources for the per-query inspector.

The inspector page needs live access to:
  - Qdrant chunks (decision metadata + Sachverhalt extract for tooltips)
  - The citation graph (indegree + edges between candidates)

Both are loaded once on first use and cached for the process lifetime.
The graph is ~160k nodes / 1.6M edges — fits in memory comfortably.
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny


_QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "bger")
_GRAPH_PATH = Path(os.environ.get("EVAL_DIR", "/app/data/eval")) / "citation_graph.pkl"


@lru_cache(maxsize=1)
def qdrant() -> QdrantClient:
    # Read env at call time so /settings updates take effect on next call
    # after qdrant.cache_clear().
    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        check_compatibility=False, timeout=30,
    )


@lru_cache(maxsize=1)
def graph():
    with open(_GRAPH_PATH, "rb") as f:
        return pickle.load(f)


# Process-wide cache for chunk-0 metadata lookups. The bger Qdrant
# collection is read-only at runtime, so cached entries never go stale
# during a pod's lifetime. Each entry is ~700 B (mostly the 600-char
# text excerpt). preload_all_metadata() at boot scrolls every
# chunk_index=0 point and fills the cache; after that, every
# metadata_for() is a dict hit, no Qdrant round-trip, completely
# insulating the inspector from pipeline-induced Qdrant load.
#
# No cap: the corpus is finite and read-only. Empirical attempts at
# 200k and 500k both filled completely; we don't actually know the
# upper bound (count() against Qdrant times out under pipeline load).
# Memory ceiling at 1 M entries is ~700 MB, comfortably within the
# pod's 6 GiB limit. The cap field is kept as a safety guard but set
# to a number larger than any plausible corpus.
_METADATA_CACHE: dict[str, dict] = {}
_METADATA_CACHE_MAX = 2_000_000


def metadata_for(decision_ids: Iterable[str]) -> dict[str, dict]:
    """Bulk fetch metadata + chunk-0 text for a set of decision_ids.

    Returns `{decision_id: {file_number, date, date_ms, language, court,
    source, text}}`. Only chunk_index=0 is fetched (the Sachverhalt
    opening). Missing decisions are simply absent from the dict.

    Pagination note: Qdrant's `scroll` returns a `next_page_offset` even
    when the filter would have matched fewer points than the page limit
    in total, so a single call with `limit=len(batch)` can drop rows on
    the floor. We loop until offset is None.
    """
    ids = {d for d in decision_ids if d}
    if not ids:
        return {}

    out: dict[str, dict] = {}
    misses: list[str] = []
    for did in ids:
        cached = _METADATA_CACHE.get(did)
        if cached is not None:
            out[did] = cached
        else:
            misses.append(did)
    if not misses:
        return out

    client = qdrant()
    BATCH = 500
    PAGE = 256
    payload_keys = ["decision_id", "file_number", "date", "date_ms",
                    "language", "court", "source", "text"]
    for i in range(0, len(misses), BATCH):
        batch = misses[i:i+BATCH]
        next_off = None
        while True:
            scrolled, next_off = client.scroll(
                collection_name=_QDRANT_COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="decision_id", match=MatchAny(any=batch)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                ]),
                limit=PAGE,
                offset=next_off,
                with_payload=payload_keys,
                with_vectors=False,
            )
            for p in scrolled:
                did = p.payload.get("decision_id")
                if not did or did in out:
                    continue
                rec = {
                    "file_number": p.payload.get("file_number"),
                    "date": p.payload.get("date"),
                    "date_ms": p.payload.get("date_ms"),
                    "language": p.payload.get("language"),
                    "court": p.payload.get("court"),
                    "source": p.payload.get("source"),
                    "text": (p.payload.get("text") or "")[:600],
                }
                out[did] = rec
                if len(_METADATA_CACHE) < _METADATA_CACHE_MAX:
                    _METADATA_CACHE[did] = rec
            if not next_off:
                break
    return out


def preload_all_metadata(log=None) -> int:
    """Scroll every chunk_index=0 point in the bger collection and fill
    `_METADATA_CACHE`. Call once at app boot in a daemon thread so all
    subsequent metadata_for() lookups are dict hits instead of Qdrant
    round-trips. ~60 k entries, ~5 min at typical Qdrant speed.

    Idempotent: re-running just refills any entries that got evicted.
    Safe to interleave with live metadata_for() calls (entries are
    populated key-by-key, no big-bang replace).

    Returns the number of cache entries after the run.
    """
    client = qdrant()
    PAGE = 1024  # bigger pages amortise per-request overhead
    payload_keys = ["decision_id", "file_number", "date", "date_ms",
                    "language", "court", "source", "text"]
    next_off = None
    pages = 0
    added = 0
    while True:
        try:
            scrolled, next_off = client.scroll(
                collection_name=_QDRANT_COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                ]),
                limit=PAGE,
                offset=next_off,
                with_payload=payload_keys,
                with_vectors=False,
            )
        except Exception as e:
            if log:
                log(f"preload_all_metadata: scroll failed at page {pages}: {e}")
            return len(_METADATA_CACHE)
        for p in scrolled:
            did = p.payload.get("decision_id")
            if not did or did in _METADATA_CACHE:
                continue
            if len(_METADATA_CACHE) >= _METADATA_CACHE_MAX:
                if log:
                    log(f"preload_all_metadata: cache cap {_METADATA_CACHE_MAX} reached")
                return len(_METADATA_CACHE)
            _METADATA_CACHE[did] = {
                "file_number": p.payload.get("file_number"),
                "date": p.payload.get("date"),
                "date_ms": p.payload.get("date_ms"),
                "language": p.payload.get("language"),
                "court": p.payload.get("court"),
                "source": p.payload.get("source"),
                "text": (p.payload.get("text") or "")[:600],
            }
            added += 1
        pages += 1
        if pages % 20 == 0 and log:
            log(f"preload_all_metadata: {pages} pages, {len(_METADATA_CACHE)} entries cached")
        if not next_off:
            break
    if log:
        log(f"preload_all_metadata: done — {pages} pages, {len(_METADATA_CACHE)} entries (added {added})")
    return len(_METADATA_CACHE)


def in_degree(decision_id: str) -> int:
    g = graph()
    if decision_id in g:
        return g.in_degree(decision_id)
    return 0


def out_degree(decision_id: str) -> int:
    g = graph()
    if decision_id in g:
        return g.out_degree(decision_id)
    return 0


def edges_within(decision_ids: Iterable[str]) -> list[tuple[str, str]]:
    """Return citation edges where both endpoints are in `decision_ids`.

    Edge direction in the graph is citing → cited. Returns the same
    direction so the front-end can draw arrows.
    """
    ids = set(decision_ids)
    if not ids:
        return []
    g = graph()
    edges: list[tuple[str, str]] = []
    # Iterate the smaller side: from each node in `ids`, walk successors.
    for node in ids:
        if node not in g:
            continue
        for nbr in g.successors(node):
            if nbr in ids:
                edges.append((node, nbr))
    return edges


def gt_indexed_status(gt_ids: list[str]) -> dict[str, bool]:
    """For each GT id, return whether at least one chunk exists in Qdrant.

    Uses per-id `count(exact=True)` calls because they hit the indexed
    `decision_id` field directly (~30 ms per id on the local cluster).
    A previous batched-scroll implementation produced false negatives —
    its `limit=len(batch)*5` under-provisioned for the empirical chunk
    distribution (median 7, p95 18, max 27 chunks per decision) and it
    discarded the scroll's `next_page_offset`, so decisions whose chunks
    landed on page 2+ were silently flagged as missing. The strict GT
    consistency filter at sample time guarantees every GT is actually
    present, so any "missing" flag from this function must be true.
    """
    client = qdrant()
    out: dict[str, bool] = {}
    for gid in gt_ids:
        if not gid:
            continue
        n = client.count(
            collection_name=_QDRANT_COLLECTION,
            count_filter=Filter(must=[
                FieldCondition(key="decision_id", match=MatchValue(value=gid))
            ]),
            exact=True,
        ).count
        out[gid] = n > 0
    return out
