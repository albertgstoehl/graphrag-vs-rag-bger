#!/usr/bin/env python3
"""
02_run_retrieval.py — Run all retrieval configurations on the evaluation queries.

PURPOSE:
    Execute 9 retrieval configurations (3 systems × 3 ranking strategies) for
    each of the 4 k values (5, 10, 15, 20), writing ranked result lists to disk.
    This is the expensive step — it embeds every query via BGE-M3 and runs
    Qdrant ANN search plus optional graph expansion.

SYSTEMS:
    rag        — Qdrant top-k cosine only, no graph
    emb_1hop   — Qdrant top-60 seed → kNN expansion in embedding space (1-hop)
    emb_2hop   — Qdrant top-60 seed → 1-hop kNN → 2-hop kNN in embedding space
    graph_1hop — Qdrant top-60 seed → add direct citation neighbours (1-hop)
    graph_2hop — Qdrant top-60 seed → 1-hop → 2-hop neighbours

RANKING STRATEGIES:
    cosine      — Qdrant cosine similarity score (descending)
    cross_encoder — Re-score with BAAI/bge-reranker-v2-m3 (multilingual)
    indegree    — Score = cosine × log(1 + in_degree) in citation graph

K VALUES: 5, 10, 20

OUTPUT:
    /data/thesis/eval/results/{system}_{ranking}_{k}.jsonl
    Each line: {"query_id": "...", "retrieved": ["id1", "id2", ...]}

USAGE:
    python3 02_run_retrieval.py
    python3 02_run_retrieval.py --dry-run   # test 2 queries, skip cross-encoder
    python3 02_run_retrieval.py --systems rag graph_1hop   # subset of systems
"""

import os
import sys
import json
import math
import logging
import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

import requests

# SentenceTransformer is used as a fallback when no TEI server is running.
# It requires torch + GPU — run this script with /data/vllm/venv/bin/python3.
try:
    from sentence_transformers import SentenceTransformer as _ST
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# ── Constants ─────────────────────────────────────────────────────────────────

# Paths — override via env vars for local dev
EVAL_DIR = Path(os.environ.get("EVAL_DIR", "data/eval"))
QUERIES_FILE = EVAL_DIR / "eval_queries.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
GRAPH_PATH = Path(os.environ.get("GRAPH_PATH", str(EVAL_DIR / "citation_graph.pkl")))

# Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "aiserver01")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "bger"
RULINGS_SOURCE = "swiss_rulings_chunked"
LEADING_SOURCE = "swiss_leading_decisions_chunked"
LEGISLATION_SOURCE = "swiss_legislation_chunked"

# TEI embedding server — the BGE-M3 text-embeddings-inference service.
# One instance deployed on aiserver01 via k8s (hostPort 8010).
# Override via TEI_HOST / TEI_PORTS env vars if needed.
TEI_HOST = os.environ.get("TEI_HOST", "aiserver01")
TEI_PORTS = [int(p) for p in os.environ.get("TEI_PORTS", "8010").split(",")]

# Cross-encoder reranker — served as TEI services on aiserver01.
# TEI exposes a /rerank endpoint that scores (query, texts) pairs in one call.
# Multiple replicas can be deployed (one per GPU); the pipeline distributes
# batches across all configured endpoints in parallel.
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_BATCH = int(os.environ.get("CROSS_ENCODER_BATCH", 32))
# Per-candidate text cap (chars) before sending to TEI /rerank. Outlier chunk-0
# texts can exceed the reranker's 8192-BPE pair limit even at batch_size=1,
# yielding 413 Payload Too Large. ~4000 chars ≈ ~1000 BPE, leaving headroom
# for the 4096-ws-token query side of the pair. Affected 10/12'678 queries in
# run #89.
CROSS_ENCODER_TEXT_CHARS = int(os.environ.get("CROSS_ENCODER_TEXT_CHARS", 4000))
TEI_RERANK_HOST = os.environ.get("TEI_RERANK_HOST", "aiserver01:8011")
TEI_RERANK_URL = (
    os.environ.get("TEI_RERANK_URL")
    or f"http://{TEI_RERANK_HOST}/rerank"
)
# Optional: comma-separated list overrides the single URL above. Empty/unset
# falls back to TEI_RERANK_URL only (keeps the single-replica path working).
TEI_RERANK_URLS = [
    u.strip() for u in os.environ.get("TEI_RERANK_URLS", "").split(",")
    if u.strip()
] or [TEI_RERANK_URL]

# Retrieval parameters
SEED_K = 60        # number of Qdrant ANN results to use as graph expansion seeds
K_VALUES = [5, 10, 15, 20]

# Graph expansion limits (cap candidate set before reranking to keep it manageable)
MAX_1HOP_CANDIDATES = 400
MAX_2HOP_CANDIDATES = 800

# Global SentenceTransformer model (loaded lazily when TEI is unavailable)
_ST_MODEL = None

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy HTTP client loggers — they emit one INFO line per Qdrant
# request which floods the SSE event broker and the run log.
for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ── Retry helper ──────────────────────────────────────────────────────────────
# aiserver01's TEI/Qdrant pods restart frequently (observed 400+ TEI restarts
# per day). A single transient ConnectionRefused or timeout would otherwise
# kill a multi-hour run. This wrapper retries on common transient errors with
# exponential backoff. Permanent errors (4xx HTTP, ValidationError) still
# propagate immediately.

import functools

_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
try:
    from qdrant_client.http.exceptions import ResponseHandlingException
    _RETRY_EXCEPTIONS = _RETRY_EXCEPTIONS + (ResponseHandlingException,)
except ImportError:
    pass


class CrossEncoderBatchFailure(RuntimeError):
    """One or more rerank batches failed permanently after all retries.

    Raised by `rank_by_cross_encoder` when at least one fan-out batch
    exhausts its retry budget. The per-query loop catches this and skips
    the CE outputs for that query, leaving the row missing from
    cross_encoder_scores.jsonl so the resume mechanism re-processes it
    on the next run. Replaces the previous behaviour of silently writing
    score=0.0 for the failed candidates.
    """
    pass


def with_retry(label: str, attempts: int = 4, base_delay: float = 1.0):
    """Decorator: retry on transient connection / timeout errors.

    Sleeps base_delay * 2**(i-1) before each retry — 1s, 2s, 4s, 8s ...
    Default 4 attempts → tolerates ~15s of service downtime.
    """
    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except _RETRY_EXCEPTIONS as e:
                    last_exc = e
                    if attempt == attempts:
                        log.error("%s: giving up after %d attempts (%s)",
                                  label, attempts, e)
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    log.warning("%s: attempt %d/%d failed (%s), retrying in %.1fs",
                                label, attempt, attempts, e, delay)
                    time.sleep(delay)
            raise last_exc  # unreachable
        return _wrapped
    return _decorator


# ── TEI Embedding ─────────────────────────────────────────────────────────────

def find_live_tei_endpoint() -> Optional[str]:
    """Probe TEI endpoints and return the URL of the first healthy one.

    The TEI servers expose a GET /health endpoint that returns 200 when ready.
    We try each configured port and return the first that responds.

    Returns:
        Base URL string like "http://aiserver01:8010", or None if none are alive.
    """
    for port in TEI_PORTS:
        url = f"http://{TEI_HOST}:{port}"
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                log.info("TEI endpoint found: %s", url)
                return url
        except requests.RequestException:
            pass
    return None


@with_retry("embed_texts")
def embed_texts(texts: list, endpoint: Optional[str]) -> list:
    """Embed a list of texts using BGE-M3.

    Tries TEI HTTP server first (fast, batched GPU). Falls back to loading
    SentenceTransformer directly if no TEI server is available.

    Args:
        texts:    list of strings to embed (keep each under ~8192 tokens)
        endpoint: base URL of the TEI server (e.g. "http://localhost:8010"),
                  or None to use direct SentenceTransformer loading.

    Returns:
        list of float vectors (list[list[float]]), one per input text
    """
    global _ST_MODEL

    if endpoint is not None:
        # TEI path: fast HTTP embedding server
        payload = {"inputs": texts}
        r = requests.post(f"{endpoint}/embed", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()  # list of list[float]

    # Fallback: load BGE-M3 directly via SentenceTransformer
    # Requires sentence_transformers + torch in the current Python environment.
    if not _ST_AVAILABLE:
        raise RuntimeError(
            "No TEI server found AND sentence_transformers is not installed. "
            "Run this script with /data/vllm/venv/bin/python3 which has torch."
        )
    if _ST_MODEL is None:
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,3,4,5,6,7,8,9,10,11")
        os.environ.setdefault("HF_HOME", "/data/thesis/hf-cache")
        log.info("Loading BGE-M3 model directly (no TEI server) ...")
        _ST_MODEL = _ST("BAAI/bge-m3", device="cuda")
        log.info("BGE-M3 loaded on CUDA.")

    # SentenceTransformer returns numpy arrays; convert to plain lists for consistency
    vecs = _ST_MODEL.encode(texts, normalize_embeddings=True, batch_size=32)
    return vecs.tolist()


# ── Qdrant Retrieval ──────────────────────────────────────────────────────────

@with_retry("qdrant_search")
def qdrant_search(
    client: QdrantClient,
    query_vector: list,
    sources: list,
    limit: int,
    exclude_decision_id: Optional[str] = None,
    query_date_ms: int = 0,
    presort: str = "cosine",
    graph=None,
) -> list:
    """Search Qdrant for the top-`limit` chunks matching `query_vector`.

    Filters by source to restrict to rulings only (not legislation).
    Results are deduplicated by decision_id — for each document we keep
    only the chunk with the highest cosine score.

    `exclude_decision_id`, `query_date_ms`: see existing semantics —
    server-side filters preventing self-leakage and future-dated leaks.

    `presort`:
        "cosine"   (default) — final truncation by descending cosine score.
        "indegree" — fetch a wider raw set, then truncate by descending
                     `graph.in_degree(decision_id)`. Selects seeds that are
                     both semantically similar to the query AND high in
                     citation-authority. Requires `graph`. Used by the
                     `rag_smart` baseline to test whether smart pre-filtering
                     alone closes the GraphRAG-1Hop gap.
        The `score` field in returned dicts always carries the cosine
        score (not indegree) so downstream cosine ranking remains
        meaningful for `rag_smart` candidates.

    Returns:
        list of dicts: [{"decision_id": str, "score": float}, ...]
        length <= limit (unique decision_ids)
    """
    if presort not in ("cosine", "indegree"):
        raise ValueError(f"unknown presort {presort!r}")
    if presort == "indegree" and graph is None:
        raise ValueError("presort='indegree' requires graph")

    # Wider over-fetch when indegree-presorting so the indegree truncation
    # has a meaningful pool to choose from. Plain cosine path keeps its
    # historical 3× / 12× over-fetch.
    if presort == "indegree":
        raw_limit = limit * 8 if not query_date_ms else limit * 24
    else:
        raw_limit = limit * (12 if query_date_ms else 3)

    must_not = []
    if exclude_decision_id:
        must_not.append(
            FieldCondition(key="decision_id", match=MatchValue(value=exclude_decision_id))
        )
    must = []
    if query_date_ms:
        must.append(FieldCondition(key="date_ms", range=Range(lt=query_date_ms)))
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=raw_limit,
        query_filter=Filter(
            should=[
                FieldCondition(key="source", match=MatchValue(value=s))
                for s in sources
            ],
            must=must or None,
            must_not=must_not or None,
        ),
        with_payload=True,
        with_vectors=False,
    )
    results = response.points

    seen: dict = {}
    for hit in results:
        did = hit.payload.get("decision_id")
        if not did:
            continue
        score = hit.score
        if did not in seen or score > seen[did]:
            seen[did] = score

    if presort == "indegree":
        # Sort by indegree desc; tie-break by cosine desc to keep the most
        # semantically relevant of any equally-popular candidates.
        sorted_docs = sorted(
            seen.items(),
            key=lambda kv: (graph.in_degree(kv[0]) if kv[0] in graph else 0, kv[1]),
            reverse=True,
        )
    else:
        sorted_docs = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return [{"decision_id": did, "score": score} for did, score in sorted_docs[:limit]]


# ── Graph Expansion ───────────────────────────────────────────────────────────

def expand_graph_1hop(
    seed_ids: list, graph, max_candidates: int,
    query_date_ms: int = 0, date_index: dict = None,
    trace_sink: list = None,
) -> set:
    """Return the set of decision_ids reachable in 1 hop from any seed node.

    Applies the same temporal closed-world assumption as Qdrant search:
    candidates with a known date >= query_date_ms are excluded.

    If `trace_sink` is given, it is appended with `(stage_name, frozenset)`
    tuples at each pipeline stage (raw / post_temporal / post_cap) — this
    powers the per-layer Recall-Ceiling analysis.
    """
    date_index = date_index or {}
    neighbours: set = set()
    for did in seed_ids:
        if did not in graph:
            continue
        neighbours.update(graph.successors(did))
        neighbours.update(graph.predecessors(did))
    # Filter out law nodes: case_to_law successors add `source="law"` nodes
    # to the candidate pool, but the evaluation ground truth is `cited_rulings`
    # only. Without this filter, indegree-ranking on graph-expansion candidates
    # gets dominated by highly-cited law articles (76-82% of top-k otherwise).
    neighbours = {n for n in neighbours if graph.nodes[n].get("source") == "ruling"}
    neighbours -= set(seed_ids)
    if trace_sink is not None:
        trace_sink.append(("raw", frozenset(neighbours)))
    # Temporal filter: exclude documents dated >= query_date_ms
    if query_date_ms:
        neighbours = {
            n for n in neighbours
            if not (date_index.get(n, 0) and date_index[n] >= query_date_ms)
        }
    if trace_sink is not None:
        trace_sink.append(("post_temporal", frozenset(neighbours)))
    if len(neighbours) > max_candidates:
        neighbours = set(
            # Deterministic tiebreak by decision_id: in_degree ties (very common
            # at 0/1/2) would otherwise be broken by set iteration order, which
            # is randomised per-process via PYTHONHASHSEED. That made the
            # graph-system pool composition non-reproducible across runs.
            sorted(neighbours, key=lambda n: (-graph.in_degree(n), n))
            [:max_candidates]
        )
    if trace_sink is not None:
        trace_sink.append(("post_cap", frozenset(neighbours)))
    return neighbours


def expand_graph_2hop(
    seed_ids: list, graph, max_candidates: int,
    query_date_ms: int = 0, date_index: dict = None,
    trace_sink: list = None,
) -> set:
    """Return decision_ids reachable in exactly 1 or 2 hops from seeds.

    Applies the same temporal closed-world assumption as Qdrant search.

    If `trace_sink` is given, stage snapshots (raw / post_temporal / post_cap)
    are appended at the outer 2-hop level (not the inner 1-hop call) for the
    per-layer Recall-Ceiling analysis.
    """
    date_index = date_index or {}
    one_hop = expand_graph_1hop(
        seed_ids, graph, max_candidates=max_candidates * 2,
        query_date_ms=query_date_ms, date_index=date_index,
        trace_sink=None,
    )

    two_hop: set = set()
    for did in one_hop:
        if did not in graph:
            continue
        two_hop.update(graph.successors(did))
        two_hop.update(graph.predecessors(did))
    # Filter out law nodes (case_to_law successors). Same rationale as in
    # expand_graph_1hop: ground truth is `cited_rulings` only, law nodes pollute
    # the candidate pool and dominate indegree-ranking otherwise.
    two_hop = {n for n in two_hop if graph.nodes[n].get("source") == "ruling"}

    all_neighbours = one_hop | two_hop
    all_neighbours -= set(seed_ids)
    if trace_sink is not None:
        trace_sink.append(("raw", frozenset(all_neighbours)))
    # Temporal filter
    if query_date_ms:
        all_neighbours = {
            n for n in all_neighbours
            if not (date_index.get(n, 0) and date_index[n] >= query_date_ms)
        }
    if trace_sink is not None:
        trace_sink.append(("post_temporal", frozenset(all_neighbours)))
    if len(all_neighbours) > max_candidates:
        all_neighbours = set(
            # Deterministic tiebreak by decision_id (see expand_graph_1hop comment).
            sorted(all_neighbours, key=lambda n: (-graph.in_degree(n), n))
            [:max_candidates]
        )
    if trace_sink is not None:
        trace_sink.append(("post_cap", frozenset(all_neighbours)))
    return all_neighbours


# ── Embedding-Space Expansion ────────────────────────────────────────────────

def _knn_neighbours_of_ids(
    source_ids: list,
    client: QdrantClient,
    per_id_k: int,
    exclude_ids: set,
) -> dict:
    """For each decision_id, fetch its chunk_index=0 vector and query kNN.

    Vector fetch and kNN search are both parallelised across source_ids via
    a thread pool. Qdrant handles concurrent requests internally with its
    own search worker pool, so on a healthy server the wall-clock drops
    roughly linearly with the worker count up to that pool's capacity.

    Returns:
        dict mapping decision_id → best cosine score (max over seeds)
    """
    if not source_ids:
        return {}

    from concurrent.futures import ThreadPoolExecutor

    # Step 1: fetch the chunk_index=0 vector for every source_id concurrently.
    def _fetch_vec(did):
        try:
            pts = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="decision_id", match=MatchValue(value=did)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                ]),
                limit=1,
                with_vectors=True,
                with_payload=["decision_id"],
            )[0]
            if pts and pts[0].vector:
                return did, pts[0].vector
        except Exception as e:
            log.debug("emb seed-vec fetch failed for %s: %s", did, e)
        return did, None

    # Step 2: run kNN for every fetched vector concurrently. The kNN call
    # filters server-side to the two ruling sources so legislation chunks
    # (~325k points under `swiss_legislation_chunked`) cannot pollute the
    # embedding-expansion pool. Without this filter the embedding variants
    # had an asymmetric pool composition compared to the graph variants
    # (which filter to `source="ruling"` after expansion).
    knn_source_filter = Filter(should=[
        FieldCondition(key="source", match=MatchValue(value=RULINGS_SOURCE)),
        FieldCondition(key="source", match=MatchValue(value=LEADING_SOURCE)),
    ])

    def _knn(item):
        did, vec = item
        if vec is None:
            return []
        try:
            response = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vec,
                limit=per_id_k * 3,
                with_payload=["decision_id", "chunk_index"],
                with_vectors=False,
                query_filter=knn_source_filter,
            )
            return list(response.points)
        except Exception as e:
            log.debug("emb kNN failed for %s: %s", did, e)
            return []

    workers = min(16, max(2, len(source_ids)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        seed_vecs = list(ex.map(_fetch_vec, source_ids))
        knn_results = list(ex.map(_knn, seed_vecs))

    neighbours: dict = {}
    for points in knn_results:
        for pt in points:
            ndid = pt.payload.get("decision_id")
            if not ndid or ndid in exclude_ids:
                continue
            score = pt.score
            if ndid not in neighbours or score > neighbours[ndid]:
                neighbours[ndid] = score

    return neighbours


def expand_emb_1hop(
    seed_docs: list,
    client: QdrantClient,
    max_candidates: int,
    query_date_ms: int = 0,
    date_index: dict = None,
    trace_sink: list = None,
    query_id: Optional[str] = None,
) -> set:
    """Return decision_ids reachable via kNN in embedding space from seeds.

    Structural analogue to expand_graph_1hop but uses embedding proximity
    instead of citation links. Supports `trace_sink` for the per-layer
    Recall-Ceiling analysis (raw / post_temporal / post_cap).

    `query_id` is excluded from kNN neighbours explicitly (defence in
    depth — the temporal filter would also catch it, but only if the
    query's own date is in `date_index`).
    """
    date_index = date_index or {}
    seed_ids = {d["decision_id"] for d in seed_docs}
    exclude_ids = seed_ids | ({query_id} if query_id else set())
    per_seed_k = max(10, max_candidates // len(seed_docs)) if seed_docs else 10

    neighbours = _knn_neighbours_of_ids(
        [d["decision_id"] for d in seed_docs], client, per_seed_k, exclude_ids
    )
    if trace_sink is not None:
        trace_sink.append(("raw", frozenset(neighbours.keys())))

    if query_date_ms:
        neighbours = {
            did: sc for did, sc in neighbours.items()
            if not (date_index.get(did, 0) and date_index[did] >= query_date_ms)
        }
    if trace_sink is not None:
        trace_sink.append(("post_temporal", frozenset(neighbours.keys())))

    if len(neighbours) > max_candidates:
        sorted_n = sorted(neighbours.items(), key=lambda x: x[1], reverse=True)
        neighbours = dict(sorted_n[:max_candidates])
    if trace_sink is not None:
        trace_sink.append(("post_cap", frozenset(neighbours.keys())))

    return set(neighbours.keys())


def expand_emb_2hop(
    seed_docs: list,
    one_hop_ids: set,
    client: QdrantClient,
    max_candidates: int,
    query_date_ms: int = 0,
    date_index: dict = None,
    trace_sink: list = None,
    query_id: Optional[str] = None,
) -> set:
    """Return decision_ids reachable in 1 or 2 embedding-space hops from seeds.

    Structural analogue to expand_graph_2hop in the citation graph. Takes the
    already-computed 1-hop neighbours and expands once more: for each
    1-hop neighbour, find ITS kNN in Qdrant. Merge with 1-hop candidates.

    `query_id` is added to the kNN exclude-set so the query never
    re-enters its own 2-hop pool, even if it would pass the temporal
    filter due to a missing date_index entry.

    Args:
        seed_docs:      original ANN seeds
        one_hop_ids:    result of expand_emb_1hop (reused, not recomputed)
        client:         QdrantClient
        max_candidates: cap on final set size

    Returns:
        set of decision_id strings (includes 1-hop + 2-hop, excludes seeds)
    """
    date_index = date_index or {}
    seed_ids = {d["decision_id"] for d in seed_docs}
    exclude = seed_ids | ({query_id} if query_id else set())

    # Sub-sample 1-hop to keep query count tractable. Sort first so the
    # selection is deterministic — `set` iteration order depends on the
    # hash seed and would otherwise jitter the 2-hop ceiling between runs.
    expansion_front = sorted(one_hop_ids)[: min(40, len(one_hop_ids))]
    per_node_k = max(10, max_candidates // max(1, len(expansion_front)))

    second_hop = _knn_neighbours_of_ids(
        expansion_front, client, per_node_k, exclude
    )

    # Merge 1-hop and 2-hop into one candidate pool
    combined: dict = {did: 1.0 for did in one_hop_ids}
    for did, sc in second_hop.items():
        if did not in combined or sc > combined[did]:
            combined[did] = sc
    if trace_sink is not None:
        trace_sink.append(("raw", frozenset(combined.keys())))

    if query_date_ms:
        combined = {
            did: sc for did, sc in combined.items()
            if not (date_index.get(did, 0) and date_index[did] >= query_date_ms)
        }
    if trace_sink is not None:
        trace_sink.append(("post_temporal", frozenset(combined.keys())))

    if len(combined) > max_candidates:
        sorted_c = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        combined = dict(sorted_c[:max_candidates])
    if trace_sink is not None:
        trace_sink.append(("post_cap", frozenset(combined.keys())))

    return set(combined.keys())


# ── Ranking Strategies ────────────────────────────────────────────────────────

def rank_by_cosine(candidates: list) -> list:
    """Sort candidate list by cosine score descending.

    For graph-expanded candidates that have no direct cosine score, we
    assign a score of 0.0 (they were added by graph topology, not ANN).

    Args:
        candidates: list of {"decision_id": str, "score": float}

    Returns:
        list sorted by score descending
    """
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def rank_by_indegree(candidates: list, graph) -> list:
    """Score = log(1 + in_degree), sort descending.

    Ranks candidates purely by their citation authority in the graph.
    Documents with high in-degree (frequently cited) rank higher,
    regardless of their cosine similarity to the query. This measures
    whether a system finds more authoritative documents, independent
    of semantic similarity.

    The log dampens the effect of very highly cited nodes (hubs) to prevent
    them from dominating the ranking.

    Args:
        candidates: list of {"decision_id": str, "score": float}
        graph:      networkx DiGraph for in-degree lookup

    Returns:
        list sorted by indegree score descending
    """
    scored = []
    for c in candidates:
        did = c["decision_id"]
        # in_degree = number of other docs that cite this doc
        indegree = graph.in_degree(did) if did in graph else 0
        scored.append({"decision_id": did, "score": math.log(1 + indegree)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def rank_by_cross_encoder(
    query_text: str,
    candidates: list,
    cross_encoder,
    batch_size: int = CROSS_ENCODER_BATCH,
) -> list:
    """Re-score candidates using TEI /rerank service(s), parallel across endpoints.

    `cross_encoder` is either a list of URLs (multi-replica deployment) or a
    single URL string (legacy / single-replica fallback). When multiple URLs
    are provided, batches are fanned out concurrently via a thread pool, each
    batch routed to the next URL in round-robin order. With N healthy
    endpoints this yields up to N× rerank throughput per query.

    Args:
        query_text:    the query string to pair with each candidate text
        candidates:    list of {"decision_id": str, "score": float, "text": str}
        cross_encoder: list[str] of /rerank URLs, str URL, or None to skip
        batch_size:    number of texts per HTTP request

    Returns:
        list of {"decision_id": str, "score": float} sorted by CE score desc
    """
    if cross_encoder is None:
        return sorted(
            [{"decision_id": c["decision_id"], "score": float(c.get("score", 0.0))}
             for c in candidates],
            key=lambda x: x["score"], reverse=True,
        )

    # Normalise to a URL list
    urls = cross_encoder if isinstance(cross_encoder, list) else [cross_encoder]

    # Build (url, offset, batch) work units — round-robin assignment so load
    # is balanced across replicas regardless of list order.
    work = []
    for i, start in enumerate(range(0, len(candidates), batch_size)):
        batch = candidates[start : start + batch_size]
        texts = [(c.get("text", "") or "")[:CROSS_ENCODER_TEXT_CHARS] for c in batch]
        url = urls[i % len(urls)]
        work.append((url, start, texts))

    all_scores: list = [0.0] * len(candidates)

    @with_retry("rerank batch")
    def _post(url, payload):
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    # `_call` returns (offset, data, error). If the retry-loop in `_post`
    # exhausts itself the failure used to be silently filled with 0.0s, which
    # produced cross_encoder_scores rows with all-zero candidates and silently
    # corrupted the dataset (run #39 incident, 2026-05-09). We now record the
    # error and let the caller decide whether to abort the query.
    def _call(unit):
        url, offset, texts = unit
        payload = {"query": query_text, "texts": texts, "raw_scores": False}
        try:
            return offset, _post(url, payload), None
        except Exception as e:
            log.warning("Rerank HTTP %s offset=%d failed permanently: %s",
                        url, offset, e)
            return offset, [], e

    # Fan out batches in parallel — one worker per replica is the right
    # concurrency level (more would just contend on the GPU).
    if len(urls) == 1:
        results = [_call(u) for u in work]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(urls)) as ex:
            results = list(ex.map(_call, work))

    failed = [(offset, err) for offset, data, err in results if err is not None]
    if failed:
        raise CrossEncoderBatchFailure(
            f"{len(failed)} of {len(work)} rerank batches failed permanently "
            f"(first offset={failed[0][0]}, error={failed[0][1]})"
        )

    for offset, data, _err in results:
        for item in data:
            idx = item["index"]
            all_scores[offset + idx] = float(item["score"])

    scored = [
        {"decision_id": c["decision_id"], "score": s}
        for c, s in zip(candidates, all_scores)
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)


# ── Candidate text fetcher ────────────────────────────────────────────────────

@with_retry("fetch_chunk_texts")
def fetch_chunk_texts(
    decision_ids: list,
    client: QdrantClient,
    cache: dict | None = None,
) -> dict:
    """Fetch the best (chunk_index=0) text for a list of decision_ids from Qdrant.

    For cross-encoder reranking we need the actual document text. We fetch
    the first chunk (chunk_index=0) as a representative text.

    If `cache` is provided, IDs already present are skipped and newly fetched
    texts are written back into the cache. This lets the caller share one
    text cache across the full run (cross-query cache hit rate is very high
    once the candidate seeds overlap, which they do as soon as a few queries
    have run).

    Returns:
        dict[decision_id -> text string]  (cache-backed when supplied)
    """
    texts: dict = cache if cache is not None else {}
    missing = [did for did in decision_ids if did not in texts]
    if not missing:
        return texts
    # Server-side filter on chunk_index=0 reduces traffic — only one row
    # per decision is returned.
    BATCH = 100
    for i in range(0, len(missing), BATCH):
        batch_ids = missing[i : i + BATCH]
        try:
            results, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=len(batch_ids) + 5,  # tight limit thanks to chunk_index filter
                with_payload=True,
                with_vectors=False,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_index", match=MatchValue(value=0)
                        ),
                        Filter(
                            should=[
                                FieldCondition(
                                    key="decision_id", match=MatchValue(value=did)
                                )
                                for did in batch_ids
                            ]
                        )
                    ]
                ),
            )
            for pt in results:
                did = pt.payload.get("decision_id")
                chunk_idx = pt.payload.get("chunk_index", 0)
                # Keep only the first chunk per document
                if did and did not in texts and chunk_idx == 0:
                    texts[did] = pt.payload.get("text", "")
        except Exception as e:
            log.warning("Text fetch failed for batch %d: %s", i, e)
    return texts


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def load_queries(path: Path) -> list:
    """Load evaluation queries from JSONL file.

    Queries with an empty `query_text` are dropped: TEI returns 413 on empty
    input and there's nothing to embed anyway. Empty rows are a known artefact
    of the old HF-based Stage 1 (when facts+considerations were both blank).
    """
    queries = []
    dropped_empty = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if not (q.get("query_text") or "").strip():
                dropped_empty += 1
                continue
            queries.append(q)
    if dropped_empty:
        log.warning("Dropped %d queries with empty query_text", dropped_empty)
    log.info("Loaded %d queries from %s", len(queries), path)
    return queries


def load_graph(path: Path):
    """Load the citation graph from a pickle file.

    The graph is a networkx DiGraph where nodes are decision_id strings and
    edges represent citations (A → B means A cites B). Node data includes
    a 'source' field ('ruling' or 'legislation').
    """
    log.info("Loading citation graph from %s ...", path)
    with open(path, "rb") as f:
        G = pickle.load(f)
    log.info(
        "Graph loaded: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )
    # Sanity check: sample nodes must look like UUIDs (36 chars with hyphens).
    # A corrupted graph built by iterating a string produces single-char nodes.
    sample = list(G.nodes())[:20]
    bad = [n for n in sample if not (isinstance(n, str) and len(n) == 36 and n.count("-") == 4)]
    if bad:
        raise RuntimeError(
            f"Graph at {path} appears corrupted — nodes look like chars, not UUIDs: {bad[:5]}\n"
            "Delete the file and rebuild with scripts/eval/build_citation_graph.py"
        )
    return G


def load_cross_encoder(model_name: str = CROSS_ENCODER_MODEL):
    """Return a list of healthy TEI /rerank URLs, or None if all unreachable.

    Each URL listed in TEI_RERANK_URLS is probed via its /health endpoint.
    Only reachable endpoints make it into the returned list — the rerank
    function will round-robin across them in parallel.
    """
    healthy = []
    for url in TEI_RERANK_URLS:
        health = url.rsplit("/", 1)[0] + "/health"
        try:
            r = requests.get(health, timeout=5)
            if r.status_code == 200:
                healthy.append(url)
                continue
            log.warning("TEI rerank %s health returned %d", url, r.status_code)
        except Exception as e:
            log.warning("TEI rerank %s unreachable: %s", url, e)
    if not healthy:
        log.warning("No TEI rerank endpoint healthy — cross_encoder will fall back to cosine")
        return None
    log.info("TEI rerank: %d endpoint(s) healthy: %s", len(healthy), healthy)
    return healthy


def _expected_output_files(
    results_dir: Path, systems: list, rankings: list, k_values: list,
) -> list:
    """Files that must contain a line for a query to count as 'complete'."""
    out = []
    for system in systems:
        for ranking in rankings:
            for k in k_values:
                out.append(results_dir / f"{system}_{ranking}_{k}.jsonl")
        out.append(results_dir / f"{system}_pool.jsonl")
        out.append(results_dir / f"{system}_layers.jsonl")
    out.append(results_dir / "cross_encoder_scores.jsonl")
    return out


def _completed_query_ids(expected_files: list) -> set:
    """Return query_ids that appear in EVERY expected output file.

    Anything else is a partial state from an interrupted run — those
    queries get re-processed and their stale lines are scrubbed by
    `_filter_outputs_to_qids` before we open the files in append mode.
    """
    if not expected_files:
        return set()
    qids_per_file = []
    for f in expected_files:
        if not f.exists():
            return set()  # Missing file → no resume coverage at all
        qids = set()
        with open(f) as fh:
            for line in fh:
                try:
                    qid = json.loads(line).get("query_id")
                except json.JSONDecodeError:
                    continue
                if qid:
                    qids.add(qid)
        qids_per_file.append(qids)
    return set.intersection(*qids_per_file)


def _filter_outputs_to_qids(expected_files: list, keep_qids: set) -> int:
    """Rewrite each output file with only lines whose query_id ∈ keep_qids.

    Returns the total number of lines dropped across all files. Called
    before opening files in append mode so a partially-written query
    doesn't end up duplicated when its retry finishes.
    """
    dropped = 0
    for f in expected_files:
        if not f.exists():
            continue
        kept = []
        with open(f) as fh:
            for line in fh:
                try:
                    qid = json.loads(line).get("query_id")
                except json.JSONDecodeError:
                    continue
                if qid in keep_qids:
                    kept.append(line if line.endswith("\n") else line + "\n")
                else:
                    dropped += 1
        with open(f, "w") as fh:
            fh.writelines(kept)
    return dropped


def run_retrieval(
    queries: list,
    client: QdrantClient,
    graph,
    tei_endpoint: str,
    cross_encoder,
    systems: list,
    k_values: list,
    results_dir: Path,
    date_index: dict = None,
    dry_run: bool = False,
    resume: bool = True,
    on_event=None,
    cancel_check=None,
) -> None:
    """Run all configured retrieval systems on all queries and write results.

    For each (system, ranking, k) triple we write a separate JSONL file.
    We embed all query texts once upfront, then apply all configurations
    to each query to avoid redundant embedding calls.

    Args:
        queries:      list of query dicts from eval_queries.jsonl
        client:       Qdrant client
        graph:        networkx DiGraph (citation graph)
        tei_endpoint: URL of live TEI embedding server
        cross_encoder: loaded CrossEncoder or None
        systems:      list of system names to run (subset of all systems)
        k_values:     list of k cutoffs
        results_dir:  directory to write output JSONL files
        dry_run:      if True, only process 2 queries
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        queries = queries[:2]
        log.info("DRY RUN: limiting to 2 queries")
        # Dry-run is by definition fresh — never inherit partial state.
        resume = False

    # Ranking strategies to run (skip cross_encoder if model not loaded)
    rankings = ["cosine", "indegree"]
    if cross_encoder is not None:
        rankings.append("cross_encoder")
    else:
        log.warning("Skipping cross_encoder ranking (model not loaded)")

    # Resume: if the on-disk results already have COMPLETE coverage for a
    # subset of `queries` under the current (systems × rankings × k_values)
    # config, skip those queries and append to the existing files. Files
    # are first scrubbed of any partial-state lines (queries that appear
    # in some output files but not all — interrupted mid-write) so the
    # post-resume run can't produce duplicates.
    expected_files = _expected_output_files(results_dir, systems, rankings, k_values)
    completed_qids: set = set()
    if resume:
        completed_qids = _completed_query_ids(expected_files)
        if completed_qids:
            requested_qids = {q["query_id"] for q in queries}
            relevant = completed_qids & requested_qids
            dropped = _filter_outputs_to_qids(expected_files, completed_qids)
            log.info(
                "RESUME: %d queries already complete (%d relevant to this run); "
                "scrubbed %d partial-state lines",
                len(completed_qids), len(relevant), dropped,
            )
            queries = [q for q in queries if q["query_id"] not in completed_qids]
            log.info("RESUME: %d queries remaining", len(queries))
            if on_event:
                on_event({
                    "type": "resume_summary",
                    "completed": len(completed_qids),
                    "relevant": len(relevant),
                    "remaining": len(queries),
                })
        else:
            log.info("RESUME: no complete output coverage found — starting fresh")

    file_mode = "a" if completed_qids else "w"

    # Build output file handles: {(system, ranking, k): open file}
    file_handles: dict = {}
    for system in systems:
        for ranking in rankings:
            for k in k_values:
                key = (system, ranking, k)
                fname = results_dir / f"{system}_{ranking}_{k}.jsonl"
                file_handles[key] = open(fname, file_mode, encoding="utf-8")
    log.info(
        "Opened %d output files in %r mode (%d systems × %d rankings × %d k-values)",
        len(file_handles), file_mode, len(systems), len(rankings), len(k_values),
    )

    # Pool files: one per system, storing the full candidate pool per query.
    # Used downstream to compute Recall-Ceiling (pool quality independent of ranking).
    pool_handles: dict = {}
    for system in systems:
        fname = results_dir / f"{system}_pool.jsonl"
        pool_handles[system] = open(fname, file_mode, encoding="utf-8")

    # Per-layer files: one per system, storing the cumulative pool size and
    # ground-truth hits at each pipeline stage (seeds → raw expansion →
    # post-temporal → post-cap). Powers the per-layer Recall-Ceiling waterfall.
    layer_handles: dict = {}
    for system in systems:
        fname = results_dir / f"{system}_layers.jsonl"
        layer_handles[system] = open(fname, file_mode, encoding="utf-8")

    # Cross-encoder score persistence: one line per query with the union
    # score map (decision_id → CE score). Powers the inspector tooltip
    # without re-querying TEI at view time.
    ce_scores_handle = open(
        results_dir / "cross_encoder_scores.jsonl", file_mode, encoding="utf-8"
    )

    # Ruling sources to search (leading decisions are also rulings)
    ruling_sources = [RULINGS_SOURCE, LEADING_SOURCE]

    # ── Embed all queries upfront ─────────────────────────────────────────
    # Append-only cache `query_embeddings.jsonl` keyed by query_id, so that
    # a resumed run skips the ~2 min TEI re-embed of queries the previous
    # attempt already vectorised. The cache is robust across re-samples
    # because (a) sampling is deterministic-superset (same query_id ⇒ same
    # query_text), and (b) we look up by query_id, never by file index.
    embed_cache_path = results_dir / "query_embeddings.jsonl"
    embed_cache: dict = {}
    if embed_cache_path.exists():
        with open(embed_cache_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = rec.get("query_id")
                vec = rec.get("embedding")
                if qid and vec:
                    embed_cache[qid] = vec
        log.info("Loaded %d cached query embeddings from %s",
                 len(embed_cache), embed_cache_path.name)

    to_embed_idx = [i for i, q in enumerate(queries)
                    if q["query_id"] not in embed_cache]
    log.info(
        "Embedding %d / %d query texts via TEI (%d already cached)",
        len(to_embed_idx), len(queries), len(queries) - len(to_embed_idx),
    )

    # TEI-Embed enforces max_batch_tokens=32768. Worst-case BPE/ws ratio for
    # German legal text is ~1.5x, so 4 queries at the 4096 ws-token cap give
    # 4 * 4096 * 1.5 = ~24'576 BPE per batch, safely under the server limit.
    # Higher batch sizes caused 413 Payload Too Large after the cap was
    # raised from 512 to 4096 ws-tokens.
    EMBED_BATCH = 4
    if to_embed_idx:
        embed_cache_handle = open(embed_cache_path, "a", encoding="utf-8")
        try:
            for i in tqdm(range(0, len(to_embed_idx), EMBED_BATCH),
                          desc="Embedding queries"):
                batch_idx = to_embed_idx[i : i + EMBED_BATCH]
                batch_texts = [queries[j]["query_text"] for j in batch_idx]
                vecs = embed_texts(batch_texts, tei_endpoint)
                for j, vec in zip(batch_idx, vecs):
                    qid = queries[j]["query_id"]
                    embed_cache[qid] = vec
                    embed_cache_handle.write(
                        json.dumps({"query_id": qid, "embedding": vec}) + "\n"
                    )
                embed_cache_handle.flush()
        finally:
            embed_cache_handle.close()

    all_embeddings = [embed_cache[q["query_id"]] for q in queries]
    log.info("Embedding done: %d vectors (cache size %d)",
             len(all_embeddings), len(embed_cache))

    # ── Main retrieval loop ───────────────────────────────────────────────
    # Run-wide text cache for cross-encoder reranking. Without this every
    # query re-fetches chunk texts for its candidate set, which dominates
    # runtime — a single decision touched by hundreds of queries would be
    # re-fetched hundreds of times. With the cache the first encounter pays
    # the Qdrant scroll, subsequent queries hit memory.
    text_cache: dict = {}

    # Per-stage timing accumulators. Used to print a breakdown every N
    # queries so we can see where the wall-clock actually goes.
    timings: dict[str, float] = {
        "qdrant_seed":      0.0,
        "graph_1hop":       0.0,
        "graph_2hop":       0.0,
        "emb_1hop":         0.0,
        "emb_2hop":         0.0,
        "fetch_text":       0.0,
        "rerank":           0.0,
        "rank_other":       0.0,
        "io_write":         0.0,
    }
    union_sizes: list[int] = []
    text_fetched_total = 0
    TIMING_REPORT_EVERY = 5

    def _now() -> float:
        return time.monotonic()

    # ── A4: Pre-warm `text_cache` from valid_ids ──────────────────────────
    # Without prewarm, the first few hundred queries each pay a Qdrant scroll
    # for their candidate texts; with prewarm the entire candidate-text
    # surface is in memory before the main loop, and every CE call hits
    # cache. Cost: one bulk fetch of ~131k chunk-0 texts (~4-7 min, ~500 MB).
    # A4 prewarm holds ~131k chunk-0 texts in RAM (~600-900 MB on a 4 GB pod),
    # which OOM-killed the kg-rag-control pod on the 4 GB Hetzner VM. The
    # prewarm is now opt-in via STAGE2_TEXT_PREWARM=1. When disabled, the
    # text_cache fills organically during Stage 2 (each unique decision_id
    # pays one Qdrant scroll the first time it appears in a CE union, every
    # subsequent CE call hits the in-memory cache).
    if os.environ.get("STAGE2_TEXT_PREWARM", "0") == "1":
        valid_ids_path_env = os.environ.get(
            "VALID_IDS_PATH", str(EVAL_DIR / "valid_ids.json")
        )
        if os.path.exists(valid_ids_path_env):
            log.info("A4: pre-warming text_cache from %s ...", valid_ids_path_env)
            with open(valid_ids_path_env) as f:
                _all_valid_ids = json.load(f)
            _t0 = time.monotonic()
            fetch_chunk_texts(_all_valid_ids, client, cache=text_cache)
            log.info("A4: text_cache pre-warmed with %d entries in %.1fs",
                     len(text_cache), time.monotonic() - _t0)
            text_fetched_total += len(text_cache)
        else:
            log.warning("A4: valid_ids.json not found at %s, skipping prewarm",
                        valid_ids_path_env)
    else:
        log.info("A4: text_cache prewarm disabled (STAGE2_TEXT_PREWARM!=1); "
                 "cache fills organically during Stage 2")

    # ── A2: Query-level ThreadPool ────────────────────────────────────────
    # Process multiple queries concurrently. Each worker computes the full
    # per-query pipeline (seed search, expansions, CE union) and returns a
    # writes-list. A single writer thread serialises file writes under a
    # lock so the per-query flush contract holds atomically.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    N_PARALLEL = int(os.environ.get("STAGE2_PARALLEL", "4"))
    log.info("A2: query-level concurrency with %d workers", N_PARALLEL)
    write_lock = Lock()
    cancelled = {"flag": False}

    def _process_query(query, qvec, q_idx):
        """Worker: full per-query pipeline. Returns dict with writes + stats.

        Side-effects allowed: text_cache (thread-safe via GIL for dict ops).
        Forbidden: file_handles, timings, union_sizes (returned as deltas).
        """
        if cancelled["flag"]:
            return None
        qid = query["query_id"]
        qtext = query["query_text"]
        query_date_ms = int(query.get("date_ms", 0))
        local_timings: dict[str, float] = {k: 0.0 for k in timings}
        local_text_fetched = 0
        local_union_size = None
        writes: list = []  # (kind, key_or_None, line_without_newline)

        def _ln(kind, key, obj):
            writes.append((kind, key, json.dumps(obj)))

        # Step 1: Get top-SEED_K Qdrant seeds (deduplicated by decision_id).
        # `exclude_decision_id=qid` removes the query's own document from
        # the seed pool — without it cosine self-match puts the query at
        # rank 1 and `graph.successors(query) == GT` is tautological.
        # `query_date_ms` enforces the closed-world temporal assumption at
        # the seed step, preventing future-dated cases from routing graph-
        # or embedding-expansion to pre-query candidates that would only be
        # reachable through citation edges published after the query date.
        _t = _now()
        seed_docs = qdrant_search(
            client, qvec, ruling_sources, limit=SEED_K,
            exclude_decision_id=qid,
            query_date_ms=query_date_ms,
        )
        local_timings["qdrant_seed"] += _now() - _t
        seed_ids = [d["decision_id"] for d in seed_docs]
        seed_score_map = {d["decision_id"]: d["score"] for d in seed_docs}

        # Optional smart-seed selection for the rag_smart baseline.
        # Cosine over-fetches a wider raw pool, indegree truncates to SEED_K.
        # Picks landmark-precedents among the semantically relevant set —
        # tests whether smart pre-filtering alone closes the GraphRAG gap.
        smart_seed_docs: list = []
        if "rag_smart" in systems:
            _t = _now()
            smart_seed_docs = qdrant_search(
                client, qvec, ruling_sources, limit=SEED_K,
                exclude_decision_id=qid,
                query_date_ms=query_date_ms,
                presort="indegree",
                graph=graph,
            )
            local_timings["qdrant_seed"] += _now() - _t

        # Per-system trace sinks for the layer waterfall analysis.
        # Each expand_* call appends (stage_name, frozenset) tuples at raw /
        # post_temporal / post_cap. Only populated if the system is active.
        trace_graph_1hop: list = [] if "graph_1hop" in systems else None
        trace_graph_2hop: list = [] if "graph_2hop" in systems else None
        trace_emb_1hop:   list = [] if "emb_1hop"   in systems else None
        trace_emb_2hop:   list = [] if "emb_2hop"   in systems else None

        # Step 2: Candidate expansion (graph + embedding space)
        # Graph 1-hop: direct citation neighbours of the seeds
        _t = _now()
        one_hop_ids = expand_graph_1hop(
            seed_ids, graph, MAX_1HOP_CANDIDATES,
            query_date_ms=query_date_ms, date_index=date_index or {},
            trace_sink=trace_graph_1hop,
        )
        local_timings["graph_1hop"] += _now() - _t
        # Graph 2-hop: 1-hop + their own citation neighbours
        _t = _now()
        two_hop_ids = expand_graph_2hop(
            seed_ids, graph, MAX_2HOP_CANDIDATES,
            query_date_ms=query_date_ms, date_index=date_index or {},
            trace_sink=trace_graph_2hop,
        )
        local_timings["graph_2hop"] += _now() - _t
        # Embedding 1-hop: kNN neighbours of seeds in embedding space
        _t = _now()
        emb_hop_ids = expand_emb_1hop(
            seed_docs, client, MAX_1HOP_CANDIDATES,
            query_date_ms=query_date_ms, date_index=date_index or {},
            trace_sink=trace_emb_1hop,
            query_id=qid,
        )
        local_timings["emb_1hop"] += _now() - _t
        # Embedding 2-hop: reuse 1-hop, expand once more
        _t = _now()
        emb_2hop_ids = expand_emb_2hop(
            seed_docs, emb_hop_ids, client, MAX_2HOP_CANDIDATES,
            query_date_ms=query_date_ms, date_index=date_index or {},
            trace_sink=trace_emb_2hop,
            query_id=qid,
        ) if "emb_2hop" in systems else set()
        local_timings["emb_2hop"] += _now() - _t

        # Merge seeds + expanded candidates into candidate sets per system
        # Assign score=0.0 to expansion-only candidates (no direct cosine score)
        def make_candidates(ids_to_add: set) -> list:
            """Merge seed_docs with new expanded IDs.

            Iterates `ids_to_add` in sorted order so dict insertion order is
            deterministic. Otherwise set-iteration over UUID strings is
            randomised per-process via PYTHONHASHSEED, which propagates into
            cosine/indegree tie-break order and CE batch composition.
            """
            merged: dict = dict(seed_score_map)  # start from seeds
            for did in sorted(ids_to_add):
                if did not in merged:
                    merged[did] = 0.0  # expansion-only: no direct cosine score
            return [{"decision_id": did, "score": sc} for did, sc in merged.items()]

        candidates_by_system = {
            "rag": [{"decision_id": d["decision_id"], "score": d["score"]}
                    for d in seed_docs],
            "emb_1hop": make_candidates(emb_hop_ids),
            "emb_2hop": make_candidates(emb_2hop_ids),
            "graph_1hop": make_candidates(one_hop_ids),
            "graph_2hop": make_candidates(two_hop_ids),
        }
        if "rag_smart" in systems:
            candidates_by_system["rag_smart"] = [
                {"decision_id": d["decision_id"], "score": d["score"]}
                for d in smart_seed_docs
            ]

        # Map: system → trace list (or None for systems with no expansion)
        trace_by_system = {
            "rag":        None,
            "rag_smart":  None,
            "emb_1hop":   trace_emb_1hop,
            "emb_2hop":   trace_emb_2hop,
            "graph_1hop": trace_graph_1hop,
            "graph_2hop": trace_graph_2hop,
        }

        # Ground truth for this query — used to compute per-layer hits inline.
        # Stored in query dict by 01_sample_queries.py as list[str].
        gt_cases = set(query.get("ground_truth_cases", []) or [])
        seed_id_set = set(seed_ids)
        # Per-system seed pool: rag_smart starts from a different seed set,
        # so its "seeds" stage in the layer waterfall reflects its own pool.
        seed_pool_by_system = {sys_name: seed_id_set for sys_name in systems}
        if "rag_smart" in systems:
            seed_pool_by_system["rag_smart"] = {
                d["decision_id"] for d in smart_seed_docs
            }

        # ── Cross-encoder rerank: precomputed ONCE over the union ─────────
        # The (query, doc) score from the reranker is independent of which
        # system surfaced the doc. We therefore rerank the union of all
        # candidates once and reuse the score map across all five systems.
        # Old code reran each system's candidate set separately, which did
        # the same work up to 5× when systems share documents (which they
        # always do — they all start from the same 60 ANN seeds).
        ce_score_map: dict | None = None
        ce_failed = False
        if cross_encoder is not None and "cross_encoder" in rankings:
            union_score: dict = {}
            for cands in candidates_by_system.values():
                for c in cands:
                    did = c["decision_id"]
                    if did not in union_score:
                        union_score[did] = c["score"]
            union_ids = list(union_score.keys())
            local_union_size = len(union_ids)
            _t = _now()
            before = len(text_cache)
            fetch_chunk_texts(union_ids, client, cache=text_cache)
            local_text_fetched += len(text_cache) - before
            local_timings["fetch_text"] += _now() - _t
            # Filter chunkless rulings (in graph but not in Qdrant) — they
            # would otherwise be sent to the cross-encoder with empty text
            # and receive model-noise scores that can steal top-k slots.
            # Also sort by decision_id so the CE batch composition is
            # deterministic across runs (set-iteration order would otherwise
            # vary per-process via PYTHONHASHSEED and cause float-score
            # drift from batch-padding asymmetry).
            union_with_text = sorted(
                (
                    {"decision_id": did, "score": s,
                     "text": text_cache.get(did, "")}
                    for did, s in union_score.items()
                    if text_cache.get(did, "").strip()
                ),
                key=lambda r: r["decision_id"],
            )
            _t = _now()
            try:
                union_ranked = rank_by_cross_encoder(
                    qtext, union_with_text, cross_encoder
                )
                ce_score_map = {r["decision_id"]: r["score"] for r in union_ranked}
                _ln("ce_scores", None,
                    {"query_id": qid, "scores": ce_score_map})
            except CrossEncoderBatchFailure as e:
                ce_failed = True
                log.warning(
                    "CE failed for query %s: %s — skipping CE outputs, "
                    "resume will retry on next run", qid, e,
                )
            local_timings["rerank"] += _now() - _t

        for system in systems:
            candidates = candidates_by_system[system]

            # Write full candidate pool for Recall-Ceiling analysis
            pool_ids = [c["decision_id"] for c in candidates]
            _ln("pool", system, {"query_id": qid, "pool": pool_ids})

            # Write per-layer waterfall: cumulative pool (seeds ∪ expansion)
            # at each pipeline stage. For rag there is no expansion — the
            # single "seeds" stage equals the final pool.
            stages = [{
                "name": "seeds",
                "pool_size": len(seed_id_set),
                "hits": len(seed_id_set & gt_cases),
            }]
            sys_trace = trace_by_system[system]
            if sys_trace:
                for stage_name, ids in sys_trace:
                    cum = seed_id_set | set(ids)
                    stages.append({
                        "name": stage_name,
                        "pool_size": len(cum),
                        "hits": len(cum & gt_cases),
                    })
            _ln("layer", system,
                {"query_id": qid, "gt_size": len(gt_cases), "stages": stages})

            for ranking in rankings:
                if ranking == "cosine":
                    ranked = rank_by_cosine(candidates)

                elif ranking == "indegree":
                    ranked = rank_by_indegree(candidates, graph)

                elif ranking == "cross_encoder":
                    # Use the precomputed union score map. For each system
                    # we just project + sort — no extra HTTP round-trip.
                    if ce_failed:
                        # CE batch failure for this query: skip the CE-ranked
                        # writes so the row stays missing from
                        # *_cross_encoder_*.jsonl. The resume mechanism then
                        # re-processes this query on the next run instead of
                        # baking a corrupted ranking into the output files.
                        continue
                    if ce_score_map is None:
                        # Cross-encoder unavailable: fall back to cosine so
                        # cross_encoder columns still get filled.
                        ranked = rank_by_cosine(candidates)
                    else:
                        # Candidates without a CE score (chunkless rulings
                        # filtered out of the union above) get a sentinel
                        # that sorts them to the tail rather than into the
                        # middle of the ranking next to legitimate 0.0
                        # scores.
                        ranked = sorted(
                            [{"decision_id": c["decision_id"],
                              "score": ce_score_map.get(
                                  c["decision_id"], -1e9)}
                             for c in candidates],
                            key=lambda x: x["score"], reverse=True,
                        )

                # Write top-k for each k value
                for k in k_values:
                    top_k_ids = [r["decision_id"] for r in ranked[:k]]
                    _ln("topk", (system, ranking, k),
                        {"query_id": qid, "retrieved": top_k_ids})

        return {
            "qid": qid,
            "writes": writes,
            "local_timings": local_timings,
            "local_union_size": local_union_size,
            "local_text_fetched": local_text_fetched,
        }
    # ── end _process_query ────────────────────────────────────────────────

    progress_lock = Lock()
    n_done = [0]

    def _flush_result(result):
        if result is None:
            return
        with write_lock:
            for kind, key, line in result["writes"]:
                if kind == "ce_scores":
                    ce_scores_handle.write(line + "\n")
                elif kind == "pool":
                    pool_handles[key].write(line + "\n")
                elif kind == "layer":
                    layer_handles[key].write(line + "\n")
                elif kind == "topk":
                    file_handles[key].write(line + "\n")
            # Per-query flush: same atomicity contract as the old sequential
            # loop. SIGKILL between two flushes loses at most one query.
            for fh in file_handles.values():
                fh.flush()
            for fh in pool_handles.values():
                fh.flush()
            for fh in layer_handles.values():
                fh.flush()
            ce_scores_handle.flush()
        with progress_lock:
            for k, v in result["local_timings"].items():
                timings[k] = timings.get(k, 0.0) + v
            if result["local_union_size"] is not None:
                union_sizes.append(result["local_union_size"])
            nonlocal_text = result["local_text_fetched"]
        # update text_fetched_total via the closure outside the lock
        _accumulate_text_fetched(nonlocal_text)
        n_done[0] += 1
        if on_event and (n_done[0] % 10 == 0 or n_done[0] == len(queries)):
            on_event({"type": "progress", "stage": "retrieval",
                      "current": n_done[0], "total": len(queries)})
        if n_done[0] % TIMING_REPORT_EVERY == 0 or n_done[0] == len(queries):
            total = sum(timings.values()) or 1.0
            avg_union = (sum(union_sizes) / len(union_sizes)) if union_sizes else 0
            log.info(
                "TIMING after %d queries — avg %.1fs/q | per-stage avg ms: %s | union avg=%d, cache=%d, fetched=%d",
                n_done[0],
                total / max(1, n_done[0]),
                ", ".join(f"{k}={1000*v/max(1,n_done[0]):.0f}" for k, v in sorted(timings.items())),
                int(avg_union),
                len(text_cache),
                text_fetched_total,
            )

    # text_fetched_total lives in the outer scope; we accumulate via a tiny
    # helper to keep `nonlocal` declarations localised.
    def _accumulate_text_fetched(delta):
        nonlocal text_fetched_total
        text_fetched_total += delta

    with ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        futures = []
        for q_idx, query in enumerate(queries):
            if cancel_check is not None and cancel_check():
                cancelled["flag"] = True
                break
            qvec = all_embeddings[q_idx]
            futures.append(ex.submit(_process_query, query, qvec, q_idx))
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Queries", unit="q"):
            if cancel_check is not None and cancel_check():
                cancelled["flag"] = True
            try:
                result = fut.result()
            except Exception as e:
                log.error("Worker failed: %s", e, exc_info=True)
                continue
            _flush_result(result)

    # Close all output files
    for fh in file_handles.values():
        fh.close()
    for fh in pool_handles.values():
        fh.close()
    for fh in layer_handles.values():
        fh.close()
    ce_scores_handle.close()

    log.info("Retrieval complete. Results written to %s", results_dir)


def run(
    systems: list = None,
    k_values: list = None,
    dry_run: bool = False,
    query_limit: int = 0,
    resume: bool = True,
    on_event=None,
    cancel_check=None,
) -> None:
    """Importable entry point. Loads resources and runs retrieval.

    Args:
        systems:      list of system names to run (default: all five)
        k_values:     list of k cutoffs (default: K_VALUES from module)
        dry_run:      process only 2 queries if True
        query_limit:  if > 0, only process the first N queries from
                      eval_queries.jsonl (the file is already shuffled with a
                      fixed seed during sampling, so a slice is roughly
                      stratified by language). Useful for fast iteration.
        on_event:     optional callback dict→None, receives progress/stage events
        cancel_check: optional callable → bool; retrieval loop checks between
                      queries and aborts if True is returned
    """
    if systems is None:
        systems = ["rag", "emb_1hop", "emb_2hop", "graph_1hop", "graph_2hop"]
    # rag_smart is opt-in (not in default list) — it's a methodological
    # follow-up baseline that selects 60 seeds by indegree-among-cosine-top-N
    # rather than pure cosine top-60. Only computed if explicitly requested.
    if k_values is None:
        k_values = K_VALUES

    def emit(event):
        if on_event:
            on_event(event)

    emit({"type": "stage_started", "stage": "retrieval"})

    # Load evaluation queries
    if not QUERIES_FILE.exists():
        raise FileNotFoundError(
            f"Queries file not found: {QUERIES_FILE} — run 01_sample_queries first"
        )
    queries = load_queries(QUERIES_FILE)
    if query_limit and query_limit > 0:
        # Stratified slice: split the file by language and take per_lang from
        # each so de/fr/it are exactly equal. Defends against the naive
        # `queries[:N]` slice that yields uneven splits — e.g. queries[:100]
        # historically gave 38/26/36 because the fixed-seed shuffle is mixed,
        # not stratified.
        per_lang = query_limit // 3
        by_lang: dict = {}
        for q in queries:
            by_lang.setdefault(q.get("language"), []).append(q)
        sliced: list = []
        for lang in ("de", "fr", "it"):
            sliced.extend(by_lang.get(lang, [])[:per_lang])
        log.info(
            "Stratified slice: %d/lang × 3 = %d queries (full set has %d)",
            per_lang, len(sliced), len(queries),
        )
        queries = sliced

    # Connect to Qdrant
    log.info("Connecting to Qdrant ...")
    client = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False, timeout=120
    )

    # Load citation graph
    graph = load_graph(GRAPH_PATH)

    # Build date index for temporal filtering of graph candidates.
    # Filters to chunk_index=0 (one representative chunk per decision) to
    # minimise scroll volume. Cached locally so remote runs only pay once.
    date_index_cache = EVAL_DIR / "date_index.json"
    if date_index_cache.exists():
        log.info("Loading date index from cache %s ...", date_index_cache)
        with open(date_index_cache) as f:
            date_index: dict = json.load(f)
        log.info("Date index loaded: %d entries", len(date_index))
    else:
        log.info("Building date index from Qdrant (first run — will be cached) ...")
        date_index = {}
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=["decision_id", "date"],
                with_vectors=False,
                scroll_filter=Filter(
                    must=[FieldCondition(key="chunk_index", match=MatchValue(value=0))]
                ),
            )
            for pt in batch:
                did = pt.payload.get("decision_id")
                raw_date = pt.payload.get("date")
                if did and raw_date and did not in date_index:
                    try:
                        date_index[did] = int(raw_date)
                    except (ValueError, TypeError):
                        pass
            if offset is None:
                break
        with open(date_index_cache, "w") as f:
            json.dump(date_index, f)
        log.info("Date index built and cached: %d entries", len(date_index))

    # Find TEI embedding endpoint
    log.info("Probing TEI embedding endpoints ...")
    tei_endpoint = find_live_tei_endpoint()
    if tei_endpoint is None:
        log.warning(
            "No live TEI endpoint found on ports %s. "
            "Falling back to direct SentenceTransformer loading (slower first query). "
            "To use TEI: docker run -d -p 8010:80 "
            "-v /data/thesis/hf-cache:/data "
            "ghcr.io/huggingface/text-embeddings-inference:1.7 "
            "--model-id BAAI/bge-m3",
            TEI_PORTS,
        )
        # tei_endpoint remains None — embed_texts() will use SentenceTransformer

    # Load cross-encoder (optional)
    cross_encoder = load_cross_encoder(CROSS_ENCODER_MODEL)

    # Run retrieval
    run_retrieval(
        queries=queries,
        client=client,
        graph=graph,
        tei_endpoint=tei_endpoint,
        cross_encoder=cross_encoder,
        systems=systems,
        k_values=k_values,
        results_dir=RESULTS_DIR,
        date_index=date_index,
        dry_run=dry_run,
        resume=resume,
        on_event=on_event,
        cancel_check=cancel_check,
    )

    emit({"type": "stage_done", "stage": "retrieval",
          "results_dir": str(RESULTS_DIR),
          "output_files": [str(p) for p in RESULTS_DIR.glob("*.jsonl")]})

    print(f"\nResults written to: {RESULTS_DIR}")


def main() -> None:
    """CLI entry point — parses args and delegates to run()."""
    parser = argparse.ArgumentParser(description="Run retrieval configurations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 2 queries (smoke test)")
    parser.add_argument("--systems", nargs="+",
                        choices=["rag", "rag_smart", "emb_1hop", "emb_2hop",
                                 "graph_1hop", "graph_2hop"],
                        default=["rag", "emb_1hop", "emb_2hop", "graph_1hop", "graph_2hop"],
                        help="Which systems to run (default: all)")
    args = parser.parse_args()
    run(systems=args.systems, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
