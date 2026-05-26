#!/usr/bin/env python3
"""
03_compute_metrics.py — Compute all evaluation metrics from retrieval results.

PURPOSE:
    Read the ranked result files produced by 02_run_retrieval.py and compute
    IR metrics for each (system, ranking, k) configuration. Also compute the
    novel Graph-Nearness metric that measures topological proximity of retrieved
    documents to the ground truth in the citation graph.

METRICS (per query, then averaged):
    Precision@k  — |retrieved ∩ ground_truth| / k
    Recall@k     — |retrieved ∩ ground_truth| / |ground_truth|
    MRR          — Mean Reciprocal Rank (1/rank of first relevant result)
    NDCG@k       — Normalised Discounted Cumulative Gain (binary relevance)
    Graph-Nearness — Mean 1/(1+distance) over retrieved docs, where distance
                     is shortest path to nearest ground-truth node in the
                     citation graph

OUTPUT:
    /data/thesis/eval/metrics/per_query_{system}_{ranking}_{k}.jsonl
    /data/thesis/eval/metrics/summary.csv

USAGE:
    python3 03_compute_metrics.py
    python3 03_compute_metrics.py --dry-run   # run on first result file only
"""

import os
import sys
import json
import math
import csv
import logging
import argparse
import pickle
from pathlib import Path
import networkx as nx
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

EVAL_DIR = Path(os.environ.get("EVAL_DIR", "data/eval"))
QUERIES_FILE = EVAL_DIR / "eval_queries.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
METRICS_DIR = EVAL_DIR / "metrics"
GRAPH_PATH = Path(os.environ.get("GRAPH_PATH", str(EVAL_DIR / "citation_graph.pkl")))

# K values must match what was used in 02_run_retrieval.py
K_VALUES = [5, 10, 15, 20]

# Graph-Nearness: maximum hop distance we bother computing.
# Paths longer than this are treated as "disconnected" (distance = infinity).
# BFS to depth 2 covers 1-hop and 2-hop neighbours, which is what the
# graph_2hop system can retrieve — so this is the natural cutoff.
MAX_NEARNESS_DEPTH = 2

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ── Derive missing k-result files (k=15 from k=20 by truncation) ────────────
#
# When a new k value is added to the evaluation (e.g. k=15), Stage 2's
# top-k result files don't carry that value yet. Since each ranking is
# emitted as a sorted top-20 list (descending by score), top-k = top-20[:k]
# for any k ≤ 20 — mathematically identical to a fresh rank-then-truncate,
# no Qdrant/TEI/graph needed. This avoids re-running the expensive Stage 2.

def derive_missing_k_result_files(
    results_dir: Path, k_values: list[int],
) -> int:
    """For any (system, ranking, k) result file missing in `results_dir`,
    derive it from the largest existing k file (by truncating each
    `retrieved` list to the smaller k). Returns the number of files
    derived. No-op if nothing's missing.

    Only handles `{system}_{ranking}_{k}.jsonl` — not pool/layers/ce_scores.
    """
    derived = 0
    # Group files by (system, ranking) so we can find the largest existing k.
    by_sysrank: dict[tuple[str, str], dict[int, Path]] = {}
    known_rankings = {"cosine", "indegree", "cross_encoder"}
    for p in results_dir.glob("*.jsonl"):
        stem = p.stem
        if stem.endswith("_pool") or stem.endswith("_layers"):
            continue
        if stem == "cross_encoder_scores" or stem == "query_embeddings":
            continue
        # Parse `{system}_{ranking}_{k}` — split off k, then ranking suffix.
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        sysrank, k_str = parts
        try:
            k = int(k_str)
        except ValueError:
            continue
        ranking = None
        system = None
        for rk in known_rankings:
            if sysrank.endswith("_" + rk):
                ranking = rk
                system = sysrank[: -(len(rk) + 1)]
                break
        if not ranking:
            continue
        by_sysrank.setdefault((system, ranking), {})[k] = p

    for (system, ranking), k_files in by_sysrank.items():
        existing_ks = sorted(k_files.keys(), reverse=True)
        if not existing_ks:
            continue
        # Take the largest existing k as the source; it has the longest
        # (sorted) `retrieved` list to truncate from.
        source_k = existing_ks[0]
        source_path = k_files[source_k]
        for k in k_values:
            if k in k_files:
                continue
            if k > source_k:
                log.warning(
                    "%s_%s_%d: cannot derive from k=%d source (would need to extend)",
                    system, ranking, k, source_k,
                )
                continue
            out_path = results_dir / f"{system}_{ranking}_{k}.jsonl"
            log.info("Deriving %s from %s ...", out_path.name, source_path.name)
            with open(source_path, encoding="utf-8") as fin, \
                 open(out_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["retrieved"] = rec.get("retrieved", [])[:k]
                    fout.write(json.dumps(rec) + "\n")
            derived += 1
    if derived:
        log.info("Derived %d new top-k result file(s) via truncation", derived)
    return derived


# ── IR Metric Functions ───────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    P@k = |{retrieved[:k]} ∩ relevant| / k

    Args:
        retrieved: ranked list of retrieved decision_ids
        relevant:  set of ground-truth decision_ids
        k:         cutoff rank

    Returns:
        float in [0, 1]
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items found in the top-k retrieved.

    R@k = |{retrieved[:k]} ∩ relevant| / |relevant|

    Returns 0.0 if the ground truth is empty (no known citations).
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list, relevant: set) -> float:
    """Reciprocal rank of the first relevant item in the ranked list.

    MRR contribution = 1/rank_of_first_hit, or 0 if no relevant item found.
    Rank is 1-indexed (first item has rank 1).
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Discounted Cumulative Gain with binary relevance.

    DCG@k = Σ_{i=1}^{k} rel_i / log2(i+1)

    where rel_i ∈ {0, 1} indicates whether the item at rank i is relevant.
    """
    dcg = 0.0
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            # log2(rank+1) is the position discount; larger rank = smaller gain
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Normalised DCG@k with binary relevance.

    NDCG@k = DCG@k / IDCG@k

    IDCG (Ideal DCG) is the DCG of a perfect ranking where all relevant items
    appear first. Returns 0.0 if there are no relevant items.
    """
    if not relevant:
        return 0.0
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    # Ideal: first min(k, |relevant|) positions are all hits
    n_ideal = min(k, len(relevant))
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


# ── Graph-Nearness Metric ─────────────────────────────────────────────────────

# Caches of BFS distance maps keyed on gt_node. Same map is reused across
# all 60+ result files (the distance from a GT node to other graph nodes
# only depends on the graph + cutoff, not on which file we're processing).
# We keep one cache per BFS view (directed citing→cited via reverse_view,
# and the legacy undirected view used for the comparison readout in the
# webui). Per pod-process the directed cache is tiny (most citers have no
# predecessors in this bipartite graph), the undirected cache reaches
# ~50 MB peak.
_BFS_DIRECTED_CACHE: dict[str, dict[str, int]] = {}
_BFS_UNDIRECTED_CACHE: dict[str, dict[str, int]] = {}

# Two views on the citation graph, both cached per id(graph) because the
# view constructor itself costs ~50 ms which would dominate a 12'678×60
# evaluation if rebuilt per call.
#
# - reverse_view (edges cited → citing) drives the DIRECTED metric: nodes
#   reachable from gt_node in the reverse view are exactly the docs Y with
#   Y → ... → gt_node in the original citing → cited direction.
# - undirected_view supplies the legacy comparison readout in the webui:
#   distance counts hops in either direction, so a retrieved precedent that
#   shares a co-citer with gt_node lands at distance 2.
_REVERSE_VIEW_CACHE: dict[int, "nx.DiGraph"] = {}
_UNDIRECTED_VIEW_CACHE: dict[int, "nx.Graph"] = {}


def _get_reverse(graph) -> "nx.DiGraph":
    key = id(graph)
    view = _REVERSE_VIEW_CACHE.get(key)
    if view is None:
        view = graph.reverse(copy=False)
        _REVERSE_VIEW_CACHE[key] = view
    return view


def _get_undirected(graph) -> "nx.Graph":
    key = id(graph)
    view = _UNDIRECTED_VIEW_CACHE.get(key)
    if view is None:
        view = graph.to_undirected(as_view=True)
        _UNDIRECTED_VIEW_CACHE[key] = view
    return view


def _min_distance_map(view, cache, gt_nodes, retrieved_set, max_depth):
    """BFS from each gt_node on `view`, return min-distance map for retrieved.

    BFS results are cached process-wide per gt_node because the same gt
    appears across many queries × all 60+ result files; the distance map
    only depends on (gt_node, view, max_depth), not on `retrieved`.
    """
    min_distances: dict = {}
    for gt_node in gt_nodes:
        bfs_result = cache.get(gt_node)
        if bfs_result is None:
            bfs_result = dict(
                nx.single_source_shortest_path_length(view, gt_node, cutoff=max_depth)
            )
            cache[gt_node] = bfs_result
        for node in retrieved_set:
            dist = bfs_result.get(node)
            if dist is not None:
                if node not in min_distances or dist < min_distances[node]:
                    min_distances[node] = dist
    return min_distances


def _score_from_min_distances(retrieved, relevant, min_distances, max_depth):
    """Bucket retrieved docs by distance and compute the mean 1/(1+d) score."""
    scores = []
    n_exact = n_near_1 = n_near_2 = n_far = 0
    for doc in retrieved:
        if doc in relevant:
            dist = 0
        elif doc in min_distances:
            dist = min_distances[doc]
        else:
            dist = max_depth + 1
        if dist == 0:
            n_exact += 1
        elif dist == 1:
            n_near_1 += 1
        elif dist == 2:
            n_near_2 += 1
        else:
            n_far += 1
            dist = float("inf")
        scores.append(0.0 if math.isinf(dist) else 1.0 / (1.0 + dist))
    nearness_score = sum(scores) / len(scores) if scores else 0.0
    return nearness_score, n_exact, n_near_1, n_near_2, n_far


def graph_nearness(
    retrieved: list,
    relevant: set,
    graph,
    max_depth: int = MAX_NEARNESS_DEPTH,
) -> dict:
    """Compute the Graph-Nearness metric for a single query.

    Graph-Nearness measures how topologically close the retrieved documents
    are to the ground truth in the citation graph, even when they are not
    exact matches. This is the novel metric of the thesis.

    The thesis-canonical variant counts the number of forward citation hops
    a retrieved doc Y needs to reach a GT node along citing → cited edges
    (a jurist follows citations from their hit down to its precedents). We
    implement it as BFS from each GT node on the reversed DiGraph, so nodes
    reachable from gt_node in the reversed view are exactly the docs Y with
    Y → ... → gt_node in the original direction.

    For comparison we also compute the legacy undirected variant (distance
    counts hops in either direction). The webui can switch between the two
    via the matching fields in the returned dict.

    Buckets per retrieved doc:
        exact  (distance=0): the retrieved doc IS a ground-truth doc
        near_1 (distance=1): 1 hop from any ground-truth doc
        near_2 (distance=2): 2 hops
        far    (>2 or disconnected): no path within max_depth hops

    Per-doc score: 1 / (1 + distance)   (exact=1.0, near_1=0.5, near_2=0.33, far=0)
    Overall: mean over all retrieved docs.

    Returns:
        nearness_score, n_exact, n_near_1, n_near_2, n_far (directed)
        nearness_score_undirected, n_near_1_undirected,
            n_near_2_undirected, n_far_undirected (legacy comparison)
        n_exact is identical in both views (depends only on `relevant`)
        and is reported once.
    """
    if not retrieved:
        return {
            "nearness_score": 0.0,
            "n_exact": 0, "n_near_1": 0, "n_near_2": 0, "n_far": 0,
            "nearness_score_undirected": 0.0,
            "n_near_1_undirected": 0,
            "n_near_2_undirected": 0,
            "n_far_undirected": 0,
        }

    reverse_view = _get_reverse(graph)
    undirected_view = _get_undirected(graph)

    gt_in_graph = [gt for gt in relevant if gt in reverse_view]
    retrieved_set = set(retrieved)

    min_dir = _min_distance_map(reverse_view, _BFS_DIRECTED_CACHE,
                                gt_in_graph, retrieved_set, max_depth)
    min_und = _min_distance_map(undirected_view, _BFS_UNDIRECTED_CACHE,
                                gt_in_graph, retrieved_set, max_depth)

    s_d, n_exact, n1_d, n2_d, far_d = _score_from_min_distances(
        retrieved, relevant, min_dir, max_depth
    )
    s_u, _, n1_u, n2_u, far_u = _score_from_min_distances(
        retrieved, relevant, min_und, max_depth
    )

    return {
        "nearness_score": s_d,
        "n_exact": n_exact,
        "n_near_1": n1_d,
        "n_near_2": n2_d,
        "n_far": far_d,
        "nearness_score_undirected": s_u,
        "n_near_1_undirected": n1_u,
        "n_near_2_undirected": n2_u,
        "n_far_undirected": far_u,
    }


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_per_query_metrics(
    queries_map: dict,
    results_file: Path,
    graph,
    k: int,
) -> list:
    """Compute all metrics for every query in a single results file.

    Args:
        queries_map:  dict[query_id -> query_dict] from eval_queries.jsonl
        results_file: path to a {system}_{ranking}_{k}.jsonl file
        graph:        networkx DiGraph
        k:            the k cutoff (must match the file's k)

    Returns:
        list of per-query metric dicts
    """
    per_query = []

    with open(results_file, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        record = json.loads(line)
        qid = record["query_id"]
        retrieved = record["retrieved"]  # already top-k ordered list

        if qid not in queries_map:
            log.warning("Query %s not found in queries_map — skipping", qid)
            continue

        query = queries_map[qid]
        relevant = set(query["ground_truth_cases"])

        # Standard IR metrics
        p_k = precision_at_k(retrieved, relevant, k)
        r_k = recall_at_k(retrieved, relevant, k)
        # F1 per query: 2·P·R / (P+R). Macro-averaged at the aggregate step
        # via aggregate_metrics; this is the right composition order (mean
        # of F1 ≠ F1 of means, Jensen's inequality).
        f1_k = (2.0 * p_k * r_k / (p_k + r_k)) if (p_k + r_k) > 0 else 0.0
        rr = reciprocal_rank(retrieved, relevant)
        nd_k = ndcg_at_k(retrieved, relevant, k)

        # Graph-Nearness (novel metric)
        nearness = graph_nearness(retrieved, relevant, graph)

        per_query.append({
            "query_id": qid,
            "language": query.get("language", "?"),
            "year": query.get("year", 0),
            "n_ground_truth": len(relevant),
            "precision": p_k,
            "recall": r_k,
            "f1": f1_k,
            "mrr": rr,
            "ndcg": nd_k,
            **nearness,  # nearness_score, n_exact, n_near_1, n_near_2, n_far
        })

    return per_query


def aggregate_metrics(per_query: list) -> dict:
    """Compute macro-averages over per-query metric dicts.

    We use macro-averaging (simple mean over queries) rather than micro-
    averaging because we want each query to have equal weight regardless
    of how many ground-truth items it has.
    """
    if not per_query:
        return {}

    keys = [
        "precision", "recall", "f1", "mrr", "ndcg",
        "nearness_score", "nearness_score_undirected",
    ]
    aggregated = {}
    for key in keys:
        vals = [r[key] for r in per_query if key in r]
        aggregated[f"mean_{key}"] = sum(vals) / len(vals) if vals else 0.0

    aggregated["n_queries"] = len(per_query)
    # Fraction of queries with at least one hit in top-k
    aggregated["hit_rate"] = sum(1 for r in per_query if r["precision"] > 0) / len(per_query)

    return aggregated


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    dry_run: bool = False,
    on_event=None,
    cancel_check=None,
) -> None:
    """Load all result files, compute metrics, write output.

    Args:
        dry_run:      process only first result file
        on_event:     optional callback dict→None, receives progress events
        cancel_check: optional callable → bool; checked between files
    """
    def emit(event):
        if on_event:
            on_event(event)

    emit({"type": "stage_started", "stage": "metrics"})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load queries ──────────────────────────────────────────────────────
    if not QUERIES_FILE.exists():
        log.error("Queries file not found: %s", QUERIES_FILE)
        sys.exit(1)
    queries_map: dict = {}
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                queries_map[q["query_id"]] = q
    log.info("Loaded %d queries", len(queries_map))

    # ── Load citation graph ───────────────────────────────────────────────
    log.info("Loading citation graph ...")
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    log.info("Graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    # ── Derive missing k-result files (e.g. k=15 from k=20) ───────────────
    # Top-k rankings are always emitted as sorted lists, so top-k = top-K[:k]
    # for any k ≤ K. Avoids re-running Stage 2 just to add a new cutoff.
    derive_missing_k_result_files(RESULTS_DIR, K_VALUES)

    # ── Enumerate result files ────────────────────────────────────────────
    result_files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not result_files:
        log.error("No result files found in %s — run 02_run_retrieval.py first", RESULTS_DIR)
        sys.exit(1)

    if dry_run:
        result_files = result_files[:1]
        log.info("DRY RUN: processing only %s", result_files[0])

    log.info("Found %d result files to evaluate", len(result_files))

    # ── Process each result file ──────────────────────────────────────────
    summary_rows = []

    if on_event:
        on_event({"type": "progress", "stage": "metrics",
                  "current": 0, "total": len(result_files)})
    for rf_idx, rf in enumerate(tqdm(result_files, desc="Result files")):
        if cancel_check is not None and cancel_check():
            log.warning("Cancel requested — stopping metrics at file %d/%d",
                        rf_idx, len(result_files))
            break
        # Parse config from filename: {system}_{ranking}_{k}.jsonl
        stem = rf.stem  # e.g. "rag_cosine_10"
        parts = stem.rsplit("_", 1)  # split off the k value
        if len(parts) != 2:
            log.warning("Unexpected filename format: %s — skipping", rf.name)
            continue
        system_ranking, k_str = parts
        try:
            k = int(k_str)
        except ValueError:
            log.warning("Could not parse k from filename: %s", rf.name)
            continue

        # Further split system_ranking: "rag_cosine", "graph_1hop_indegree", etc.
        # We need to handle underscores in system names (graph_1hop, graph_2hop)
        # Strategy: split from the right, known rankings are: cosine, indegree, cross_encoder
        known_rankings = {"cosine", "indegree", "cross_encoder"}
        ranking = None
        system = None
        for rk in known_rankings:
            if system_ranking.endswith("_" + rk):
                ranking = rk
                system = system_ranking[: -(len(rk) + 1)]
                break
        if not ranking:
            log.warning("Could not parse system/ranking from: %s — skipping", stem)
            continue

        log.info("Processing: system=%s ranking=%s k=%d", system, ranking, k)

        # Resume from any previously-written per_query file with full
        # query coverage AND matching qid set AND the current metric schema.
        # Stage 3 used to redo all 57 files on every restart — with
        # graph_nearness costing ~10 min/file, that lost hours of work each
        # time the pod was bounced.
        #
        # Schema check: presence of `nearness_score_undirected` is the
        # sentinel for the current schema (added 2026-05-21).
        #
        # Qid-set check: previously the resume accepted any file with
        # `len(existing) >= len(queries_map) - 5`. That allowed a file from
        # a different evaluation (e.g. 3000 old qids) to satisfy a new run
        # (e.g. 1500 different qids) — the stale aggregate would silently
        # land in summary.csv. We now require the qid set in the file to
        # match the current `queries_map` within a ±5 tolerance.
        per_query_file = METRICS_DIR / f"per_query_{system}_{ranking}_{k}.jsonl"
        per_query = None
        if per_query_file.exists():
            existing = []
            try:
                with open(per_query_file, encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            existing.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                existing = []
            # Schema sentinel: `nearness_score_undirected` was added
            # 2026-05-21 alongside the directed/undirected split of
            # graph_nearness. Any file with the undirected nearness field
            # has the full current schema; older files are recomputed.
            has_new_schema = bool(existing) and "nearness_score_undirected" in existing[0]
            existing_qids = {r.get("query_id") for r in existing if r.get("query_id")}
            target_qids = set(queries_map.keys())
            qid_diff = len(existing_qids ^ target_qids)
            qids_match = qid_diff <= 5
            if has_new_schema and qids_match:
                log.info(
                    "RESUME: %s already complete (%d rows, qid set matches) — skipping",
                    per_query_file.name, len(existing),
                )
                per_query = existing
            elif existing and not has_new_schema:
                log.info(
                    "RESUME: %s lacks `f1` column — recomputing",
                    per_query_file.name,
                )
            elif existing and not qids_match:
                extra = len(existing_qids - target_qids)
                missing = len(target_qids - existing_qids)
                log.info(
                    "RESUME: %s has stale qid set (%d extra, %d missing vs %d target) — recomputing",
                    per_query_file.name, extra, missing, len(target_qids),
                )

        if per_query is None:
            # Compute per-query metrics fresh
            per_query = compute_per_query_metrics(
                queries_map, rf, graph, k,
            )
            with open(per_query_file, "w", encoding="utf-8") as fout:
                for row in per_query:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Compute aggregate
        agg = aggregate_metrics(per_query)
        summary_rows.append({
            "system": system,
            "ranking": ranking,
            "k": k,
            **agg,
        })

        if on_event:
            on_event({"type": "progress", "stage": "metrics",
                      "current": rf_idx + 1, "total": len(result_files)})

    # ── Compute Recall-Ceiling per system (pool quality) ──────────────────
    # Recall-Ceiling = |pool ∩ ground_truth| / |ground_truth|
    # Upper bound on recall achievable by any ranking — independent of k.
    pool_rows = []
    pool_files = sorted(RESULTS_DIR.glob("*_pool.jsonl"))
    for pf in pool_files:
        system = pf.stem.replace("_pool", "")
        ceilings = []
        precisions = []
        f1s = []
        pool_sizes = []
        with open(pf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec["query_id"]
                pool = set(rec["pool"])
                if qid not in queries_map:
                    continue
                relevant = set(queries_map[qid]["ground_truth_cases"])
                if not relevant:
                    continue
                hits = len(pool & relevant)
                rec_v = hits / len(relevant)
                # Pool-Precision per query — query-mean weights each query
                # equally regardless of pool/GT size.
                prc_v = hits / len(pool) if pool else 0.0
                # Pool-F1 per query: same harmonic mean as ranked F1, but on
                # pool recall/precision. Useful when the pool is huge and
                # precision is near zero — F1 stays sensitive to changes in
                # the smaller of the two while showing it on the same scale.
                f1_v = (2.0 * prc_v * rec_v / (prc_v + rec_v)) if (prc_v + rec_v) > 0 else 0.0
                ceilings.append(rec_v)
                precisions.append(prc_v)
                f1s.append(f1_v)
                pool_sizes.append(len(pool))
        if ceilings:
            pool_rows.append({
                "system": system,
                "recall_ceiling": sum(ceilings) / len(ceilings),
                "precision_ceiling": sum(precisions) / len(precisions),
                "f1_ceiling": sum(f1s) / len(f1s),
                "mean_pool_size": sum(pool_sizes) / len(pool_sizes),
                "n_queries": len(ceilings),
            })
    if pool_rows:
        pool_file = METRICS_DIR / "recall_ceiling.csv"
        with open(pool_file, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(pool_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pool_rows)
        log.info("Recall-Ceiling written to %s", pool_file)
        print("\n" + "=" * 60)
        print("POOL CEILINGS (Recall + Precision)")
        print("=" * 60)
        print(f"  {'System':<15} {'Recall':>10} {'Precision':>10} {'Pool Size':>12}")
        print("  " + "-" * 50)
        for row in sorted(pool_rows, key=lambda x: -x["recall_ceiling"]):
            print(f"  {row['system']:<15} {row['recall_ceiling']:>10.4f} "
                  f"{row['precision_ceiling']:>10.4f} "
                  f"{row['mean_pool_size']:>12.1f}")

    # ── Per-layer Recall-Ceiling (pipeline waterfall) ─────────────────────
    # For each system, we emitted a stage-by-stage snapshot of the cumulative
    # candidate pool (seeds, raw expansion, post-temporal, post-cap) along
    # with the number of ground-truth hits at each stage. Aggregating across
    # all queries shows where in the pipeline recall is gained or lost.
    layer_rows = []
    for lf in sorted(RESULTS_DIR.glob("*_layers.jsonl")):
        system = lf.stem.replace("_layers", "")
        per_stage: dict = {}  # stage_name → {"order": int, "ceilings": [], "precisions": [], "f1s": [], "sizes": []}
        with open(lf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gt = rec.get("gt_size", 0)
                if not gt:
                    continue
                for i, stg in enumerate(rec["stages"]):
                    bucket = per_stage.setdefault(
                        stg["name"],
                        {"order": i, "ceilings": [], "precisions": [],
                         "f1s": [], "sizes": []},
                    )
                    rec_v = stg["hits"] / gt
                    pool_size = stg["pool_size"]
                    prc_v = (stg["hits"] / pool_size) if pool_size else 0.0
                    f1_v = (2.0 * prc_v * rec_v / (prc_v + rec_v)) if (prc_v + rec_v) > 0 else 0.0
                    bucket["ceilings"].append(rec_v)
                    bucket["precisions"].append(prc_v)
                    bucket["f1s"].append(f1_v)
                    bucket["sizes"].append(pool_size)
        for name, data in sorted(per_stage.items(), key=lambda kv: kv[1]["order"]):
            if not data["ceilings"]:
                continue
            layer_rows.append({
                "system": system,
                "stage": name,
                "stage_order": data["order"],
                "mean_recall_ceiling": sum(data["ceilings"]) / len(data["ceilings"]),
                "mean_precision_ceiling": sum(data["precisions"]) / len(data["precisions"]),
                "mean_f1_ceiling": sum(data["f1s"]) / len(data["f1s"]),
                "mean_pool_size": sum(data["sizes"]) / len(data["sizes"]),
                "n_queries": len(data["ceilings"]),
            })
    if layer_rows:
        layer_file = METRICS_DIR / "recall_ceiling_layers.csv"
        with open(layer_file, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(layer_rows[0].keys()))
            writer.writeheader()
            writer.writerows(layer_rows)
        log.info("Per-layer Pool Ceilings written to %s", layer_file)
        print("\n" + "=" * 80)
        print("POOL CEILINGS WATERFALL (per pipeline stage)")
        print("=" * 80)
        print(
            f"  {'System':<12} {'Stage':<16} "
            f"{'Recall':>9} {'Precision':>10} {'Pool Size':>10} {'ΔRec':>8}"
        )
        print("  " + "-" * 70)
        last_system = None
        prev_ceil = None
        for row in layer_rows:
            if row["system"] != last_system:
                last_system = row["system"]
                prev_ceil = None
            delta = ""
            if prev_ceil is not None:
                delta = f"{row['mean_recall_ceiling'] - prev_ceil:+.4f}"
            print(
                f"  {row['system']:<12} {row['stage']:<16} "
                f"{row['mean_recall_ceiling']:>9.4f} "
                f"{row['mean_precision_ceiling']:>10.4f} "
                f"{row['mean_pool_size']:>10.1f} {delta:>8}"
            )
            prev_ceil = row["mean_recall_ceiling"]

    # ── Write summary CSV ─────────────────────────────────────────────────
    if summary_rows:
        summary_file = METRICS_DIR / "summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(summary_file, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        log.info("Summary written to %s", summary_file)

    # ── Write per-language summary CSV ────────────────────────────────────
    # One row per (system, ranking, k, language). Aggregates each
    # per_query file's rows grouped by `language`, so the heatmap UI can
    # render a "de / fr / it / all" filter without re-reading the JSONLs.
    by_lang_rows: list[dict] = []
    for rf in result_files:
        stem = rf.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        system_ranking, k_str = parts
        try:
            k = int(k_str)
        except ValueError:
            continue
        ranking = None
        system = None
        for rk in ("cosine", "indegree", "cross_encoder"):
            if system_ranking.endswith("_" + rk):
                ranking = rk
                system = system_ranking[: -(len(rk) + 1)]
                break
        if not ranking:
            continue
        per_query_file = METRICS_DIR / f"per_query_{system}_{ranking}_{k}.jsonl"
        if not per_query_file.exists():
            continue
        by_lang: dict[str, list[dict]] = {}
        with open(per_query_file, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lang = rec.get("language", "?")
                by_lang.setdefault(lang, []).append(rec)
        for lang, recs in sorted(by_lang.items()):
            agg = aggregate_metrics(recs)
            by_lang_rows.append({
                "system": system,
                "ranking": ranking,
                "k": k,
                "language": lang,
                **agg,
            })
    if by_lang_rows:
        by_lang_file = METRICS_DIR / "summary_by_language.csv"
        # Stable column order: descriptors first, metrics second.
        descriptor_fields = ["system", "ranking", "k", "language"]
        metric_fields = [f for f in by_lang_rows[0].keys()
                         if f not in descriptor_fields]
        fieldnames = descriptor_fields + metric_fields
        with open(by_lang_file, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(by_lang_rows)
        log.info("Per-language summary written to %s (%d rows)",
                 by_lang_file, len(by_lang_rows))

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)

    # Group by k for readability
    for k_filter in sorted(set(r["k"] for r in summary_rows)):
        rows_k = [r for r in summary_rows if r["k"] == k_filter]
        rows_k.sort(key=lambda x: (x["system"], x["ranking"]))

        print(f"\n  k={k_filter}")
        header = (
            f"  {'System':<15} {'Ranking':<15} "
            f"{'P@k':>7} {'R@k':>7} {'F1@k':>7} {'MRR':>7} {'NDCG':>7} "
            f"{'Nearness':>9} {'HitRate':>8}"
        )
        print(header)
        print("  " + "-" * 96)

        for row in rows_k:
            print(
                f"  {row['system']:<15} {row['ranking']:<15} "
                f"{row.get('mean_precision', 0):>7.4f} "
                f"{row.get('mean_recall', 0):>7.4f} "
                f"{row.get('mean_f1', 0):>7.4f} "
                f"{row.get('mean_mrr', 0):>7.4f} "
                f"{row.get('mean_ndcg', 0):>7.4f} "
                f"{row.get('mean_nearness_score', 0):>9.4f} "
                f"{row.get('hit_rate', 0):>8.4f}"
            )

    print("\n" + "=" * 100)
    print(f"Summary CSV: {METRICS_DIR / 'summary.csv'}")
    print(f"Per-query files: {METRICS_DIR}/per_query_*.jsonl")
    print("=" * 100)

    emit({"type": "stage_done", "stage": "metrics",
          "metrics_dir": str(METRICS_DIR),
          "n_configs": len(summary_rows)})


# Back-compat alias
main = run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only the first result file (smoke test)"
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
