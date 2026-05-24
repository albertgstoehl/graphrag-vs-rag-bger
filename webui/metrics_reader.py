"""Reads the CSV/JSONL artifacts produced by 03_compute_metrics.py.

Kept separate from executor so the metrics pages can be rendered even
when no pipeline is running.
"""

from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _eval_dir() -> Path:
    return Path(os.environ.get("EVAL_DIR", "/app/data/eval"))


def metrics_dir() -> Path:
    return _eval_dir() / "metrics"


def results_dir() -> Path:
    return _eval_dir() / "results"


def summary() -> list[dict]:
    p = metrics_dir() / "summary.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_by_language() -> list[dict]:
    """One row per (system, ranking, k, language). Returns [] if Stage 3
    hasn't produced it yet (older runs predate the per-language split)."""
    p = metrics_dir() / "summary_by_language.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recall_ceiling() -> list[dict]:
    p = metrics_dir() / "recall_ceiling.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recall_ceiling_layers() -> list[dict]:
    p = metrics_dir() / "recall_ceiling_layers.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def per_query_config_names() -> list[str]:
    """List every per_query_{system}_{ranking}_{k}.jsonl stem."""
    return sorted([
        p.stem.replace("per_query_", "")
        for p in metrics_dir().glob("per_query_*.jsonl")
    ])


def per_query_metric_for(config: str, query_id: str) -> Optional[dict]:
    """Find the per-query metric record for one config + query_id."""
    p = metrics_dir() / f"per_query_{config}.jsonl"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        for line in f:
            if query_id in line:
                try:
                    rec = json.loads(line)
                    if rec.get("query_id") == query_id:
                        return rec
                except json.JSONDecodeError:
                    continue
    return None


def all_per_query_metrics(query_id: str) -> list[dict]:
    """Collect the per-query metric row for all configs (for drill-down)."""
    out: list[dict] = []
    for config in per_query_config_names():
        rec = per_query_metric_for(config, query_id)
        if rec:
            rec["_config"] = config
            out.append(rec)
    return out


def evaluated_query_ids() -> set:
    """Set of query_ids that the latest metrics output actually evaluated.

    Reads from `metrics/per_query_*.jsonl`. All such files share the same
    qid set (one row per evaluated query, per (system × ranking × k)
    config), so we only have to read one. Returns an empty set if no
    metrics output exists yet — callers should fall back to the live
    queries file in that case.
    """
    for f in metrics_dir().glob("per_query_*.jsonl"):
        out: set = set()
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        qid = json.loads(line).get("query_id")
                    except json.JSONDecodeError:
                        continue
                    if qid:
                        out.add(qid)
            return out
        except OSError:
            continue
    return set()


def load_queries(limit: Optional[int] = None) -> list[dict]:
    """Load eval_queries.jsonl, optionally truncated."""
    p = _eval_dir() / "eval_queries.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


# Per-query lookups used to scan a JSONL file linearly per call (~130 ms
# cold per file, × 5 functions = ~650 ms per qid). Now we read each file
# ONCE into a `{query_id: parsed_record}` index and reuse it. Pod-wide
# lifetime; invalidated by `invalidate_caches()` when a run completes
# (scrub may have rewritten lines). Memory: ~3000 qids × ~few KB record
# × ~50 files ≈ ~30 MB, fine.
@lru_cache(maxsize=128)
def _index_jsonl_by_qid(path_str: str) -> dict[str, dict]:
    p = Path(path_str)
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("query_id")
            if qid:
                out[qid] = rec
    return out


def query_by_id(query_id: str) -> Optional[dict]:
    return _index_jsonl_by_qid(str(_eval_dir() / "eval_queries.jsonl")).get(query_id)


def retrieved_for(query_id: str) -> dict[str, list[str]]:
    """For every `{system}_{ranking}_{k}.jsonl` in results, find this query's
    retrieved list. Keys: 'rag_cosine_10', values: list[decision_id]."""
    out: dict[str, list[str]] = {}
    for p in sorted(results_dir().glob("*.jsonl")):
        name = p.name
        if (name.endswith("_pool.jsonl") or name.endswith("_layers.jsonl")
                or name == "query_embeddings.jsonl"
                or name == "cross_encoder_scores.jsonl"):
            continue
        rec = _index_jsonl_by_qid(str(p)).get(query_id)
        if rec is not None:
            out[p.stem] = rec.get("retrieved", [])
    return out


def pool_for(query_id: str) -> dict[str, list[dict]]:
    """For each system's `*_pool.jsonl`, return the candidate pool of one
    query as `[{decision_id, score}, ...]`. Score is cosine for seeds,
    0.0 for expansion-only candidates."""
    out: dict[str, list[dict]] = {}
    for p in sorted(results_dir().glob("*_pool.jsonl")):
        system = p.name.replace("_pool.jsonl", "")
        rec = _index_jsonl_by_qid(str(p)).get(query_id)
        if rec is not None:
            out[system] = rec.get("candidates", rec.get("pool", []))
    return out


def seeds_for(query_id: str) -> set[str]:
    """The 60 ANN seeds for one query.

    Seeds are system-independent (every system starts from the same
    cosine top-60), and the RAG pool IS the seed set by construction
    (RAG has no expansion). Reading the seeds from `rag_pool.jsonl`
    avoids the need for a separate persistence file.
    """
    rec = _index_jsonl_by_qid(str(results_dir() / "rag_pool.jsonl")).get(query_id)
    if rec is None:
        return set()
    ids = rec.get("pool") or rec.get("candidates") or []
    return {c["decision_id"] if isinstance(c, dict) else c for c in ids}


def cross_encoder_scores_for(query_id: str) -> dict[str, float]:
    """Optional: load cross-encoder scores for one query if persisted.

    Reads `results/cross_encoder_scores.jsonl`. Each line is
    `{"query_id": "...", "scores": {"<did>": float, ...}}`. Returns the
    scores dict (decision_id → score) or {} if the file isn't present
    (older runs predate the persistence patch)."""
    rec = _index_jsonl_by_qid(
        str(results_dir() / "cross_encoder_scores.jsonl")
    ).get(query_id)
    return rec.get("scores", {}) if rec else {}


@lru_cache(maxsize=1)
def _latest_per_query_metric_index_cached() -> dict:
    out: dict[tuple[str, str, int], dict[str, dict]] = {}
    for p in metrics_dir().glob("per_query_*.jsonl"):
        stem = p.stem.replace("per_query_", "")
        # stem like "graph_1hop_cosine_10" — split last 2 components off.
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        system, ranking, k_str = parts
        try:
            k = int(k_str)
        except ValueError:
            continue
        bucket: dict[str, dict] = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    qid = rec.get("query_id")
                    if qid:
                        bucket[qid] = rec
                except json.JSONDecodeError:
                    continue
        out[(system, ranking, k)] = bucket
    return out


def latest_per_query_metric_index() -> dict[tuple[str, str, int], dict[str, dict]]:
    """Pre-build an index over all per_query_*.jsonl files for the table view.

    Returns `{(system, ranking, k): {query_id: metric_record}}`. Cached
    for the lifetime of the pod — invalidated explicitly via
    `invalidate_caches()` when the executor completes a Stage 3 run.

    An mtime-based cache key was tempting but broken: while a run is
    in progress, the result/metrics files get appended to constantly,
    so mtime-keyed caching invalidated on every request. ~850 ms cold.
    """
    return _latest_per_query_metric_index_cached()


@lru_cache(maxsize=1)
def _per_query_pool_metrics_cached() -> dict:
    """Compute per-query Pool-Recall AND Pool-Precision in a single pass
    over `*_pool.jsonl` — both share the |pool ∩ GT| intersection and
    re-doing the IO would be wasteful.

    Returns `{query_id: {system: {"recall": float, "precision": float,
                                  "pool_size": int}}}`.
    Recall    = |pool ∩ GT| / |GT|                 (1.0 when GT is empty)
    Precision = |pool ∩ GT| / |pool|               (0.0 when pool is empty)
    """
    queries = {q["query_id"]: set(q.get("ground_truth_cases", []))
               for q in load_queries()}
    out: dict[str, dict[str, dict]] = {qid: {} for qid in queries}
    for p in sorted(results_dir().glob("*_pool.jsonl")):
        system = p.name.replace("_pool.jsonl", "")
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = rec.get("query_id")
                if not qid or qid not in queries:
                    continue
                pool_ids = {c["decision_id"] if isinstance(c, dict) else c
                            for c in rec.get("candidates", rec.get("pool", []))}
                gt = queries[qid]
                hit = len(pool_ids & gt)
                pool_size = len(pool_ids)
                recall = (hit / len(gt)) if gt else 0.0
                precision = (hit / pool_size) if pool_size else 0.0
                f1 = (2.0 * precision * recall / (precision + recall)) \
                     if (precision + recall) > 0 else 0.0
                out[qid][system] = {
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "pool_size": pool_size,
                }
    return out


def per_query_pool_metrics() -> dict[str, dict[str, dict]]:
    """Per-query Pool-Recall + Pool-Precision per system (post_cap pool).
    Cached pod-lifetime; invalidated by `invalidate_caches()`. ~3.9 s cold.
    """
    return _per_query_pool_metrics_cached()


def per_query_pool_recall() -> dict[str, dict[str, float]]:
    """Back-compat shim: per-query Pool-Recall only, flat float per system.

    Returns `{query_id: {system: pool_recall_float}}`. Prefer
    `per_query_pool_metrics()` for new callers — it returns precision
    + pool size for the same IO cost.
    """
    return {qid: {s: m["recall"] for s, m in cells.items()}
            for qid, cells in _per_query_pool_metrics_cached().items()}


def invalidate_caches() -> None:
    """Drop all aggregate caches. The executor calls this when a Stage 2
    or Stage 3 run finishes — that's when the per_query_*.jsonl /
    *_pool.jsonl files get a final, complete snapshot worth re-reading.
    """
    _latest_per_query_metric_index_cached.cache_clear()
    _per_query_pool_metrics_cached.cache_clear()
    # The shared per-file qid index — any of the per-qid lookups
    # (query_by_id, retrieved_for, pool_for, seeds_for,
    # cross_encoder_scores_for) reads through this.
    _index_jsonl_by_qid.cache_clear()


def pipeline_matrix() -> dict:
    """Build the consolidated pipeline matrix data: one row per system,
    one column per pipeline stage. Pool stages come from
    `recall_ceiling_layers.csv`; ranking stages come from `summary.csv`.

    Returns:
      {
        "systems": ["rag", "emb_1hop", ...],
        "pool_stages": [{"key": "seeds", "label": "seeds", "tooltip": "..."}, ...],
        "rank_stages": [{"key": "cosine_5", "label": "cos@5", ...}, ...],
        "cells": {
          (system, stage_key): {"value": float, "pool_size": float|None,
                                "n_queries": int, "kind": "pool"|"rank"}
        },
      }
    """
    POOL_TOOLTIPS = {
        "seeds": (
            "Top-60 chunks by cosine similarity against the query vector in "
            "Qdrant, deduplicated to unique decision_ids. The query's own "
            "decision is excluded server-side and future-dated decisions are "
            "dropped via the indexed `date_ms` range filter."
        ),
        "raw": (
            "Seeds plus this system's expansion candidates, before any "
            "filtering. Graph systems take `graph.successors(seed)` over the "
            "citation graph; embedding systems take kNN neighbours of each "
            "seed in BGE-M3 vector space. RAG has no expansion."
        ),
        "post_temporal": (
            "Raw set with all decisions whose `date_ms ≥ query.date_ms` "
            "removed. Enforces the closed-world assumption: a candidate is "
            "only retained if it existed at query time."
        ),
        "post_cap": (
            "post_temporal truncated to the system's pool cap (RAG: 60, "
            "1-hop: 400, 2-hop: 800), by descending cosine score for "
            "embedding paths and descending indegree for graph paths. This "
            "is the final candidate pool fed into the ranker."
        ),
    }
    RANK_TOOLTIPS = {
        "cosine": (
            "Final pool ranked by cosine similarity to the query, top-k "
            "returned. Expansion-only candidates without a direct cosine "
            "score (graph hops) are scored 0, so this ranking is essentially "
            "blind to the citation pathway — included for completeness, not "
            "as a primary comparison strategy."
        ),
        "cross_encoder": (
            "Final pool ranked by the BAAI/bge-reranker-v2-m3 cross-encoder. "
            "Each (query, candidate-text) pair is scored via TEI's /rerank "
            "endpoint. Computed once per query over the union of all "
            "systems' candidates and reused — reranker scores are "
            "system-independent. Top-k returned."
        ),
        "indegree": (
            "Final pool ranked by `log(1 + indegree)` from the citation "
            "graph. Query-independent: favours globally-cited landmark "
            "BGEs. Top-k returned."
        ),
    }

    pool_stages = [
        {"key": s, "label": s, "tooltip": POOL_TOOLTIPS[s]}
        for s in ("seeds", "raw", "post_temporal", "post_cap")
    ]
    short_rank = {"cosine": "cos", "cross_encoder": "CE", "indegree": "ind"}
    rank_stages = []
    for ranking in ("cosine", "cross_encoder", "indegree"):
        for k in (5, 10, 15, 20):
            rank_stages.append({
                "key": f"{ranking}_{k}",
                "label": f"{short_rank[ranking]}@{k}",
                "tooltip": RANK_TOOLTIPS[ranking],
                "ranking": ranking,
                "k": k,
            })

    cells: dict = {}

    # Pool stages from recall_ceiling_layers.csv. As of 2026-05-10 the
    # CSV also carries `mean_precision_ceiling` per (system, stage), so a
    # full Pool-Precision waterfall is available for every stage. Older
    # CSVs without the column fall through to a post_cap-only fallback
    # below (computed live from `*_pool.jsonl`).
    layers_p = metrics_dir() / "recall_ceiling_layers.csv"
    csv_has_precision = False
    if layers_p.exists():
        with open(layers_p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                system = row["system"]
                stage = row["stage"]
                cell = {
                    "value": float(row["mean_recall_ceiling"]),
                    "pool_size": float(row["mean_pool_size"]),
                    "n_queries": int(row["n_queries"]),
                    "kind": "pool",
                }
                if "mean_precision_ceiling" in row and row["mean_precision_ceiling"] != "":
                    cell["precision"] = float(row["mean_precision_ceiling"])
                    csv_has_precision = True
                if "mean_f1_ceiling" in row and row["mean_f1_ceiling"] != "":
                    cell["f1"] = float(row["mean_f1_ceiling"])
                cells[(system, stage)] = cell

    # Fallback: when the CSV is from an older run (no precision column),
    # compute Pool-Precision live for the post_cap stage from
    # `*_pool.jsonl`. Earlier stages (seeds/raw/post_temporal) stay
    # precision-less in this fallback path — those need a Stage 3 rerun.
    if not csv_has_precision:
        pool_metrics = _per_query_pool_metrics_cached()
        pool_precision_sum: dict[str, float] = {}
        pool_precision_n: dict[str, int] = {}
        for cells_by_sys in pool_metrics.values():
            for sys, m in cells_by_sys.items():
                p = m.get("precision")
                if p is None:
                    continue
                pool_precision_sum[sys] = pool_precision_sum.get(sys, 0.0) + p
                pool_precision_n[sys] = pool_precision_n.get(sys, 0) + 1
        for sys, n in pool_precision_n.items():
            c = cells.get((sys, "post_cap"))
            if c is None or n == 0:
                continue
            c["precision"] = pool_precision_sum[sys] / n

    # Ranking stages from summary.csv (mean_recall = recall@k)
    summary_p = metrics_dir() / "summary.csv"
    systems: set[str] = set()
    if summary_p.exists():
        with open(summary_p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                system = row["system"]
                systems.add(system)
                ranking = row["ranking"]
                k = int(row["k"])
                key = f"{ranking}_{k}"
                rank_cell = {
                    "value": float(row["mean_recall"]),
                    "pool_size": None,
                    "n_queries": int(row["n_queries"]),
                    "precision": float(row["mean_precision"]),
                    "mrr": float(row["mean_mrr"]),
                    "ndcg": float(row["mean_ndcg"]),
                    "hit_rate": float(row["hit_rate"]),
                    "kind": "rank",
                }
                if "mean_f1" in row and row["mean_f1"] != "":
                    rank_cell["f1"] = float(row["mean_f1"])
                cells[(system, key)] = rank_cell

    # Order systems sensibly
    canonical = ["rag", "rag_smart", "emb_1hop", "emb_2hop", "graph_1hop", "graph_2hop"]
    sys_list = [s for s in canonical if s in systems]
    sys_list += sorted(systems - set(canonical))

    return {
        "systems": sys_list,
        "pool_stages": pool_stages,
        "rank_stages": rank_stages,
        "cells": {f"{sys}|{stage}": v for (sys, stage), v in cells.items()},
    }


def latest_run_header() -> dict:
    """Build a compact run-metadata header for the pipeline matrix page.

    n_queries / lang_breakdown / gt_total / gt_unique are derived from
    the queries that were actually evaluated, not from the current
    `eval_queries.jsonl`. The file grows with auto-expand re-samples,
    but the metrics on disk reflect a frozen snapshot of whatever was
    in place when 03_compute_metrics last ran. Reading the per-query
    metric files (one line per evaluated query) gives the honest set.

    Falls back to the queries file when no metrics output exists yet
    (i.e. before the very first metrics run), so the page still
    renders something sensible during cold-start.
    """
    from .app_state import store
    runs = store.list_recent(limit=20)
    # Prefer a done run that actually computed metrics — skip_metrics
    # runs share their result files with the run that does compute,
    # so attributing the metrics to a metrics-running run is honest.
    latest = next(
        (r for r in runs if r.get("status") == "done" and not r.get("skip_metrics")),
        None,
    ) or next((r for r in runs if r.get("status") == "done"), None)

    by_lang: dict[str, int] = {}
    total_gt = 0
    n_queries = 0
    # Per-query metric files carry `language` and `n_ground_truth` directly.
    # That's the stable source — eval_queries.jsonl can drift from the
    # evaluated set when sampling re-runs change the candidate pool (it's
    # only a deterministic superset of past samples while metadata stays
    # constant; rebuilding decision_metadata.json invalidates that).
    for p in metrics_dir().glob("per_query_*.jsonl"):
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not rec.get("query_id"):
                        continue
                    lang = rec.get("language", "?")
                    by_lang[lang] = by_lang.get(lang, 0) + 1
                    total_gt += int(rec.get("n_ground_truth", 0) or 0)
                    n_queries += 1
        except OSError:
            continue
        break  # one per_query file is enough

    if n_queries == 0:
        # No metrics yet — fall back to current file as a placeholder so
        # the page still renders something during cold start.
        for q in load_queries():
            by_lang[q.get("language", "?")] = by_lang.get(q.get("language", "?"), 0) + 1
            total_gt += len(q.get("ground_truth_cases", []))
            n_queries += 1

    return {
        "run": latest,
        "n_queries": n_queries,
        "lang_breakdown": by_lang,
        "gt_total": total_gt,
    }


def downloadable_artifacts() -> list[dict]:
    """List of {name, path, size_bytes, mtime} for all eval artifacts."""
    items = []
    for base in [metrics_dir(), results_dir(), _eval_dir()]:
        if not base.exists():
            continue
        for p in sorted(base.glob("*")):
            if p.is_file() and p.suffix in {".csv", ".jsonl", ".json"}:
                st = p.stat()
                items.append({
                    "name": p.name,
                    "rel_path": str(p.relative_to(_eval_dir())),
                    "size_bytes": st.st_size,
                    "mtime": st.st_mtime,
                })
    return items
