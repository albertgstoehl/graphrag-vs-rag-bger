from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import metrics_reader
from ..app_state import templates


router = APIRouter()


SYSTEMS = ["rag", "rag_smart", "emb_1hop", "emb_2hop", "graph_1hop", "graph_2hop"]
RANKINGS = ["cosine", "cross_encoder", "indegree"]
K_VALUES = [5, 10, 15, 20]
RANKED_METRIC_FIELDS = ("recall", "precision", "mrr", "ndcg",
                       "nearness_score", "nearness_score_undirected")
POOL_METRIC_FIELDS = ("recall", "precision")


@router.get("/metrics", response_class=HTMLResponse)
def metrics_summary(request: Request):
    rows = metrics_reader.summary()
    rows_by_lang = metrics_reader.summary_by_language()
    return templates.TemplateResponse(
        request, "metrics_summary.html",
        {"rows": rows, "rows_by_lang": rows_by_lang, "systems": SYSTEMS,
         "rankings": RANKINGS, "k_values": K_VALUES},
    )


@router.get("/metrics/recall-ceiling", response_class=HTMLResponse)
def metrics_ceiling(request: Request):
    return templates.TemplateResponse(
        request, "metrics_ceiling.html",
        {"matrix": metrics_reader.pipeline_matrix(),
         "header": metrics_reader.latest_run_header(),
         "systems": SYSTEMS},
    )


@router.get("/metrics/graph-nearness", response_class=HTMLResponse)
def metrics_nearness(request: Request):
    return templates.TemplateResponse(
        request, "metrics_nearness.html",
        {"rows": metrics_reader.summary(), "systems": SYSTEMS,
         "k_values": K_VALUES},
    )


@router.get("/api/per-query")
def api_per_query(mode: str = "ranked",
                  ranking: str = "cross_encoder",
                  k: int = 10,
                  language: str = "all"):
    """Per-query rows feeding the drilldown panel embedded under each
    aggregate metrics view.

    `mode='ranked'`: rows for one (ranking, k); each row has values per
    system per ranked metric
    (recall/precision/mrr/ndcg/nearness_score/nearness_score_undirected).

    `mode='pool'`: rows of per-query Pool-Recall + Pool-Precision
    (post_cap pool ∩ GT, post_cap |pool|) per system. Ranking-independent.

    `language`: optional filter ('all', 'de', 'fr', 'it'). Any value other
    than 'all' restricts the row set to that language. Filter happens on
    the already-loaded qid_meta dict — no IO penalty.

    Both modes piggyback on caches already preloaded at boot
    (`latest_per_query_metric_index`, `per_query_pool_metrics`), so the
    endpoint is dict-lookup fast — no IO.
    """
    if mode not in ("ranked", "pool"):
        raise HTTPException(400, f"unknown mode {mode}")

    idx = metrics_reader.latest_per_query_metric_index()
    if not idx:
        return JSONResponse({"rows": [], "systems": SYSTEMS, "mode": mode,
                             "ranking": ranking, "k": k})

    # Pull per-qid metadata (lang/year/n_gt) from any bucket — they all
    # share the qid set.
    qid_meta: dict[str, dict] = {}
    for bucket in idx.values():
        for qid, rec in bucket.items():
            qid_meta[qid] = {
                "language": rec.get("language"),
                "year": rec.get("year"),
                "n_gt": int(rec.get("n_ground_truth", 0) or 0),
            }
        break

    # Drop qids that aren't in eval_queries.jsonl (so a row click can't 404
    # on /inspector/{qid}). Same self-protection inspector.py used to do.
    available = {q["query_id"] for q in metrics_reader.load_queries()}
    qid_meta = {qid: m for qid, m in qid_meta.items() if qid in available}

    # Optional language filter — restricts the row set to one of de/fr/it.
    # Stage 3's per_query records carry `language` directly, so this is a
    # pure in-memory filter, no re-read.
    if language and language != "all":
        qid_meta = {
            qid: m for qid, m in qid_meta.items() if m.get("language") == language
        }

    if mode == "pool":
        per_q = metrics_reader.per_query_pool_metrics()
        rows = []
        for qid, meta in qid_meta.items():
            cells = per_q.get(qid, {})
            values: dict[str, dict] = {}
            for s in SYSTEMS:
                m = cells.get(s)
                if m:
                    values[s] = {f: m.get(f) for f in POOL_METRIC_FIELDS}
                else:
                    values[s] = {f: None for f in POOL_METRIC_FIELDS}
            rows.append({
                "query_id": qid,
                "language": meta["language"],
                "year": meta["year"],
                "n_gt": meta["n_gt"],
                "values": values,
            })
        return JSONResponse({"rows": rows, "systems": SYSTEMS, "mode": mode,
                             "metric_fields": list(POOL_METRIC_FIELDS)})

    # mode == "ranked"
    if ranking not in RANKINGS or k not in K_VALUES:
        raise HTTPException(400, f"bad ranking/k: {ranking}/{k}")
    # If Stage 3 hasn't produced any per_query file for this (ranking, k)
    # yet — possible mid-run, since the writer streams cosine first then
    # indegree then cross_encoder — the table would otherwise render 3000
    # rows of dashes. Surface an explicit empty result instead.
    if not any(idx.get((s, ranking, k)) for s in SYSTEMS):
        return JSONResponse({"rows": [], "systems": SYSTEMS, "mode": mode,
                             "ranking": ranking, "k": k,
                             "metric_fields": list(RANKED_METRIC_FIELDS)})
    rows = []
    for qid, meta in qid_meta.items():
        values: dict[str, dict] = {}
        for s in SYSTEMS:
            rec = idx.get((s, ranking, k), {}).get(qid)
            if rec:
                values[s] = {f: rec.get(f) for f in RANKED_METRIC_FIELDS}
            else:
                values[s] = {f: None for f in RANKED_METRIC_FIELDS}
        rows.append({
            "query_id": qid,
            "language": meta["language"],
            "year": meta["year"],
            "n_gt": meta["n_gt"],
            "values": values,
        })
    return JSONResponse({"rows": rows, "systems": SYSTEMS, "mode": mode,
                         "ranking": ranking, "k": k,
                         "metric_fields": list(RANKED_METRIC_FIELDS)})


@router.get("/downloads", response_class=HTMLResponse)
def downloads_page(request: Request):
    return templates.TemplateResponse(
        request, "downloads.html",
        {"artifacts": metrics_reader.downloadable_artifacts()},
    )


@router.get("/downloads/{rel_path:path}")
def download_file(rel_path: str):
    import os
    base = Path(os.environ.get("EVAL_DIR", "/app/data/eval"))
    target = (base / rel_path).resolve()
    # Prevent path traversal
    if base.resolve() not in target.parents and target != base.resolve():
        raise HTTPException(403, "Path traversal refused")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(target, filename=target.name)
