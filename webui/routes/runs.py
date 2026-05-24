from __future__ import annotations

import json
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse, PlainTextResponse
from typing import Annotated

from ..app_state import templates, executor, store


router = APIRouter()

_EVAL_DIR = Path(os.environ.get("EVAL_DIR", "/app/data/eval"))


@lru_cache(maxsize=1)
def _candidate_pool_sizes() -> dict:
    """Per-language candidate pool from the strict-GT filter.

    Mirrors the filter in `01_sample_queries.py:295-348` exactly: a
    decision is a valid candidate iff it is in valid_ids, has
    metadata + language ∈ {de,fr,it} + date_ms, and has at least one
    case_to_case successor that is also in valid_ids and pre-dates it.

    Computing this takes ~7 s (loads the 100 MB graph, the metadata
    cache, and iterates 131k IDs). We cache the result for the
    process lifetime so the New-Run form is responsive after the
    first hit. The cache is invalidated on pod restart, which is
    when re-sampling would have changed anything anyway.
    """
    valid_ids_path = _EVAL_DIR / "valid_ids.json"
    metadata_path = _EVAL_DIR / "decision_metadata.json"
    graph_path = _EVAL_DIR / "citation_graph.pkl"
    if not (valid_ids_path.exists() and metadata_path.exists() and graph_path.exists()):
        # Fall back to a generous hard cap if any input is missing —
        # the form should still render rather than 500.
        return {"de": 5000, "fr": 5000, "it": 5000, "_estimate": True}
    with open(valid_ids_path) as f:
        valid_ids = set(json.load(f))
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    pool = {"de": 0, "fr": 0, "it": 0}
    for did in valid_ids:
        md = metadata.get(did)
        if not md:
            continue
        lang = md.get("language")
        if lang not in pool:
            continue
        qd = md.get("date_ms", 0)
        if not qd or did not in graph:
            continue
        for tgt, attrs in graph[did].items():
            if attrs.get("type") != "case_to_case":
                continue
            if tgt not in valid_ids:
                continue
            cd = metadata.get(tgt, {}).get("date_ms", 0)
            if cd and cd < qd:
                pool[lang] += 1
                break
    return pool


def _eval_queries_stats() -> dict:
    """Return live stats from eval_queries.jsonl + valid_ids.json.

    Drives the dynamic max + per-language hint on the New-Run form. Re-read
    on every request — eval_queries.jsonl is small (~5 MB) and rarely
    changes, so caching adds complexity without measurable wins.

    The strict-GT verification (every ground-truth case is in `valid_ids`)
    is normally guaranteed by `01_sample_queries.py:build_ground_truth`,
    which already filters GT to `valid_ids` at sampling time. We still
    verify here so a stale or hand-edited file shows a warning instead of
    silently producing un-evaluable queries.
    """
    queries_path = _EVAL_DIR / "eval_queries.jsonl"
    valid_ids_path = _EVAL_DIR / "valid_ids.json"
    if not queries_path.exists():
        pool = _candidate_pool_sizes()
        pool_max_stratified = min(pool["de"], pool["fr"], pool["it"]) * 3
        return {
            "total": 0, "per_lang": {}, "max_stratified": 0,
            "pool": pool, "pool_max_stratified": pool_max_stratified,
            "verified": 0, "all_verified": False, "missing": True,
        }
    valid_ids: set = set()
    if valid_ids_path.exists():
        with open(valid_ids_path) as f:
            valid_ids = set(json.load(f))
    by_lang: dict = {}
    verified = 0
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            lang = q.get("language", "??")
            by_lang[lang] = by_lang.get(lang, 0) + 1
            if valid_ids:
                gts = q.get("ground_truth_cases") or []
                if gts and all(g in valid_ids for g in gts):
                    verified += 1
    total = sum(by_lang.values())
    max_stratified = (min(by_lang.values()) * 3) if by_lang else 0
    pool = _candidate_pool_sizes()
    pool_max_stratified = min(pool["de"], pool["fr"], pool["it"]) * 3
    return {
        "total": total,
        "per_lang": by_lang,
        "max_stratified": max_stratified,
        "pool": pool,
        "pool_max_stratified": pool_max_stratified,
        "verified": verified,
        "all_verified": bool(valid_ids) and verified == total,
        "missing": False,
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    latest_active = store.latest_running()
    recent = store.list_recent(limit=10)
    return templates.TemplateResponse(
        request, "index.html",
        {"latest_active": latest_active, "recent": recent},
    )


@router.get("/runs/new", response_class=HTMLResponse)
def new_form(request: Request):
    return templates.TemplateResponse(
        request, "run_new.html",
        {"qstats": _eval_queries_stats()},
    )


def _normalise_run_params(query_limit: int, skip_sample: bool) -> tuple[int, int, bool]:
    """Apply the form's auto-expand rules.

    Returns ``(query_limit, per_language_n, effective_skip_sample)``:
        - ``query_limit`` floored to a multiple of 3 and clamped to the
          per-language candidate pool ceiling.
        - ``per_language_n`` non-zero whenever Stage 1 is going to run.
          Previously this was only set when ``needs_resample`` flipped true,
          which meant that if the user clicked "Run" with skip_sample=False
          and ql ≤ existing max_stratified, Stage 1 was invoked with
          per_language_n=0 and silently fell back to its hard-coded
          PER_LANGUAGE_N=500 default — overwriting eval_queries.jsonl with
          1500 rows regardless of what the user asked for (run #69 incident).
        - ``effective_skip_sample`` overrides the user's choice when
          expansion is needed — Stage 1 has to run regardless.

    Used by both POST /runs (form submission) and POST /runs/{id}/continue
    (resume of a previous interrupted/failed run) so both paths agree on
    when re-sampling is implicit.
    """
    ql = (max(0, int(query_limit or 0)) // 3) * 3
    stats = _eval_queries_stats()
    if ql > stats["pool_max_stratified"]:
        ql = stats["pool_max_stratified"]
    needs_resample = ql > stats["max_stratified"]
    effective_skip_sample = skip_sample and not needs_resample
    # Always derive per_language_n from the user's requested ql when Stage 1
    # is going to run. Setting it to 0 only when Stage 1 is skipped is a
    # safe no-op (Stage 1 isn't invoked anyway).
    per_language_n = (ql // 3) if not effective_skip_sample else 0
    return ql, per_language_n, effective_skip_sample


@router.post("/runs")
async def start_run(
    request: Request,
    systems: list[str] = Form(...),
    skip_sample: Annotated[str | None, Form()] = None,
    skip_retrieval: Annotated[str | None, Form()] = None,
    skip_metrics: Annotated[str | None, Form()] = None,
    dry_run: Annotated[str | None, Form()] = None,
    query_limit: Annotated[int, Form()] = 0,
    resume: Annotated[str | None, Form()] = None,
):
    if executor.is_busy():
        raise HTTPException(409, "Another run is already in progress")
    ql, per_language_n, effective_skip_sample = _normalise_run_params(
        query_limit=int(query_limit or 0),
        skip_sample=bool(skip_sample),
    )
    run_id = await executor.start(
        skip_sample=effective_skip_sample,
        skip_retrieval=bool(skip_retrieval),
        skip_metrics=bool(skip_metrics),
        systems=systems,
        rankings=["cosine", "cross_encoder", "indegree"],
        k_values=[5, 10, 15, 20],
        dry_run=bool(dry_run),
        query_limit=ql,
        per_language_n=per_language_n,
        resume=bool(resume),
    )
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.post("/runs/{run_id}/continue")
async def continue_run(run_id: int):
    """Spawn a fresh run with the prior run's config + resume=on.

    The new run reuses systems / skip_* / query_limit from the row we're
    continuing. Auto-expand still applies — if the file shrunk between
    runs and the original cap is now > the file's size, Stage 1 will
    re-trigger. The file's deterministic-superset property means past
    queries are still inside.
    """
    if executor.is_busy():
        raise HTTPException(409, "Another run is already in progress")
    row = store.get(run_id)
    if not row:
        raise HTTPException(404, f"Run {run_id} not found")
    if row.get("status") not in ("interrupted", "aborted", "failed"):
        raise HTTPException(
            400,
            "Only interrupted/aborted/failed runs can be continued; "
            f"run {run_id} is in status {row.get('status')!r}",
        )
    systems = [s for s in (row.get("systems") or "").split(",") if s]
    ql, per_language_n, effective_skip_sample = _normalise_run_params(
        query_limit=int(row.get("query_limit") or 0),
        skip_sample=bool(row.get("skip_sample")),
    )
    new_id = await executor.start(
        skip_sample=effective_skip_sample,
        skip_retrieval=bool(row.get("skip_retrieval")),
        skip_metrics=bool(row.get("skip_metrics")),
        systems=systems,
        rankings=["cosine", "cross_encoder", "indegree"],
        k_values=[5, 10, 15, 20],
        dry_run=False,
        query_limit=ql,
        per_language_n=per_language_n,
        resume=True,
    )
    return RedirectResponse(f"/runs/{new_id}", status_code=303)


@router.get("/runs", response_class=HTMLResponse)
def list_runs(request: Request):
    runs = store.list_recent(limit=200)
    return templates.TemplateResponse(request, "runs_list.html", {"runs": runs})


_STAGE_MARKER_RE = re.compile(r"=== (?:(SKIP)\s+(\w+)|(\w+)\s+(started|finished|FAILED))")


def _initial_stage_states(run: dict, log_path: Path) -> dict:
    """Initial stepper state, in the order Sample → Retrieval → Metrics.

    For finished runs we scan the log for the `=== STAGE started/finished
    ===` and `=== SKIP stage ===` markers the executor writes. That gives
    a precise picture even if the SSE log-tail replay (last 500 lines)
    misses the early stage transitions on very long runs.

    For live runs we still seed with skip-flags + 'pending'; the Alpine
    component updates state from `stage_started` / `stage_done` /
    `stage_skipped` SSE events as they come in.
    """
    stages: dict = {}
    for stage in ("sample", "retrieval", "metrics"):
        skipped = bool(run.get(f"skip_{stage}"))
        stages[stage] = {
            "state": "skipped" if skipped else "pending",
            "current": 0,
            "total": 0,
        }
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = _STAGE_MARKER_RE.search(line)
                    if not m:
                        continue
                    if m.group(1) == "SKIP":
                        stage = m.group(2).lower()
                        if stage in stages:
                            stages[stage]["state"] = "skipped"
                    else:
                        stage = m.group(3).lower()
                        action = m.group(4).lower()
                        if stage not in stages:
                            continue
                        if action == "started":
                            stages[stage]["state"] = "running"
                        elif action == "finished":
                            stages[stage]["state"] = "done"
                        elif action == "failed":
                            stages[stage]["state"] = "failed"
        except OSError:
            pass
    # Belt-and-braces: a 'done' run with no log still shows correct stepper.
    status = run.get("status")
    if status == "done":
        for s in stages.values():
            if s["state"] == "pending":
                s["state"] = "done"
    elif status in ("failed", "aborted", "interrupted"):
        # Terminal-failed run: nothing more is going to run for this row.
        # Anything still 'running' (the stage that got cut off) AND anything
        # still 'pending' (downstream stages that never started) collapse to
        # 'failed' so the stepper doesn't lie. The overall status (the pill
        # at the top of the page) carries the failed/aborted/interrupted
        # nuance; the stepper is just "which stages were attempted".
        for s in stages.values():
            if s["state"] in ("running", "pending"):
                s["state"] = "failed"
    return stages


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    row = store.get(run_id)
    if not row:
        raise HTTPException(404, f"Run {run_id} not found")
    is_current = executor._current_run_id == run_id
    initial_stages = _initial_stage_states(row, executor.log_path(run_id))
    return templates.TemplateResponse(
        request, "run_detail.html",
        {
            "run": row,
            "is_current": is_current,
            "initial_stages_json": json.dumps(initial_stages),
        },
    )


@router.post("/runs/{run_id}/abort")
async def abort_run(run_id: int):
    ok = await executor.abort(run_id)
    if not ok:
        raise HTTPException(400, "Run is not the active one or already finished")
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.get("/runs/{run_id}/log.txt", response_class=PlainTextResponse)
def run_log_raw(run_id: int):
    path = executor.log_path(run_id)
    if not path.exists():
        raise HTTPException(404, "No log yet")
    return path.read_text(encoding="utf-8", errors="replace")
