# webui — kg-rag-control

Web UI for driving the KG-RAG Legal evaluation pipeline from a browser.
FastAPI + HTMX + Alpine.js + Tailwind, SQLite run history, Server-Sent-Events
for live logs, Chart.js for the recall-ceiling / graph-nearness plots.

## What this contains

- `app.py` — FastAPI entry (`uvicorn webui.app:app`)
- `app_state.py` — shared singletons (`store`, `executor`, `templates`)
- `db.py` — SQLite schema + CRUD for the `runs` table
- `executor.py` — spawns pipeline stages as child processes, pipes structured
  events + stdout/stderr back to the SSE broker + log file
- `metrics_reader.py` — parses `data/eval/metrics/*.csv` and `results/*.jsonl`
  so metric pages can be rendered without touching the pipeline
- `routes/` — one module each for runs, SSE stream, metrics
- `templates/` — Jinja2 templates (base, dashboard, run-list, run-detail,
  metrics summary/ceiling/nearness/per-query, downloads)
- `Dockerfile` — multi-stage Python 3.12-slim build; bakes in the full repo
- `requirements.txt` — only UI-level deps (FastAPI, Uvicorn, Jinja2, SSE)

Pipeline deps (`qdrant-client`, `datasets`, `networkx`, `requests`, `tqdm`)
are installed inside the Dockerfile — not in `requirements.txt` — so local
dev can run either the UI alone (with an existing venv) or the full image.

## Local dev

```bash
# from repo root
cd ~/kg-rag-legal
.venv/bin/pip install -r webui/requirements.txt

# state + eval dirs (env-var driven so they don't clash with /app/* defaults)
WEBUI_STATE_DIR=/tmp/kgrag-state \
EVAL_DIR=./data/eval \
QDRANT_HOST=aiserver01 \
TEI_RERANK_HOST=aiserver01:8011 \
.venv/bin/python -m uvicorn webui.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open `http://127.0.0.1:8000/`. The UI will read metrics from the local
`data/eval/metrics/` if it exists.

## Build the container image

The image is sideloaded into k3s containerd on the host (no registry):

```bash
cd ~/kg-rag-legal
sudo docker build -t kg-rag-control:latest -f webui/Dockerfile .
sudo docker save kg-rag-control:latest | sudo k3s ctr images import -
```

Image ends up ~560 MB; most of it is pyarrow + networkx dependencies.

## Deploy to the local k3s cluster

Manifests + scripts live **outside** this repo at
`~/.openfang/apps/kg-rag-control/` so infra state doesn't clutter the thesis
sources.

```bash
~/.openfang/apps/kg-rag-control/scripts/build.sh   # build + sideload
~/.openfang/apps/kg-rag-control/scripts/deploy.sh  # kubectl apply
```

The UI then runs at `http://ubuntu-4gb-hel1-1:30846` (reachable via Tailscale).

## How the executor talks to the pipeline

Each of `scripts/eval/0{1,2,3}_*.py` exposes an importable `run()` function:

```python
def run(
    # … stage-specific kwargs …,
    on_event=None,     # callback receiving dict events
    cancel_check=None, # callable → bool; True means "stop soon"
) -> None: ...
```

The executor spawns a `multiprocessing.Process` per stage, wires up:

- **Event queue** — `on_event` pushes `{"type": "stage_started"|"progress"|"stage_done"|"error", …}` dicts
- **Log capture** — `sys.stdout` / `sys.stderr` in the child are replaced by
  `_LogToQueue` so every `print()` and `logging` line becomes a
  `{"type": "log", "line": ...}` event
- **Cancellation** — `cancel_flag` is a `mp.Value('i')`; the executor sets
  it to 1 on abort, stages check `cancel_check()` in hot loops

Events are broadcast to two places concurrently:

1. `RunBroker` — in-memory pub/sub that SSE subscribers read from
2. `logs/run_{id}.log` on the `webui-state` PVC so replays work after the
   run ends

## Deviation from the shell wrapper

The UI does **not** call `scripts/eval/04_run_pipeline.sh`. That shell
wrapper remains as the CLI entry point for ad-hoc runs. The UI goes
directly through the Python `run()` functions because:

- Fragile stdout-grepping would be the only way to get progress from a bash
  subprocess
- `SIGTERM` to bash orphans its Python children (pipeline corruption risk)
- `python3 0X_foo.py` startup reloads qdrant_client / networkx every stage —
  in-process import amortises the cost once
- Tracebacks surface directly in the UI instead of being consumed by shell
- CLI-arg quoting (`--systems rag,graph_1hop`) is brittle; kwargs are type-safe

## Integration points outside the repo

- **aiserver01 services** (defined in `/data/thesis/k8s/` on aiserver01):
  - Qdrant 6333
  - TEI-embed 8010 (bge-m3)
  - TEI-rerank 8011 (bge-reranker-v2-m3) — added when the UI was built
- **k3s manifests** + **build/deploy scripts**:
  `~/.openfang/apps/kg-rag-control/{k8s,scripts}/`
- **Ops wiki page**: `~/.openfang/wikis/engineer/pages/kg-rag-control.md`
