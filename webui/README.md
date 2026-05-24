# webui

Web UI for driving the evaluation pipeline from a browser and inspecting
per-query results. FastAPI + HTMX + Alpine.js + Tailwind, SQLite run
history, Server-Sent-Events for live logs, Chart.js for the recall-ceiling
and graph-nearness plots.

## What this contains

- `app.py` — FastAPI entry (`uvicorn webui.app:app`)
- `app_state.py` — shared singletons (`store`, `executor`, `templates`)
- `db.py` — SQLite schema and CRUD for the `runs` table
- `executor.py` — spawns pipeline stages as child processes, pipes
  structured events plus stdout/stderr back to the SSE broker and log file
- `metrics_reader.py` — parses `data/eval/metrics/*.csv` and
  `results/*.jsonl` so metric pages render without touching the pipeline
- `routes/` — one module each for runs, SSE stream, metrics, settings
- `templates/` — Jinja2 templates (base, dashboard, run-list, run-detail,
  metrics summary/ceiling/nearness/per-query, downloads, settings)
- `Dockerfile` — multi-stage Python 3.12-slim build, bakes in the full repo
- `requirements.txt` — UI-level deps (FastAPI, Uvicorn, Jinja2, SSE)

Pipeline deps (`qdrant-client`, `datasets`, `networkx`, `requests`,
`tqdm`) are installed in the Dockerfile but not in `requirements.txt`, so
local dev can run the UI alone with an existing pipeline venv.

## Local dev

```bash
# from repo root
pip install -r webui/requirements.txt

WEBUI_STATE_DIR=/tmp/kgrag-state \
EVAL_DIR=./data/eval \
QDRANT_HOST=localhost \
TEI_HOST=localhost \
TEI_RERANK_HOST=localhost:8011 \
python -m uvicorn webui.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open `http://127.0.0.1:8000/`. Hosts and ports are also editable at
runtime under `/settings`, which writes to the state directory and pushes
the values into `os.environ` before each pipeline run.

## Build the container image

```bash
docker build -t kg-rag-control:latest -f webui/Dockerfile .
```

Image weighs roughly 560 MB, most of it pyarrow plus networkx
dependencies. To run it standalone:

```bash
docker run --rm -p 8000:8000 \
  -e QDRANT_HOST=host.docker.internal \
  -e TEI_HOST=host.docker.internal \
  -v $(pwd)/data:/app/data \
  kg-rag-control:latest
```

## How the executor talks to the pipeline

Each of `scripts/eval/0{1,2,3}_*.py` exposes an importable `run()`
function:

```python
def run(
    # ... stage-specific kwargs ...,
    on_event=None,     # callback receiving dict events
    cancel_check=None, # callable -> bool, True means "stop soon"
) -> None: ...
```

The executor spawns a `multiprocessing.Process` per stage and wires up:

- Event queue. `on_event` pushes
  `{"type": "stage_started"|"progress"|"stage_done"|"error", ...}` dicts
- Log capture. `sys.stdout` and `sys.stderr` in the child are replaced by
  `_LogToQueue` so every `print()` and `logging` line becomes a
  `{"type": "log", "line": ...}` event
- Cancellation. `cancel_flag` is a `mp.Value('i')`. The executor sets it
  to 1 on abort, stages check `cancel_check()` in hot loops

Events are broadcast to two places concurrently:

1. `RunBroker`, an in-memory pub/sub that SSE subscribers read from
2. `logs/run_{id}.log` in the state directory so replays work after the
   run ends

## Deviation from the shell wrapper

The UI does not call `scripts/eval/04_run_pipeline.sh`. The shell wrapper
remains as the CLI entry point for ad-hoc runs. The UI goes directly
through the Python `run()` functions because:

- Fragile stdout-grepping would be the only way to get progress from a
  bash subprocess
- `SIGTERM` to bash orphans its Python children, creating pipeline
  corruption risk
- `python3 0X_foo.py` startup reloads qdrant_client and networkx every
  stage. In-process import amortises the cost once
- Tracebacks surface in the UI instead of being consumed by shell
- CLI-arg quoting is brittle, kwargs are type-safe

## Required services

The UI assumes the following are reachable at the hosts and ports
configured in `/settings`:

- Qdrant on port 6333, holding the `bger` collection with BGE-M3 vectors
- TEI embed on port 8010, serving `BAAI/bge-m3`
- TEI rerank on ports 8011-8014 (one or more replicas), serving
  `BAAI/bge-reranker-v2-m3`

See the repository README for a docker-compose snippet that brings these
up locally.
