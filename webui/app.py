"""FastAPI entrypoint for kg-rag-control."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .app_state import executor, store
from .routes import runs as runs_routes
from .routes import sse as sse_routes
from .routes import metrics as metrics_routes
from .routes import inspector as inspector_routes
from .routes import settings as settings_routes
from .routes import graph_overview as graph_overview_routes


app = FastAPI(title="kg-rag-control", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "busy": executor.is_busy()}


@app.get("/api/state")
def state():
    from . import inspector_data
    return {
        "busy": executor.is_busy(),
        "current_run_id": executor._current_run_id,
        "recent": store.list_recent(limit=5),
        "metadata_cache_size": len(inspector_data._METADATA_CACHE),
    }


_WEBUI_DIR = Path(__file__).resolve().parent
app.mount(
    "/static",
    StaticFiles(directory=str(_WEBUI_DIR / "static")),
    name="static",
)

app.include_router(runs_routes.router)
app.include_router(sse_routes.router)
app.include_router(metrics_routes.router)
app.include_router(inspector_routes.router)
app.include_router(settings_routes.router)
app.include_router(graph_overview_routes.router)
