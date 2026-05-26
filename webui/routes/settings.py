"""Editable runtime settings — aiserver host + ports.

Persisted via SettingsStore (SQLite on the webui-state PVC) so they
survive pod restarts. On save we re-apply to `os.environ` and clear the
inspector's QdrantClient cache so the next inspector request rebuilds
the client against the new host."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from typing import Annotated

from .. import inspector_data
from ..app_state import settings, templates


router = APIRouter()


@router.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "values": settings.get_all(),
            "derived": settings.derive_env(),
            "saved": request.query_params.get("saved") == "1",
        },
    )


@router.post("/settings")
async def settings_save(
    aiserver_host: Annotated[str, Form()],
    qdrant_port: Annotated[str, Form()],
    tei_embed_port: Annotated[str, Form()],
    tei_rerank_ports: Annotated[str, Form()],
):
    settings.set_many({
        "aiserver_host": aiserver_host,
        "qdrant_port": qdrant_port,
        "tei_embed_port": tei_embed_port,
        "tei_rerank_ports": tei_rerank_ports,
    })
    settings.apply_to_env()
    # Drop the cached QdrantClient so the next inspector request reconnects
    # against the new host.
    try:
        inspector_data.qdrant.cache_clear()
    except Exception:
        pass
    return RedirectResponse(url="/settings?saved=1", status_code=303)
