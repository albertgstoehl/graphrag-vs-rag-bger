from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse

from ..app_state import executor, store


router = APIRouter()


@router.get("/runs/{run_id}/events")
async def run_events(run_id: int, request: Request):
    row = store.get(run_id)
    if not row:
        raise HTTPException(404, "Run not found")

    broker = executor.get_broker(run_id)
    log_path = executor.log_path(run_id)

    async def event_generator():
        # If the run is no longer active, replay only the LAST 500 lines of
        # the log file as "log" events (full historical view is available via
        # /runs/{id}/log.txt — streaming the whole file freezes the browser
        # for any sizeable run).
        TAIL = 500
        if broker is None:
            if log_path.exists():
                # Read last TAIL lines without slurping the whole file.
                from collections import deque
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    last_lines = deque(f, maxlen=TAIL)
                for line in last_lines:
                    if await request.is_disconnected():
                        return
                    yield {"event": "log",
                           "data": json.dumps({"line": line.rstrip("\n")})}
            yield {"event": "done", "data": json.dumps({"status": row["status"]})}
            return

        async for event in broker.subscribe():
            if await request.is_disconnected():
                return
            yield {
                "event": event.get("type", "event"),
                "data": json.dumps(event),
            }
            if event.get("type") == "stream_closed":
                return

    return EventSourceResponse(event_generator())
