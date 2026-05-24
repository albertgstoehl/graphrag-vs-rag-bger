"""Pipeline executor: spawns each stage in its own subprocess, streams
events + log lines back to the UI via asyncio pub/sub.

Each stage runs as `multiprocessing.Process` so memory is released between
stages and crashes can't take down the UI. Progress events go through an
`mp.Queue`; stdout/stderr are captured via a pipe.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from .db import RunStore


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPTS = REPO_ROOT / "scripts" / "eval"


class _LogToQueue:
    """File-like wrapper that forwards stdout/stderr lines to an mp.Queue
    as ``{"type":"log","line":...}`` events."""

    def __init__(self, queue: mp.Queue, stream: str):
        self._queue = queue
        self._stream = stream
        self._buf = ""

    def write(self, data: str) -> int:
        self._buf += data
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            try:
                self._queue.put_nowait({"type": "log", "line": line, "stream": self._stream})
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        if self._buf:
            try:
                self._queue.put_nowait({"type": "log", "line": self._buf, "stream": self._stream})
            except Exception:
                pass
            self._buf = ""

    def isatty(self) -> bool:
        return False


def _stage_worker(
    stage: str,
    kwargs: dict,
    event_queue: mp.Queue,
    cancel_flag,
) -> None:
    """Executed in a child process. Loads the stage module and calls run()."""
    import importlib.util
    import logging as _logging

    # Redirect stdout/stderr so every print/log line flows through the queue
    # to the UI. Keep raw buffers pointing at the original fds so tqdm still
    # renders a progress bar in the pod's stdout (visible via kubectl logs).
    sys.stdout = _LogToQueue(event_queue, "stdout")
    sys.stderr = _LogToQueue(event_queue, "stderr")
    # Re-wire root logger handlers to use the queue-backed stderr.
    for h in list(_logging.root.handlers):
        if isinstance(h, _logging.StreamHandler):
            h.setStream(sys.stderr)

    script_map = {
        "sample":    EVAL_SCRIPTS / "01_sample_queries.py",
        "retrieval": EVAL_SCRIPTS / "02_run_retrieval.py",
        "metrics":   EVAL_SCRIPTS / "03_compute_metrics.py",
    }
    script_path = script_map[stage]
    spec = importlib.util.spec_from_file_location(f"stage_{stage}", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def on_event(event: dict) -> None:
        try:
            event_queue.put_nowait(event)
        except Exception:
            pass

    def cancel_check() -> bool:
        return bool(cancel_flag.value)

    try:
        mod.run(on_event=on_event, cancel_check=cancel_check, **kwargs)
        event_queue.put({"type": "stage_exit", "stage": stage, "exit_code": 0})
    except SystemExit as exc:
        # Pipeline scripts call sys.exit(1) on missing inputs.
        code = int(exc.code) if isinstance(exc.code, int) else 1
        event_queue.put({"type": "error", "stage": stage,
                         "message": f"SystemExit({exc.code})",
                         "traceback": traceback.format_exc()})
        event_queue.put({"type": "stage_exit", "stage": stage, "exit_code": code})
    except Exception as exc:
        tb = traceback.format_exc()
        event_queue.put({"type": "error", "stage": stage, "message": str(exc), "traceback": tb})
        event_queue.put({"type": "stage_exit", "stage": stage, "exit_code": 1})
        raise


class RunBroker:
    """In-memory pub/sub for a single run's events.

    Multiple SSE subscribers can each get every event. History is kept so
    late joiners replay from the start (bounded to keep memory sane).
    """

    MAX_HISTORY = 300

    def __init__(self):
        self._history: list[dict] = []
        self._subscribers: list[asyncio.Queue] = []
        self._done = False

    def publish(self, event: dict) -> None:
        self._history.append(event)
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def close(self) -> None:
        self._done = True
        self.publish({"type": "stream_closed"})

    async def subscribe(self):
        q: asyncio.Queue = asyncio.Queue()
        # Replay history first
        for ev in self._history:
            await q.put(ev)
        if not self._done:
            self._subscribers.append(q)
        try:
            while True:
                ev = await q.get()
                yield ev
                if ev.get("type") == "stream_closed":
                    break
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)


class Executor:
    """Manages a single running pipeline at a time."""

    def __init__(self, store: RunStore, log_dir: Path):
        self.store = store
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_run_id: Optional[int] = None
        self._current_broker: Optional[RunBroker] = None
        self._current_process: Optional[mp.Process] = None
        self._current_cancel_flag = None
        self._lock = asyncio.Lock()

    def is_busy(self) -> bool:
        return self._current_run_id is not None

    def get_broker(self, run_id: int) -> Optional[RunBroker]:
        if run_id == self._current_run_id:
            return self._current_broker
        return None

    def log_path(self, run_id: int) -> Path:
        return self.log_dir / f"run_{run_id}.log"

    async def start(
        self,
        *,
        skip_sample: bool,
        skip_retrieval: bool,
        skip_metrics: bool,
        systems: list[str],
        rankings: list[str],
        k_values: list[int],
        dry_run: bool = False,
        query_limit: int = 0,
        per_language_n: int = 0,
        resume: bool = True,
    ) -> int:
        async with self._lock:
            if self.is_busy():
                raise RuntimeError(
                    f"Another run is in progress (#{self._current_run_id})"
                )
            run_id = self.store.create_run(
                skip_sample=skip_sample,
                skip_retrieval=skip_retrieval,
                skip_metrics=skip_metrics,
                systems=systems, rankings=rankings, k_values=k_values,
                log_path=str(self.log_path(0)),  # placeholder, updated below
                query_limit=query_limit,
                resume=resume,
            )
            log_path = self.log_path(run_id)
            self._current_run_id = run_id
            self._current_broker = RunBroker()
            self._current_cancel_flag = mp.Value("i", 0)

        # Kick off pipeline as a background task
        asyncio.create_task(self._run_pipeline(
            run_id=run_id,
            log_path=log_path,
            skip_sample=skip_sample,
            skip_retrieval=skip_retrieval,
            skip_metrics=skip_metrics,
            systems=systems,
            k_values=k_values,
            dry_run=dry_run,
            query_limit=query_limit,
            per_language_n=per_language_n,
            resume=resume,
        ))
        return run_id

    async def abort(self, run_id: int) -> bool:
        if run_id != self._current_run_id:
            return False
        if self._current_cancel_flag is not None:
            self._current_cancel_flag.value = 1
            if self._current_broker:
                self._current_broker.publish(
                    {"type": "abort_requested", "ts": datetime.utcnow().isoformat()}
                )
        return True

    async def _run_pipeline(
        self, *,
        run_id: int,
        log_path: Path,
        skip_sample: bool,
        skip_retrieval: bool,
        skip_metrics: bool,
        systems: list[str],
        k_values: list[int],
        dry_run: bool,
        query_limit: int = 0,
        per_language_n: int = 0,
        resume: bool = True,
    ) -> None:
        self.store.mark_running(run_id)
        broker = self._current_broker
        broker.publish({"type": "run_started", "run_id": run_id})

        final_status = "done"
        final_exit = 0
        final_error: Optional[str] = None
        log_fh = open(log_path, "w", buffering=1)  # line-buffered

        def log_line(line: str) -> None:
            log_fh.write(line.rstrip("\n") + "\n")
            broker.publish({"type": "log", "line": line.rstrip("\n")})

        try:
            sample_kwargs: dict = {"dry_run": dry_run}
            if per_language_n and per_language_n > 0:
                sample_kwargs["per_language_n"] = per_language_n
            stages = [
                ("sample",    skip_sample,    sample_kwargs),
                ("retrieval", skip_retrieval, {"systems": systems, "dry_run": dry_run,
                                                "query_limit": query_limit,
                                                "resume": resume}),
                ("metrics",   skip_metrics,   {"dry_run": dry_run}),
            ]
            for stage_name, skip, kwargs in stages:
                if skip:
                    log_line(f"=== SKIP {stage_name} ===")
                    broker.publish({"type": "stage_skipped", "stage": stage_name})
                    continue
                if self._current_cancel_flag.value:
                    log_line("=== ABORTED by user ===")
                    final_status = "aborted"
                    break
                log_line(f"=== {stage_name.upper()} started ===")
                exit_code = await self._run_stage(
                    stage_name, kwargs, broker, log_line,
                )
                if exit_code != 0:
                    final_status = "failed"
                    final_exit = exit_code
                    final_error = f"Stage {stage_name} exited with code {exit_code}"
                    log_line(f"=== {stage_name.upper()} FAILED (exit={exit_code}) ===")
                    break
                log_line(f"=== {stage_name.upper()} finished ===")
        except asyncio.CancelledError:
            # Pod is shutting down — uvicorn sent SIGTERM and asyncio is
            # cancelling pending tasks. Without this clause, CancelledError
            # (a BaseException) skips the `except Exception` below, falls
            # through to `finally`, and the run gets marked 'done' with the
            # default status. Mark it 'interrupted' instead so the next
            # session can offer a Continue path; re-raise so asyncio still
            # sees the task as cancelled.
            final_status = "interrupted"
            final_exit = 1
            final_error = "interrupted by pod shutdown — restart in progress"
            try:
                log_line("=== INTERRUPTED by pod shutdown ===")
            except Exception:
                pass
            raise
        except Exception as exc:
            final_status = "failed"
            final_exit = 1
            final_error = str(exc)
            log_line(f"=== executor exception: {exc} ===")
            log_line(traceback.format_exc())
        finally:
            log_fh.close()
            self.store.mark_finished(
                run_id,
                status=final_status,
                exit_code=final_exit,
                error_message=final_error,
            )
            # Drop the metrics_reader caches — the per_query_*.jsonl and
            # *_pool.jsonl files just got their final state for this run,
            # so any previously cached aggregate is now stale.
            try:
                from . import metrics_reader
                metrics_reader.invalidate_caches()
            except Exception:
                pass
            broker.publish({"type": "run_finished", "status": final_status})
            broker.close()
            self._current_run_id = None
            self._current_broker = None
            self._current_process = None
            self._current_cancel_flag = None

    async def _run_stage(
        self,
        stage: str,
        kwargs: dict,
        broker: RunBroker,
        log_line,
    ) -> int:
        """Spawn a stage in a child process and stream its events + logs."""
        event_queue: mp.Queue = mp.Queue()
        cancel_flag = self._current_cancel_flag

        # Pipe stdout/stderr from the child back to us via a temp file.
        # multiprocessing.Process doesn't give us direct stdout access, so we
        # just rely on the structured events for progress + the child writing
        # its tqdm + logging to its own stdout which is inherited from us
        # (the uvicorn worker's stdout). For now, we only capture structured
        # events; log lines go through emit({"type":"log"}) if the stage wants.
        # For richer log capture we'd need to spawn via subprocess with PIPE —
        # acceptable compromise for v1.

        proc = mp.Process(
            target=_stage_worker,
            args=(stage, kwargs, event_queue, cancel_flag),
        )
        proc.start()
        self._current_process = proc

        exit_code = 0
        loop = asyncio.get_event_loop()

        async def pump_events() -> int:
            local_exit = 0
            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 1.0)
                except Exception:
                    # Queue empty + timeout. Check if process still alive.
                    if not proc.is_alive():
                        break
                    continue
                if event is None:
                    continue
                # log_line() persists to disk AND publishes via the broker.
                # For non-log events we still need an explicit publish.
                if event.get("type") == "log":
                    log_line(event.get("line", ""))
                else:
                    broker.publish({**event, "ts": datetime.utcnow().isoformat()})
                if event.get("type") == "stage_exit":
                    local_exit = int(event.get("exit_code", 0))
                    break
                if event.get("type") == "error":
                    tb = event.get("traceback", "")
                    for line in tb.splitlines():
                        log_line(line)
            return local_exit

        try:
            exit_code = await pump_events()
        finally:
            # Give child a moment to exit cleanly
            await loop.run_in_executor(None, proc.join, 10.0)
            if proc.is_alive():
                proc.terminate()
                await loop.run_in_executor(None, proc.join, 5.0)
                if proc.is_alive():
                    proc.kill()
            # Prefer the child's OS exit code only if we didn't already get a
            # structured exit event with a non-zero code.
            if exit_code == 0 and proc.exitcode not in (None, 0):
                exit_code = int(proc.exitcode)

        return int(exit_code or 0)
