"""Shared application singletons. Imported by app.py and the route modules.

Avoids circular imports between routes and the executor/store."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.templating import Jinja2Templates

from .db import RunStore
from .executor import Executor
from .settings import SettingsStore


STATE_DIR = Path(os.environ.get("WEBUI_STATE_DIR", "/app/webui/state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = STATE_DIR / "logs"
DB_PATH = STATE_DIR / "runs.db"
SETTINGS_DB_PATH = STATE_DIR / "settings.db"

_WEBUI_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_WEBUI_DIR / "templates"))

# Settings first — they push QDRANT_HOST / TEI_* into os.environ before any
# other module reads those env vars. Child run processes inherit env from
# this process, so updating settings affects the next pipeline run.
settings = SettingsStore(SETTINGS_DB_PATH)
settings.apply_to_env()

store = RunStore(DB_PATH)
store.interrupt_all_running()  # clean up orphaned runs on startup

executor = Executor(store=store, log_dir=LOG_DIR)


# Preload heavy on-disk caches in a background thread so the first
# user-visible request doesn't pay a multi-second cold-load penalty.
# Loading happens in a daemon thread so app boot is unblocked; if a
# request lands before the thread finishes, the cached function just
# blocks until done.
#
# What gets warmed:
#   • inspector_data.graph()                — 100 MB pickle, ~4 s
#   • metrics_reader.latest_per_query_metric_index() — ~850 ms for 3k qids
#   • metrics_reader.per_query_pool_recall()         — ~3.9 s for 3k qids
#   • inspector_data.preload_all_metadata()  — ~60 k chunk-0 entries from
#     Qdrant, ~5 min cold. After this, every metadata_for() is a dict
#     hit (~0 ms), insulating the inspector from pipeline-induced
#     Qdrant load. Done last because it's the longest-running and
#     non-blocking — partial completion is still useful.
def _preload_caches() -> None:
    import sys
    def _p(msg: str) -> None:
        print(f"[preload] {msg}", file=sys.stderr, flush=True)
    try:
        from . import inspector_data, metrics_reader
        _p("citation graph …")
        inspector_data.graph()
        _p("per-query metric index …")
        metrics_reader.latest_per_query_metric_index()
        _p("pool recall …")
        metrics_reader.per_query_pool_recall()
        # Pre-warm the per-file qid indices used by query_by_id /
        # retrieved_for / pool_for / seeds_for / cross_encoder_scores_for.
        # Each file is parsed once into {qid: record} so per-qid lookups
        # become O(1) dict access. ~50 files × ~100-300 ms each = ~10 s
        # at boot, eliminates the per-qid file-scan cost (was ~650 ms
        # combined for the 5 functions on a fresh qid).
        _p("metrics_reader file indices …")
        from pathlib import Path
        from webui.metrics_reader import _index_jsonl_by_qid, _eval_dir, results_dir
        try:
            _index_jsonl_by_qid(str(_eval_dir() / "eval_queries.jsonl"))
            for p in sorted(results_dir().glob("*.jsonl")):
                if p.name == "query_embeddings.jsonl":
                    continue
                _index_jsonl_by_qid(str(p))
            _p("metrics_reader indices: done")
        except Exception as e:
            _p(f"metrics_reader index preload error: {e}")
        _p("bger metadata (this is the slow one) …")
        n = inspector_data.preload_all_metadata(log=_p)
        _p(f"done — {n} metadata entries cached")
    except Exception as e:
        _p(f"thread error: {e}")


import threading
threading.Thread(target=_preload_caches, daemon=True, name="cache-preload").start()
