"""Persistent application settings, editable from the UI.

Stored in `webui-state` PVC alongside `runs.db`. On startup these values
are pushed into `os.environ` so the existing env-var contracts in
`inspector_data.py` and `scripts/eval/02_run_retrieval.py` continue to
work — the settings layer is just a runtime-editable source for those
env vars.

Settings only take effect on:
  - the next pipeline run (env is inherited by each child stage process);
  - the next inspector Qdrant call (we clear the QdrantClient cache when
    settings change).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


_DEFAULTS: dict[str, str] = {
    # aiserver01-1 on Tailscale (replaces 100.95.37.85). Editable in /settings.
    "aiserver_host": "100.116.242.70",
    "qdrant_port": "6333",
    "tei_embed_port": "8010",
    "tei_rerank_ports": "8011,8012,8013,8014",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class SettingsStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            for k, default in _DEFAULTS.items():
                row = c.execute(
                    "SELECT value FROM app_settings WHERE key=?", (k,)
                ).fetchone()
                if row is None:
                    seed = _seed_from_env(k) or default
                    c.execute(
                        "INSERT INTO app_settings (key, value) VALUES (?, ?)",
                        (k, seed),
                    )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_all(self) -> dict[str, str]:
        with self._conn() as c:
            rows = c.execute("SELECT key, value FROM app_settings").fetchall()
            d = {r["key"]: r["value"] for r in rows}
            for k, default in _DEFAULTS.items():
                d.setdefault(k, default)
            return d

    def set_many(self, values: dict[str, str]) -> None:
        with self._conn() as c:
            for k, v in values.items():
                if k not in _DEFAULTS:
                    continue
                c.execute(
                    "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, str(v).strip()),
                )

    def derive_env(self) -> dict[str, str]:
        s = self.get_all()
        host = s["aiserver_host"].strip()
        rerank_ports = [p.strip() for p in s["tei_rerank_ports"].split(",") if p.strip()]
        urls = ",".join(f"http://{host}:{p}/rerank" for p in rerank_ports)
        return {
            "QDRANT_HOST": host,
            "QDRANT_PORT": s["qdrant_port"].strip(),
            "TEI_HOST": host,
            "TEI_PORTS": s["tei_embed_port"].strip(),
            "TEI_RERANK_HOST": f"{host}:{rerank_ports[0]}" if rerank_ports else host,
            "TEI_RERANK_URLS": urls,
        }

    def apply_to_env(self) -> None:
        for k, v in self.derive_env().items():
            os.environ[k] = v


def _seed_from_env(key: str) -> Optional[str]:
    """First-boot seed: respect whatever the deployment manifest already
    set as env, so the DB starts in sync with the pod environment."""
    if key == "aiserver_host":
        return os.environ.get("QDRANT_HOST") or os.environ.get("TEI_HOST")
    if key == "qdrant_port":
        return os.environ.get("QDRANT_PORT")
    if key == "tei_embed_port":
        v = os.environ.get("TEI_PORTS")
        if v:
            return v.split(",")[0].strip()
        return None
    if key == "tei_rerank_ports":
        v = os.environ.get("TEI_RERANK_URLS")
        if v:
            ports: list[str] = []
            for u in v.split(","):
                u = u.strip().rstrip("/")
                # http://host:port/rerank → port
                if "://" in u:
                    u = u.split("://", 1)[1]
                u = u.split("/", 1)[0]
                if ":" in u:
                    ports.append(u.rsplit(":", 1)[1])
            if ports:
                return ",".join(ports)
        v = os.environ.get("TEI_RERANK_HOST")
        if v and ":" in v:
            return v.rsplit(":", 1)[1]
        return None
    return None
