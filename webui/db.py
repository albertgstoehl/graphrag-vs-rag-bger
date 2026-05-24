"""SQLite persistence for the kg-rag-control UI.

Schema: one `runs` table tracking each pipeline invocation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at     TEXT NOT NULL,
  finished_at    TEXT,
  status         TEXT NOT NULL,
  exit_code      INTEGER,
  skip_sample    INTEGER DEFAULT 0,
  skip_retrieval INTEGER DEFAULT 0,
  skip_metrics   INTEGER DEFAULT 0,
  systems        TEXT,
  rankings       TEXT,
  k_values       TEXT,
  log_path       TEXT,
  duration_s     INTEGER,
  error_message  TEXT,
  query_limit    INTEGER DEFAULT 0,
  resume         INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
"""

# Idempotent migration for existing DBs that pre-date the column.
_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN query_limit INTEGER DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN resume INTEGER DEFAULT 1",
]


class RunStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            # Apply additive migrations; ignore "duplicate column" errors
            # on already-migrated DBs.
            for stmt in _MIGRATIONS:
                try:
                    c.execute(stmt)
                except sqlite3.OperationalError:
                    pass

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def create_run(
        self,
        *,
        skip_sample: bool,
        skip_retrieval: bool,
        skip_metrics: bool,
        systems: list[str],
        rankings: list[str],
        k_values: list[int],
        log_path: str,
        query_limit: int = 0,
        resume: bool = True,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO runs (
                    started_at, status,
                    skip_sample, skip_retrieval, skip_metrics,
                    systems, rankings, k_values, log_path, query_limit,
                    resume
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._now(),
                    int(skip_sample), int(skip_retrieval), int(skip_metrics),
                    ",".join(systems),
                    ",".join(rankings),
                    ",".join(str(k) for k in k_values),
                    log_path,
                    int(query_limit or 0),
                    int(bool(resume)),
                ),
            )
            return cur.lastrowid

    def mark_running(self, run_id: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status='running', started_at=? WHERE id=?",
                (self._now(), run_id),
            )

    def mark_finished(
        self, run_id: int, *,
        status: str,
        exit_code: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with self._conn() as c:
            row = c.execute(
                "SELECT started_at FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            started = datetime.fromisoformat(row["started_at"])
            dur = int((datetime.now(timezone.utc) - started).total_seconds())
            c.execute(
                """
                UPDATE runs SET
                    status=?, finished_at=?, exit_code=?,
                    duration_s=?, error_message=?
                WHERE id=?
                """,
                (status, self._now(), exit_code, dur, error_message, run_id),
            )

    def get(self, run_id: int) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_recent(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def latest_running(self) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM runs WHERE status IN ('queued','running') "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def interrupt_all_running(self, note: str = "interrupted by pod restart") -> int:
        """Called on startup to clean up orphaned runs from crashed pods."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE runs SET status='failed', error_message=?, "
                "finished_at=? WHERE status IN ('queued','running')",
                (note, self._now()),
            )
            return cur.rowcount
