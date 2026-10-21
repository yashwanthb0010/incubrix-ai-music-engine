"""
Database persistence for IncuBrix AI Background Music Engine.
Thread-safe SQLite storage for Job state, Idempotency tracking, and Manifests.
"""

from __future__ import annotations
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from incubrix.schemas import JobStatus, JobStatusResponse, JobAssets


class Database:
    def __init__(self, db_path: str):
        # Handle sqlite:/// URI prefix
        if db_path.startswith("sqlite:///"):
            clean_path = db_path.replace("sqlite:///", "")
        else:
            clean_path = db_path

        self.db_path = Path(clean_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()

            # Jobs table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                request_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                error_json TEXT,
                assets_json TEXT NOT NULL,
                manifest_json TEXT
            );
            """)

            # Idempotency table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                idempotency_key TEXT PRIMARY KEY,
                request_hash TEXT NOT NULL,
                job_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            );
            """)

            conn.commit()
            conn.close()

    def save_idempotency_key(self, key: str, request_hash: str, job_id: str) -> None:
        self._init_tables()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO idempotency_keys (idempotency_key, request_hash, job_id, created_at) VALUES (?, ?, ?, ?)",
                (key, request_hash, job_id, now),
            )
            conn.commit()
            conn.close()

    def get_idempotency_record(self, key: str) -> Optional[Tuple[str, str]]:
        """Returns (request_hash, job_id) if key exists."""
        self._init_tables()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT request_hash, job_id FROM idempotency_keys WHERE idempotency_key = ?", (key,))
            row = cur.fetchone()
            conn.close()
            if row:
                return row["request_hash"], row["job_id"]
            return None

    def create_job(self, job_id: str, request_data: Dict[str, Any], initial_warnings: list[str]) -> None:
        self._init_tables()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO jobs (job_id, status, progress, created_at, updated_at, request_json, warnings_json, assets_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    JobStatus.QUEUED.value,
                    0,
                    now,
                    now,
                    json.dumps(request_data),
                    json.dumps(initial_warnings),
                    json.dumps({}),
                ),
            )
            conn.commit()
            conn.close()

    def update_job_progress(
        self,
        job_id: str,
        status: JobStatus,
        progress: int,
        warnings: Optional[list[str]] = None,
        error: Optional[Dict[str, Any]] = None,
        assets: Optional[Dict[str, Any]] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._init_tables()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()

            # Read current values if not provided
            cur.execute("SELECT warnings_json, assets_json, manifest_json FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return

            current_warnings = json.loads(row["warnings_json"]) if row["warnings_json"] else []
            if warnings is not None:
                # Merge unique warnings
                for w in warnings:
                    if w not in current_warnings:
                        current_warnings.append(w)

            current_assets = json.loads(row["assets_json"]) if row["assets_json"] else {}
            if assets is not None:
                current_assets.update(assets)

            manifest_str = json.dumps(manifest) if manifest is not None else row["manifest_json"]
            error_str = json.dumps(error) if error is not None else None
            completed_at = now if status in (JobStatus.SUCCEEDED, JobStatus.FAILED) else None

            cur.execute(
                """
                UPDATE jobs
                SET status = ?, progress = ?, updated_at = ?, completed_at = COALESCE(?, completed_at),
                    warnings_json = ?, error_json = ?, assets_json = ?, manifest_json = ?
                WHERE job_id = ?
                """,
                (
                    status.value,
                    progress,
                    now,
                    completed_at,
                    json.dumps(current_warnings),
                    error_str,
                    json.dumps(current_assets),
                    manifest_str,
                    job_id,
                ),
            )
            conn.commit()
            conn.close()

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        self._init_tables()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            conn.close()

            if not row:
                return None

            error_obj = json.loads(row["error_json"]) if row["error_json"] else None
            assets_dict = json.loads(row["assets_json"]) if row["assets_json"] else {}
            warnings_list = json.loads(row["warnings_json"]) if row["warnings_json"] else []

            return JobStatusResponse(
                job_id=row["job_id"],
                status=JobStatus(row["status"]),
                progress=row["progress"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                warnings=warnings_list,
                error=error_obj,
                assets=JobAssets(**assets_dict),
            )

    def get_job_manifest(self, job_id: str) -> Optional[Dict[str, Any]]:
        self._init_tables()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT manifest_json FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            conn.close()
            if row and row["manifest_json"]:
                return json.loads(row["manifest_json"])
            return None

    def get_job_request(self, job_id: str) -> Optional[Dict[str, Any]]:
        self._init_tables()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT request_json FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            conn.close()
            if row and row["request_json"]:
                return json.loads(row["request_json"])
            return None


_db_singleton: Optional[Database] = None


def get_db(db_url: Optional[str] = None) -> Database:
    global _db_singleton
    if _db_singleton is None:
        from incubrix.config import get_settings
        url = db_url or get_settings().database_url
        _db_singleton = Database(url)
    return _db_singleton
