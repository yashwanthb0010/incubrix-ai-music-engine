"""
Asynchronous Background Worker for IncuBrix AI Background Music Engine.
Manages concurrency and ensures HTTP requests return immediately with HTTP 202.
"""

from __future__ import annotations
import asyncio
from typing import Dict, Any, Optional
from incubrix.config import Settings, get_settings
from incubrix.queue.pipeline import execute_job_pipeline


class AsyncJobQueue:
    def __init__(self, concurrency: int = 2):
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._running_jobs = set()
        self._total_processed = 0

    async def enqueue_job(self, job_id: str, settings: Optional[Settings] = None) -> None:
        """Enqueue job to run asynchronously in the background."""
        asyncio.create_task(self._process_worker(job_id, settings))

    async def _process_worker(self, job_id: str, settings: Optional[Settings] = None) -> None:
        async with self._semaphore:
            self._running_jobs.add(job_id)
            try:
                await execute_job_pipeline(job_id, settings)
            finally:
                self._running_jobs.discard(job_id)
                self._total_processed += 1

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_jobs": len(self._running_jobs),
            "concurrency_limit": self.concurrency,
            "total_processed": self._total_processed,
            "is_ready": True,
        }


_queue_singleton: Optional[AsyncJobQueue] = None


def get_queue(settings: Optional[Settings] = None) -> AsyncJobQueue:
    global _queue_singleton
    if _queue_singleton is None:
        cfg = settings or get_settings()
        _queue_singleton = AsyncJobQueue(concurrency=cfg.queue_concurrency)
    return _queue_singleton
