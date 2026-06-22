"""
Batch Optimization
Process multiple prompts in parallel with progress tracking.
"""

import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


class BatchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchJob:
    job_id: str
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    model_adapter: Optional[str] = None
    task_type: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'job_id': self.job_id,
            'prompt': self.prompt[:100] + '...' if len(self.prompt) > 100 else self.prompt,
            'status': self.status.value,
            'latency_ms': round(self.latency_ms, 2),
            'started_at': self.started_at,
            'completed_at': self.completed_at,
        }
        if self.error:
            d['error'] = self.error
        if self.result:
            d['result'] = {
                'optimized_prompt': getattr(self.result, 'optimized_prompt', None),
                'compression_ratio': getattr(self.result, 'compression_ratio', None),
                'tokens_saved': getattr(self.result, 'tokens_saved', None),
            }
        return d


@dataclass
class BatchRequest:
    batch_id: str
    jobs: List[BatchJob] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    max_concurrency: int = 4
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    organization_id: Optional[str] = None
    user_id: Optional[str] = None

    @property
    def total_jobs(self) -> int:
        return len(self.jobs)

    @property
    def completed_jobs(self) -> int:
        return sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)

    @property
    def failed_jobs(self) -> int:
        return sum(1 for j in self.jobs if j.status == JobStatus.FAILED)

    @property
    def progress(self) -> float:
        if not self.jobs:
            return 0.0
        return (self.completed_jobs + self.failed_jobs) / len(self.jobs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_id': self.batch_id,
            'status': self.status.value,
            'total_jobs': self.total_jobs,
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'progress': round(self.progress, 4),
            'max_concurrency': self.max_concurrency,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'jobs': [j.to_dict() for j in self.jobs],
        }


class BatchProcessor:
    """
    Batch optimization processor with concurrency control and progress tracking.
    """

    def __init__(self, max_concurrency: int = 4):
        self.max_concurrency = max_concurrency
        self._batches: Dict[str, BatchRequest] = {}
        self._lock = threading.Lock()

    def create_batch(
        self,
        prompts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
        model_adapter: Optional[str] = None,
        task_type: Optional[str] = None,
        max_concurrency: int = 4,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> BatchRequest:
        batch_id = f"batch-{uuid.uuid4().hex[:12]}"
        now = time.time()

        jobs = []
        for i, prompt in enumerate(prompts):
            ctx = contexts[i] if contexts and i < len(contexts) else {}
            jobs.append(BatchJob(
                job_id=f"job-{uuid.uuid4().hex[:8]}",
                prompt=prompt, context=ctx,
                model_adapter=model_adapter, task_type=task_type,
            ))

        batch = BatchRequest(
            batch_id=batch_id, jobs=jobs,
            max_concurrency=max_concurrency,
            created_at=now,
            organization_id=organization_id, user_id=user_id,
        )
        with self._lock:
            self._batches[batch_id] = batch
        return batch

    def execute_batch(
        self,
        batch_id: str,
        optimize_fn: Callable,
    ) -> Optional[BatchRequest]:
        batch = self._batches.get(batch_id)
        if not batch:
            return None

        batch.status = BatchStatus.RUNNING
        batch.started_at = time.time()

        def run_job(job: BatchJob):
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            try:
                result = optimize_fn(
                    prompt=job.prompt,
                    context=job.context,
                    model_adapter=job.model_adapter,
                )
                job.result = result
                job.status = JobStatus.COMPLETED
            except Exception as e:
                job.error = str(e)
                job.status = JobStatus.FAILED
            finally:
                job.completed_at = time.time()
                job.latency_ms = job.duration_ms

        with ThreadPoolExecutor(max_workers=batch.max_concurrency) as executor:
            futures = {executor.submit(run_job, job): job for job in batch.jobs}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        batch.status = BatchStatus.COMPLETED
        batch.completed_at = time.time()
        return batch

    def get_batch(self, batch_id: str) -> Optional[BatchRequest]:
        return self._batches.get(batch_id)

    def cancel_batch(self, batch_id: str) -> Optional[BatchRequest]:
        batch = self._batches.get(batch_id)
        if batch and batch.status in (BatchStatus.RUNNING, BatchStatus.PENDING):
            batch.status = BatchStatus.CANCELLED
            batch.completed_at = time.time()
        return batch

    def list_batches(
        self, organization_id: Optional[str] = None, user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        batches = list(self._batches.values())
        if organization_id:
            batches = [b for b in batches if b.organization_id == organization_id]
        if user_id:
            batches = [b for b in batches if b.user_id == user_id]
        return [b.to_dict() for b in batches]

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._batches)
        completed = sum(1 for b in self._batches.values() if b.status == BatchStatus.COMPLETED)
        return {
            'total_batches': total,
            'completed': completed,
            'total_jobs': sum(b.total_jobs for b in self._batches.values()),
            'completed_jobs': sum(b.completed_jobs for b in self._batches.values()),
        }
