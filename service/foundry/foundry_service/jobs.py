"""The job table and the worker thread: accept now, compute later.

This is the reason the contract is shaped the way it is. A fold outlives the
HTTP request, a client that dies mid-run reattaches by job id, and `health`
reports the queue. The Rust reference in `src/fold/server.rs` does the same
thing with a tokio task per job; here one worker thread drains a FIFO queue,
because one GPU runs one fold at a time anyway and serialising the work is
what keeps that true.

Every rejected job gets a recorded reason in `JobStatus.message`, mirroring
the harness rule that nothing is dropped silently.
"""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable

from .contract import FoldResult, JobSpec, JobState, JobStatus

Runner = Callable[[JobSpec], FoldResult]


class JobTable:
    """Job ids, states and results, plus the worker that fills them in."""

    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        self._jobs: dict[str, JobStatus] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, JobSpec]] = queue.Queue()
        self._counter = itertools.count(1)
        self._worker = threading.Thread(target=self._work, name="fh-foundry-jobs", daemon=True)
        self._worker.start()

    def submit(self, job: JobSpec) -> tuple[str, int]:
        """Record a queued job and hand it to the worker.

        Returns the job id and the queue position at submit time, counted the
        way `src/fold/server.rs` counts it: the number of non-terminal jobs
        already in the table.
        """
        with self._lock:
            job_id = f"job-{next(self._counter):06}"
            queue_position = self._queue_depth_locked()
            self._jobs[job_id] = JobStatus(job_id=job_id, state=JobState.QUEUED)
            self._events[job_id] = threading.Event()
        self._queue.put((job_id, job))
        return job_id, queue_position

    def status(self, job_id: str) -> JobStatus | None:
        with self._lock:
            found = self._jobs.get(job_id)
            return None if found is None else _copy(found)

    def queue_depth(self) -> int:
        with self._lock:
            return self._queue_depth_locked()

    def wait(self, job_id: str, timeout: float) -> bool:
        """Block until the job reaches a terminal state, or the timeout ends.

        The timeout is a deadlock guard, not a timing assertion: the healthy
        case signals in milliseconds, and a False return names a worker that
        never finished. Unknown job ids return False immediately.
        """
        with self._lock:
            event = self._events.get(job_id)
        if event is None:
            return False
        return event.wait(timeout)

    def _queue_depth_locked(self) -> int:
        return sum(1 for job in self._jobs.values() if not job.state.is_terminal())

    def _set_state(self, job_id: str, state: JobState) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.state = state

    def _finish(
        self,
        job_id: str,
        state: JobState,
        result: FoldResult | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.state = state
                job.result = result
                job.message = message
            event = self._events.get(job_id)
        if event is not None:
            event.set()

    def _work(self) -> None:
        while True:
            job_id, job = self._queue.get()
            self._set_state(job_id, JobState.RUNNING)
            try:
                result = self._runner(job)
            except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                reason = str(exc) or repr(exc)
                self._finish(
                    job_id,
                    JobState.FAILED,
                    message=f"{type(exc).__name__}: {reason}",
                )
                continue
            # The runner cannot know its job id; the table owns it, the same
            # way the Rust server overwrites result.job_id.
            result.job_id = job_id
            self._finish(job_id, JobState.SUCCEEDED, result=result)


def _copy(job: JobStatus) -> JobStatus:
    return JobStatus(
        job_id=job.job_id,
        state=job.state,
        message=job.message,
        result=job.result,
    )
