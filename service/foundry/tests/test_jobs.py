"""The job table: accept now, compute later, record every failure."""

from __future__ import annotations

import threading

from foundry_service.contract import JobState
from foundry_service.jobs import JobTable
from tests.fakes import fold_result, job_spec

# Generous deadlock guard, per the AGENTS.md exception: the healthy case
# signals in milliseconds, and a timeout names the worker that never
# finished instead of hanging the suite.
WAIT_TIMEOUT = 10.0


class Gate:
    """Blocks the runner until the test releases it. No sleeps involved."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, job):
        self.entered.set()
        if not self.release.wait(WAIT_TIMEOUT):
            raise AssertionError(f"the test never released the gate for job {job.name}")
        return fold_result()


def test_a_submitted_job_gets_a_sequential_id_and_reaches_succeeded():
    table = JobTable(lambda job: fold_result())
    job_id, position = table.submit(job_spec(name="one"))
    assert job_id == "job-000001"
    assert position == 0
    assert table.wait(job_id, WAIT_TIMEOUT), f"{job_id} never reached a terminal state"
    status = table.status(job_id)
    assert status.state == JobState.SUCCEEDED
    assert status.message is None
    # The table owns the job id and stamps it onto the runner's result.
    assert status.result.job_id == job_id
    assert table.queue_depth() == 0


def test_a_second_job_queues_behind_a_running_one():
    gate = Gate()
    table = JobTable(gate)
    first_id, first_position = table.submit(job_spec(name="first"))
    assert gate.entered.wait(WAIT_TIMEOUT), "the worker never picked up the first job"
    assert first_position == 0
    assert table.status(first_id).state == JobState.RUNNING

    second_id, second_position = table.submit(job_spec(name="second"))
    assert second_position == 1
    assert table.status(second_id).state == JobState.QUEUED
    assert table.queue_depth() == 2

    gate.release.set()
    assert table.wait(second_id, WAIT_TIMEOUT), "the second job never ran after the first"
    assert table.status(first_id).state == JobState.SUCCEEDED
    assert table.status(second_id).state == JobState.SUCCEEDED
    assert table.queue_depth() == 0


def test_a_runner_failure_is_recorded_with_its_reason():
    def explode(job):
        raise RuntimeError("rf3 checkpoint not found")

    table = JobTable(explode)
    job_id, _ = table.submit(job_spec())
    assert table.wait(job_id, WAIT_TIMEOUT), f"{job_id} never reached a terminal state"
    status = table.status(job_id)
    assert status.state == JobState.FAILED
    assert status.result is None
    assert "rf3 checkpoint not found" in status.message
    assert table.queue_depth() == 0


def test_an_unknown_job_id_has_no_status_and_no_wait():
    table = JobTable(lambda job: fold_result())
    assert table.status("job-999999") is None
    assert table.wait("job-999999", WAIT_TIMEOUT) is False


def test_status_returns_a_copy_so_callers_cannot_mutate_the_table():
    table = JobTable(lambda job: fold_result())
    job_id, _ = table.submit(job_spec())
    assert table.wait(job_id, WAIT_TIMEOUT)
    snapshot = table.status(job_id)
    snapshot.state = JobState.QUEUED
    assert table.status(job_id).state == JobState.SUCCEEDED
