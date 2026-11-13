"""The three routes over a real socket, against hand-written fake runners.

No test here runs RF3 or needs a GPU. Every server binds 127.0.0.1 port 0.
Waiting uses the job table's terminal-state event with a generous deadlock
guard; nothing sleeps and nothing depends on a time window.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from foundry_service.contract import PROTOCOL
from foundry_service.rf3 import EarlyStopped
from foundry_service.server import make_server, serve_in_background
from tests.fakes import fold_result, job_spec

WAIT_TIMEOUT = 10.0
SOCKET_TIMEOUT = 10.0  # deadlock guard, not a timing assertion


class Service:
    def __init__(self, runner) -> None:
        self.server = make_server("127.0.0.1", 0, runner)
        self.thread = serve_in_background(self.server)
        host, port = self.server.server_address[0], self.server.server_address[1]
        self.base = f"http://{host}:{port}"

    @property
    def jobs(self):
        return self.server.jobs

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(WAIT_TIMEOUT)


def get(base: str, path: str) -> tuple[int, dict]:
    request = urllib.request.Request(base + path, method="GET")
    return _send(request)


def post(base: str, path: str, payload: dict | str) -> tuple[int, dict]:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    request = urllib.request.Request(
        base + path,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send(request)


def _send(request) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def fold_request(protocol: str = PROTOCOL) -> dict:
    return {"protocol": protocol, "job": job_spec().to_wire()}


@pytest.fixture()
def service():
    running = Service(lambda job: fold_result())
    yield running
    running.close()


def test_health_reports_the_exact_protocol_and_capability_set(service):
    status, health = get(service.base, "/v1/health")
    assert status == 200
    # The Rust client refuses a major-version mismatch on this string.
    assert health["protocol"] == "fh.fold/v1"
    # `gpus` is omitted while empty: the Rust field carries
    # skip_serializing_if = "Vec::is_empty", and this service reports no GPU
    # inventory. `models` is non-empty, so it is present.
    assert set(health) == {
        "protocol",
        "service",
        "version",
        "capabilities",
        "models",
        "queue_depth",
    }
    assert health["capabilities"] == ["fold"]
    assert "rf3" in health["models"]
    assert health["queue_depth"] == 0


def test_a_fold_round_trips_through_the_queue_with_every_field_the_harness_reads(service):
    status, accepted = post(service.base, "/v1/fold", fold_request())
    assert status == 202
    assert set(accepted) == {"job_id", "queue_position"}
    job_id = accepted["job_id"]

    assert service.jobs.wait(job_id, WAIT_TIMEOUT), f"{job_id} never reached a terminal state"
    status, body = get(service.base, f"/v1/fold/{job_id}")
    assert status == 200
    # Optional fields that are None are omitted, per the serde rules.
    assert set(body) == {"job_id", "state", "result"}
    assert body["state"] == "succeeded"

    result = body["result"]
    assert set(result) == {"job_id", "model", "confidence", "structure", "elapsed_ms"}
    assert result["job_id"] == job_id

    confidence = result["confidence"]
    assert set(confidence) == {
        "iptm",
        "ptm",
        "plddt_mean",
        "interface_plddt",
        "pae_interface",
        "interface_contacts",
        "clashes",
        "hydrophobic_patch",
        "net_charge",
        "plddt",
        "atom_confidence",
    }

    structure = result["structure"]
    assert structure["format"] == "ca_trace"
    assert "inline" not in structure
    assert "artifact_url" not in structure
    assert set(structure["ca_trace"][0]) == {
        "chain_index",
        "index",
        "residue",
        "xyz",
        "plddt",
        "designed",
    }


def test_a_second_job_queues_behind_a_running_one_and_health_shows_the_depth():
    entered = threading.Event()
    release = threading.Event()

    def gated_runner(job):
        entered.set()
        if not release.wait(WAIT_TIMEOUT):
            raise AssertionError(f"the gate was never released for {job.name}")
        return fold_result()

    running = Service(gated_runner)
    try:
        _, first = post(running.base, "/v1/fold", fold_request())
        assert entered.wait(WAIT_TIMEOUT), "the worker never picked up the first job"

        _, second = post(running.base, "/v1/fold", fold_request())
        assert second["queue_position"] == 1
        _, body = get(running.base, f"/v1/fold/{second['job_id']}")
        assert body["state"] == "queued"

        _, health = get(running.base, "/v1/health")
        assert health["queue_depth"] == 2

        release.set()
        assert running.jobs.wait(second["job_id"], WAIT_TIMEOUT), "the second job never finished"
        _, body = get(running.base, f"/v1/fold/{second['job_id']}")
        assert body["state"] == "succeeded"
    finally:
        release.set()
        running.close()


def test_a_failed_job_records_its_reason_and_carries_no_result():
    def early_stopped_runner(job):
        raise EarlyStopped(
            "RF3 early-stopped on design-7: mean pLDDT 0.42 is below the "
            "early_stopping_plddt_threshold and no structure was written"
        )

    running = Service(early_stopped_runner)
    try:
        _, accepted = post(running.base, "/v1/fold", fold_request())
        job_id = accepted["job_id"]
        assert running.jobs.wait(job_id, WAIT_TIMEOUT), f"{job_id} never reached a terminal state"
        status, body = get(running.base, f"/v1/fold/{job_id}")
        assert status == 200
        assert body["state"] == "failed"
        assert "early-stopped" in body["message"]
        assert "no structure" in body["message"]
        assert "result" not in body
    finally:
        running.close()


def test_an_unknown_job_is_a_404_with_a_reason(service):
    status, body = get(service.base, "/v1/fold/job-999999")
    assert status == 404
    assert body == {"error": "unknown_job", "detail": "no job job-999999"}


def test_a_major_version_mismatch_is_refused_like_the_rust_server_refuses_it(service):
    status, body = post(service.base, "/v1/fold", fold_request(protocol="fh.fold/v2"))
    assert status == 400
    assert body["error"] == "protocol_mismatch"
    assert "fh.fold/v1" in body["detail"]
    assert "fh.fold/v2" in body["detail"]


def test_malformed_json_is_a_400_that_names_the_problem(service):
    status, body = post(service.base, "/v1/fold", "{not json")
    assert status == 400
    assert body["error"] == "malformed_request"


def test_a_job_missing_a_required_field_is_a_400_that_names_the_field(service):
    request = fold_request()
    del request["job"]["diffusion_steps"]
    status, body = post(service.base, "/v1/fold", request)
    assert status == 400
    assert body["error"] == "invalid_job"
    assert "diffusion_steps" in body["detail"]


def test_an_unknown_route_is_a_404(service):
    status, body = get(service.base, "/v1/nope")
    assert status == 404
    assert body["error"] == "not_found"
