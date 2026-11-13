"""The HTTP server: the three routes of `fh.fold/v1`, on the standard library.

Framework choice. No FastAPI, no uvicorn, no Flask. The contract is three
JSON routes with accept-then-poll semantics; the polling side needs a thread
per connection, which `ThreadingHTTPServer` provides, and the compute side
already runs on its own worker thread (`jobs.JobTable`). The standard
library does this without pain, and `AGENTS.md` values few dependencies:
this service adds zero Python dependencies of its own on top of foundry's.

Run it with:

    python -m foundry_service --host 127.0.0.1 --port 8080

Binding. The default is loopback, and the precedent is `src/fold/server.rs`
in this repo: a compute service is what you deploy on a GPU machine for a
remote harness to reach, so `--host` accepts a routable address, but this
service authenticates nobody. A non-loopback bind puts an unauthenticated
compute endpoint on that interface, and the deployment needs protection of
its own (a tunnel, a firewall, a reverse proxy). The `--host 0.0.0.0` flag
does not add authentication, it only widens the exposure.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from . import __version__
from .contract import (
    PROTOCOL,
    ROUTE_FOLD,
    ROUTE_HEALTH,
    ErrorBody,
    FoldRequest,
    Health,
    JobAccepted,
    major_version,
)
from .jobs import JobTable, Runner

SERVICE_NAME = "fh-foundry-service"
DEFAULT_MODELS = ("rf3", "rf3_latest", "rf3_preprint")


class FoundryHandler(BaseHTTPRequestHandler):
    """The whole protocol. Mirrors the handlers in `src/fold/server.rs`."""

    protocol_version = "HTTP/1.1"
    server_version = SERVICE_NAME
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = urlsplit(self.path).path
        if path == ROUTE_HEALTH:
            self._send_json(HTTPStatus.OK, self._health().to_wire())
            return
        job_id = _job_id_from(path)
        if job_id is not None:
            status = self.server.jobs.status(job_id)
            if status is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    ErrorBody("unknown_job", f"no job {job_id}").to_wire(),
                )
            else:
                self._send_json(HTTPStatus.OK, status.to_wire())
            return
        self._send_json(HTTPStatus.NOT_FOUND, ErrorBody("not_found", f"no route {path}").to_wire())

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        path = urlsplit(self.path).path
        if path != ROUTE_FOLD:
            self._send_json(
                HTTPStatus.NOT_FOUND, ErrorBody("not_found", f"no route {path}").to_wire()
            )
            return

        body = self._read_body()
        if body is None:
            return
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                ErrorBody("malformed_request", f"body is not valid JSON: {exc}").to_wire(),
            )
            return
        if not isinstance(payload, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                ErrorBody("malformed_request", "body must be a JSON object").to_wire(),
            )
            return

        try:
            request = FoldRequest.from_wire(payload)
        except (ValueError, TypeError, KeyError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, ErrorBody("invalid_job", str(exc)).to_wire())
            return

        if major_version(request.protocol) != major_version(PROTOCOL):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                ErrorBody(
                    "protocol_mismatch",
                    f"service speaks {PROTOCOL}, request declared {request.protocol}",
                ).to_wire(),
            )
            return

        job_id, queue_position = self.server.jobs.submit(request.job)
        print(
            f"[{SERVICE_NAME}] accepted {job_id} '{request.job.name}' "
            f"model={request.job.model} chains={len(request.job.chains)} "
            f"queue_position={queue_position}",
            file=sys.stderr,
            flush=True,
        )
        self._send_json(
            HTTPStatus.ACCEPTED,
            JobAccepted(job_id=job_id, queue_position=queue_position).to_wire(),
        )

    def _health(self) -> Health:
        return Health(
            protocol=PROTOCOL,
            service=SERVICE_NAME,
            version=__version__,
            capabilities=["fold"],
            models=list(self.server.models),
            gpus=[],
            queue_depth=self.server.jobs.queue_depth(),
        )

    def _read_body(self) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                ErrorBody("malformed_request", "request body is empty").to_wire(),
            )
            return None
        return self.rfile.read(length)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Status polling would flood stderr, and the handler logs nothing by
    # default; submissions log explicitly in do_POST and errors land here.
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def log_error(self, format: str, *args) -> None:  # noqa: A002
        print(f"[{SERVICE_NAME}] {format % args}", file=sys.stderr, flush=True)


class FoundryServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], jobs: JobTable, models: tuple[str, ...]) -> None:
        self.jobs = jobs
        self.models = models
        super().__init__(addr, FoundryHandler)


def _job_id_from(path: str) -> str | None:
    prefix = ROUTE_FOLD + "/"
    if not path.startswith(prefix):
        return None
    job_id = unquote(path[len(prefix) :])
    return job_id or None


def make_server(
    host: str, port: int, runner: Runner, models: tuple[str, ...] = DEFAULT_MODELS
) -> FoundryServer:
    """Bind `host:port`. Port 0 asks the OS for a free port."""
    return FoundryServer((host, port), JobTable(runner), models)


def serve_in_background(server: FoundryServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, name="fh-foundry-http", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fh-foundry-service",
        description="Reference fh.fold/v1 compute service running RosettaCommons/foundry RF3.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default 127.0.0.1). This service authenticates nobody; "
        "a non-loopback bind exposes an unauthenticated compute endpoint.",
    )
    parser.add_argument("--port", type=int, default=8080, help="port to bind (default 8080)")
    args = parser.parse_args(argv)

    # Imported here so `--help`, the tests and every non-GPU code path run
    # without foundry installed. rf3.py is the module that needs it.
    from .rf3 import rf3_runner  # noqa: PLC0415

    server = make_server(args.host, args.port, rf3_runner)
    host, port = server.server_address[0], server.server_address[1]
    print(
        f"[{SERVICE_NAME}] listening on http://{host}:{port} "
        f"(protocol {PROTOCOL}, no authentication; a non-loopback bind is exposed)",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
