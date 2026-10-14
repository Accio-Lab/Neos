#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx


HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

INFERENCE_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/responses"}


class InferenceActivity:
    pass

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.last_activity = clock()
        self.inflight = 0
        self.received = 0
        self.closing = False

    def begin(self) -> bool:
        with self.lock:
            if self.closing:
                return False
            self.inflight += 1
            self.received += 1
            self.last_activity = self.clock()
            return True

    def end(self) -> None:
        with self.lock:
            self.inflight -= 1
            self.last_activity = self.clock()

    def claim_idle_shutdown(self, timeout: float) -> bool:
        with self.lock:
            if timeout <= 0 or self.closing or self.inflight or self.clock() - self.last_activity < timeout:
                return False
            self.closing = True
            return True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"received": self.received, "inflight": self.inflight,
                    "idle_seconds": round(self.clock() - self.last_activity, 3),
                    "shutdown_requested": self.closing}


def idle_monitor(server, activity, timeout, notice_path, stopped):
    while not stopped.wait(min(5.0, max(0.05, timeout / 4))):
        if not activity.claim_idle_shutdown(timeout):
            continue
        notice = {"reason": "inference_idle_timeout", "proxy_pid": os.getpid(),
                  "at_unix": time.time(), "timeout_seconds": timeout,
                  "activity": activity.snapshot()}
        if notice_path is not None:
            try:
                notice_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = notice_path.with_name(notice_path.name + f".tmp-{os.getpid()}")
                temporary.write_text(json.dumps(notice) + "\n")
                os.replace(temporary, notice_path)
            except OSError as error:
                print(f"proxy_idle_notice_write_failed={error}; continuing shutdown", flush=True)
        print("proxy_idle_shutdown=" + json.dumps(notice), flush=True)
        server.shutdown()
        return


class BackendPool:
    def __init__(self, endpoints: list[str], timeout: float) -> None:
        self.endpoints = [value.rstrip("/") for value in endpoints]
        self.clients = [httpx.Client(timeout=timeout) for _ in self.endpoints]
        self.inflight = [0] * len(self.endpoints)
        self.completed = [0] * len(self.endpoints)
        self.errors = [0] * len(self.endpoints)
        self.total_ms = [0.0] * len(self.endpoints)
        self.cursor = 0
        self.lock = threading.Lock()
        self.activity = InferenceActivity()

    def acquire(self, excluded: set[int]) -> int:
        with self.lock:
            candidates = [i for i in range(len(self.endpoints)) if i not in excluded]
            if not candidates:
                candidates = list(range(len(self.endpoints)))
            minimum = min(self.inflight[i] for i in candidates)
            tied = [i for i in candidates if self.inflight[i] == minimum]
            selected = tied[self.cursor % len(tied)]
            self.cursor += 1
            self.inflight[selected] += 1
            return selected

    def release(self, index: int, *, ok: bool, elapsed_ms: float) -> None:
        with self.lock:
            self.inflight[index] -= 1
            self.completed[index] += int(ok)
            self.errors[index] += int(not ok)
            self.total_ms[index] += elapsed_ms

    def forward(
        self,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> tuple[int, dict[str, str], bytes, int]:
        excluded: set[int] = set()
        failures: list[str] = []
        for _ in range(min(3, len(self.endpoints))):
            index = self.acquire(excluded)
            excluded.add(index)
            started = time.perf_counter()
            ok = False
            try:
                response = self.clients[index].request(
                    method,
                    self.endpoints[index] + path,
                    content=body,
                    headers=headers,
                )
                payload = response.content
                if response.status_code >= 500:
                    failures.append(f"backend={index} status={response.status_code}")
                    continue
                ok = True
                output_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() not in HOP_HEADERS
                }
                return response.status_code, output_headers, payload, index
            except Exception as exc:  
                failures.append(f"backend={index} {type(exc).__name__}: {exc}")
            finally:
                self.release(
                    index,
                    ok=ok,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
        payload = json.dumps(
            {
                "error": {
                    "message": "; ".join(failures)[-4000:],
                    "type": "local_mei_backends_unavailable",
                }
            }
        ).encode()
        return 502, {"content-type": "application/json"}, payload, -1

    def health(self) -> dict[str, Any]:
        backend_health = []
        for index, (endpoint, client) in enumerate(zip(self.endpoints, self.clients)):
            healthy = False
            error = None
            try:
                response = client.get(endpoint.removesuffix("/v1") + "/health", timeout=2)
                healthy = response.status_code == 200
                if not healthy:
                    error = f"status={response.status_code}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            with self.lock:
                completed = self.completed[index]
                backend_health.append(
                    {
                        "index": index,
                        "endpoint": endpoint,
                        "healthy": healthy,
                        "inflight": self.inflight[index],
                        "completed": completed,
                        "errors": self.errors[index],
                        "mean_ms": round(self.total_ms[index] / completed, 3)
                        if completed
                        else None,
                        "error": error,
                    }
                )
        return {
            "status": "ok" if any(row["healthy"] for row in backend_health) else "error",
            "healthy_backends": sum(row["healthy"] for row in backend_health),
            "total_backends": len(backend_health),
            "backends": backend_health,
            "inference_activity": self.activity.snapshot(),
        }


def handler_class(pool: BackendPool, max_body_bytes: int):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, headers: dict[str, str], payload: bytes) -> None:
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(payload)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True

        def do_GET(self) -> None:  
            if self.path.rstrip("/") in {"", "/health"}:
                health = pool.health()
                status = 200 if health["status"] == "ok" else 503
                self._send(
                    status,
                    {"content-type": "application/json"},
                    json.dumps(health).encode(),
                )
                return
            status, headers, payload, _ = pool.forward(
                "GET", self.path, b"", {}
            )
            self._send(status, headers, payload)

        def do_POST(self) -> None:  
            try:
                size = int(self.headers.get("content-length") or 0)
            except ValueError:
                size = -1
            if size < 0 or size > max_body_bytes:
                self._send(
                    413,
                    {"content-type": "application/json"},
                    b'{"error":{"message":"request body too large"}}',
                )
                return
            inference = self.path.split("?", 1)[0].rstrip("/") in INFERENCE_PATHS
            if inference and not pool.activity.begin():
                self._send(503, {"content-type": "application/json"},
                           b'{"error":{"message":"mei shutting down after idle timeout"}}')
                return
            try:
                body = self.rfile.read(size)
                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in HOP_HEADERS
                }
                status, response_headers, payload, backend = pool.forward(
                    "POST", self.path, body, headers
                )
                response_headers["x-local-mei-backend"] = str(backend)
                self._send(status, response_headers, payload)
            finally:
                if inference:
                    pool.activity.end()

        def log_message(self, fmt: str, *args: Any) -> None:
            print(
                f"proxy_request client={self.client_address[0]} "
                + (fmt % args),
                flush=True,
            )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=48080)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-body-mib", type=int, default=128)
    parser.add_argument("--idle-timeout-seconds", type=float, default=0,
                        help="Opt-in shutdown after no real inference requests; health probes do not reset it.")
    parser.add_argument("--idle-notice-file", type=Path)
    args = parser.parse_args()
    if args.idle_timeout_seconds < 0:
        parser.error("idle timeout must be nonnegative")
    pool = BackendPool(args.backend, args.timeout)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_class(pool, args.max_body_mib * 1024 * 1024),
    )
    server.daemon_threads = True
    stopped = threading.Event()
    if args.idle_timeout_seconds:
        threading.Thread(target=idle_monitor, args=(server, pool.activity,
            args.idle_timeout_seconds, args.idle_notice_file, stopped), daemon=True).start()
    print(
        f"proxy_ready host={args.host} port={args.port} backends={len(args.backend)}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stopped.set()
        server.server_close()
        for client in pool.clients:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
