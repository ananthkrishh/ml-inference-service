"""With several uvicorn workers, /metrics must add up ALL workers' numbers.

Starts a real uvicorn with 2 worker processes (like the Docker image) and checks that
the request count Prometheus would see equals the requests actually sent.
"""

import os
import socket
import subprocess
import sys
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _predict_count(text: str) -> float:
    for family in text_string_to_metric_families(text):
        if family.name == "http_request_duration_seconds":
            return sum(
                s.value
                for s in family.samples
                if s.name.endswith("_count") and s.labels.get("route") == "/predict"
            )
    return 0.0


def test_two_workers_report_combined_metrics(model_path, tmp_path, customer):
    port = _free_port()
    env = {
        **os.environ,
        "MODEL_PATH": str(model_path),
        "PROMETHEUS_MULTIPROC_DIR": str(tmp_path),
        "OLLAMA_BASE_URL": "http://127.0.0.1:9",
    }
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--workers", "2"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=1).json()["model_loaded"]:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        time.sleep(1.5)  # let the second worker finish starting too

        # New connection per request, so the OS spreads them over both workers.
        for _ in range(40):
            assert httpx.post(f"{base}/predict", json=customer).status_code == 200

        files = {p.name.rsplit("_", 1)[-1] for p in tmp_path.glob("*.db")}
        assert len(files) >= 2, f"expected metric files from 2 worker PIDs, got {files}"
        assert _predict_count(httpx.get(f"{base}/metrics").text) == 40
    finally:
        proc.terminate()
        proc.wait(timeout=10)
