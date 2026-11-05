"""Prometheus metrics: /metrics output, request timing labels, model timings, errors.

Metrics live in one global registry shared by all tests, so each test reads a value
BEFORE and AFTER its request and checks the difference.
"""

import httpx
from prometheus_client import REGISTRY

from tests.conftest import OLLAMA_URL


def value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_serves_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    for name in (
        "http_request_duration_seconds",
        "http_requests_in_flight",
        "model_inference_seconds",
        "llm_tokens_per_second",
        "inference_errors_total",
    ):
        assert f"# TYPE {name}" in r.text


def test_predict_is_timed_by_route_and_status(client, customer):
    labels = {"method": "POST", "route": "/predict", "status": "200"}
    before = value("http_request_duration_seconds_count", **labels)
    model_before = value("model_inference_seconds_count", model="sklearn")

    assert client.post("/predict", json=customer).status_code == 200

    assert value("http_request_duration_seconds_count", **labels) == before + 1
    assert value("model_inference_seconds_count", model="sklearn") == model_before + 1


def test_error_status_is_recorded(client):
    labels = {"method": "POST", "route": "/predict", "status": "422"}
    before = value("http_request_duration_seconds_count", **labels)
    client.post("/predict", json={"tenure": -1})
    assert value("http_request_duration_seconds_count", **labels) == before + 1


def test_unknown_paths_share_one_label(client):
    """Bots probing random URLs must not create a new time series each."""
    labels = {"method": "GET", "route": "unmatched", "status": "404"}
    before = value("http_request_duration_seconds_count", **labels)
    client.get("/wp-login.php")
    client.get("/.env")
    assert value("http_request_duration_seconds_count", **labels) == before + 2
    assert "wp-login" not in client.get("/metrics").text


def test_metrics_scrapes_are_not_timed(client):
    client.get("/metrics")
    assert 'route="/metrics"' not in client.get("/metrics").text


def test_in_flight_is_back_to_zero_after_requests(client, customer):
    client.post("/predict", json=customer)
    assert value("http_requests_in_flight") == 0


def test_llm_success_records_timing_tokens_and_speed(client, respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3.2:3b",
                "response": "hi",
                "prompt_eval_count": 5,
                "eval_count": 20,
                "eval_duration": 500_000_000,  # 20 tokens in 0.5 s = 40 tok/s
                "total_duration": 600_000_000,
            },
        )
    )
    llm_before = value("model_inference_seconds_count", model="llm")
    tps_before = value("llm_tokens_per_second_count")
    completion_before = value("llm_tokens_total", kind="completion")

    assert client.post("/generate", json={"prompt": "hi"}).status_code == 200

    assert value("model_inference_seconds_count", model="llm") == llm_before + 1
    assert value("llm_tokens_per_second_count") == tps_before + 1
    assert value("llm_tokens_total", kind="completion") == completion_before + 20


def test_llm_failures_are_counted_by_reason(client, respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(side_effect=httpx.ReadTimeout("slow"))
    before = value("inference_errors_total", model="llm", reason="timeout")
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 504
    assert value("inference_errors_total", model="llm", reason="timeout") == before + 1


def test_missing_model_is_counted(client_without_model, customer):
    before = value("inference_errors_total", model="sklearn", reason="model_not_loaded")
    assert client_without_model.post("/predict", json=customer).status_code == 503
    after = value("inference_errors_total", model="sklearn", reason="model_not_loaded")
    assert after == before + 1
