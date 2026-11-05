"""POST /generate and the Ollama part of /health, with Ollama mocked by respx.

No real Ollama is needed: respx intercepts the HTTP calls our httpx client makes.
"""

import json

import httpx
import pytest

from tests.conftest import OLLAMA_URL

# What Ollama's /api/generate returns (durations are in nanoseconds).
OLLAMA_REPLY = {
    "model": "llama3.2:3b",
    "response": "Churn is when customers stop using a service.",
    "done": True,
    "total_duration": 1_500_000_000,
    "load_duration": 200_000_000,
    "prompt_eval_count": 12,
    "eval_count": 40,
    "eval_duration": 1_000_000_000,
}


def test_generate_returns_text_and_stats(client, respx_mock):
    route = respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, json=OLLAMA_REPLY)
    )
    r = client.post("/generate", json={"prompt": "What is churn?"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == OLLAMA_REPLY["response"]
    assert body["completion_tokens"] == 40
    assert body["tokens_per_second"] == 40.0  # 40 tokens in 1 second
    assert body["load_seconds"] == 0.2
    assert body["total_seconds"] == 1.5

    sent = json.loads(route.calls.last.request.content)
    assert sent["prompt"] == "What is churn?"
    assert sent["stream"] is False
    assert sent["keep_alive"]  # keeps the model warm between requests
    assert sent["options"]["num_predict"] == 256  # server default cap


def test_max_tokens_is_capped_by_server(client, respx_mock):
    route = respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, json=OLLAMA_REPLY)
    )
    client.post("/generate", json={"prompt": "hi", "max_tokens": 2000})  # asks for more
    assert json.loads(route.calls.last.request.content)["options"]["num_predict"] == 256

    client.post("/generate", json={"prompt": "hi", "max_tokens": 50, "temperature": 0})
    sent = json.loads(route.calls.last.request.content)["options"]
    assert sent == {"num_predict": 50, "temperature": 0}


@pytest.mark.parametrize(
    ("ollama_behaviour", "expected_status"),
    [
        (httpx.ConnectError("refused"), 503),  # Ollama not running
        (httpx.ReadTimeout("slow"), 504),  # Ollama too slow
        (httpx.Response(404, json={"error": "model not found"}), 503),  # model not pulled
        (httpx.Response(500, text="boom"), 502),  # Ollama broke
    ],
)
def test_ollama_failures_map_to_clear_status_codes(
    client, respx_mock, ollama_behaviour, expected_status
):
    route = respx_mock.post(f"{OLLAMA_URL}/api/generate")
    if isinstance(ollama_behaviour, Exception):
        route.mock(side_effect=ollama_behaviour)
    else:
        route.mock(return_value=ollama_behaviour)

    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == expected_status
    assert r.json()["detail"]


def test_model_not_pulled_message_says_how_to_fix(client, respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(return_value=httpx.Response(404))
    assert "ollama pull" in client.post("/generate", json={"prompt": "hi"}).json()["detail"]


@pytest.mark.parametrize(
    "body",
    [{}, {"prompt": ""}, {"prompt": "hi", "max_tokens": 0}, {"prompt": "hi", "extra": 1}],
)
def test_invalid_generate_request_is_422(client, body):
    assert client.post("/generate", json=body).status_code == 422


def test_health_ok_when_model_and_ollama_up(client, respx_mock):
    respx_mock.get(f"{OLLAMA_URL}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama_reachable"] is True


def test_health_degraded_when_ollama_down(client):
    # Nothing mocked: port 9 refuses the connection, like a stopped Ollama.
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ollama_reachable"] is False
    assert body["model_loaded"] is True
