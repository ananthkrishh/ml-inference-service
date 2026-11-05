"""Thin async wrapper around Ollama's HTTP API (/api/generate and /api/tags).

Ollama runs Llama 3 in its own process/container. We talk to it over HTTP with ONE
shared httpx.AsyncClient (created in the FastAPI lifespan), so TCP connections are
reused instead of opened for every request.

Errors are turned into three exception types so the route can map them cleanly:
    OllamaTimeout      -> 504 Gateway Timeout   (Ollama too slow)
    OllamaUnavailable  -> 503 Service Unavailable (not running / model not pulled)
    OllamaError        -> 502 Bad Gateway        (anything else odd from Ollama)
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

NS_PER_S = 1_000_000_000  # Ollama reports durations in nanoseconds


class OllamaError(Exception):
    """Ollama answered, but not with something we can use."""


class OllamaUnavailable(OllamaError):
    """Ollama isn't reachable, or the model hasn't been pulled."""


class OllamaTimeout(OllamaError):
    """Ollama didn't answer within the configured timeout."""


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_seconds: float  # whole request inside Ollama, including model load
    load_seconds: float  # time spent loading the model into memory (0 when warm)
    tokens_per_second: float  # generation speed (completion tokens / eval time)


class OllamaClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        model: str,
        keep_alive: str = "10m",
        max_tokens: int = 256,
    ) -> None:
        self.http = http
        self.model = model
        # keep_alive: how long Ollama keeps the model in memory after a request.
        # Without it, an idle model is unloaded and the next request pays seconds of load time.
        self.keep_alive = keep_alive
        self.max_tokens = max_tokens  # hard cap; requests can ask for fewer, never more

    async def generate(
        self, prompt: str, max_tokens: int | None = None, temperature: float | None = None
    ) -> Generation:
        # num_predict caps the answer length; long answers are the main driver of latency.
        num_predict = min(max_tokens or self.max_tokens, self.max_tokens)
        options: dict = {"num_predict": num_predict}
        if temperature is not None:
            options["temperature"] = temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # one JSON reply instead of a token-by-token stream
            "keep_alive": self.keep_alive,
            "options": options,
        }

        try:
            resp = await self.http.post("/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            name = exc.__class__.__name__
            raise OllamaTimeout(f"Ollama did not answer in time ({name})") from exc
        except httpx.TransportError as exc:
            raise OllamaUnavailable(f"Cannot reach Ollama: {exc.__class__.__name__}") from exc

        if resp.status_code == 404:
            # Ollama returns 404 when the model isn't pulled yet.
            raise OllamaUnavailable(
                f"Model {self.model!r} not found in Ollama; run `ollama pull {self.model}`"
            )
        if resp.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        eval_count = int(data.get("eval_count", 0))
        eval_seconds = data.get("eval_duration", 0) / NS_PER_S
        return Generation(
            text=data.get("response", ""),
            model=data.get("model", self.model),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=eval_count,
            total_seconds=round(data.get("total_duration", 0) / NS_PER_S, 3),
            load_seconds=round(data.get("load_duration", 0) / NS_PER_S, 3),
            tokens_per_second=round(eval_count / eval_seconds, 1) if eval_seconds else 0.0,
        )

    async def is_reachable(self, timeout_s: float = 1.0) -> bool:
        """Cheap liveness check for /health: list local models with a short timeout."""
        try:
            resp = await self.http.get("/api/tags", timeout=timeout_s)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200
