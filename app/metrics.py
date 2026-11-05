"""Prometheus metrics, the request-timing middleware and the /metrics endpoint.

Metrics (all scraped from GET /metrics):
  http_request_duration_seconds   histogram  {method, route, status}  whole request, end to end
  http_requests_in_flight         gauge                               requests being handled now
  model_inference_seconds         histogram  {model}                  just the model call
                                                                      (sklearn | llm)
  llm_tokens_per_second           histogram                           Llama 3 generation speed
  llm_tokens_total                counter    {kind}                   prompt / completion tokens
  inference_errors_total          counter    {model, reason}          failed inferences by cause

Note for later (step 8): with several uvicorn workers each process has its own counters;
that needs prometheus_client's multiprocess mode (PROMETHEUS_MULTIPROC_DIR).
"""

from __future__ import annotations

import time

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Buckets span ~1 ms (/predict) up to 60 s (a long LLM answer), so both fit on one chart.
LATENCY_BUCKETS = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60,
)  # fmt: skip

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency, measured by the middleware",
    ["method", "route", "status"],
    buckets=LATENCY_BUCKETS,
)
IN_FLIGHT = Gauge("http_requests_in_flight", "HTTP requests currently being handled")
MODEL_INFERENCE = Histogram(
    "model_inference_seconds",
    "Time spent inside the model call only (no HTTP, no validation)",
    ["model"],
    buckets=LATENCY_BUCKETS,
)
LLM_TOKENS_PER_SECOND = Histogram(
    "llm_tokens_per_second",
    "LLM generation speed reported by Ollama",
    buckets=(1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200),
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens processed", ["kind"])
INFERENCE_ERRORS = Counter("inference_errors_total", "Failed inferences", ["model", "reason"])

UNMATCHED_ROUTE = "unmatched"
SKIP_ROUTES = {"/metrics"}  # don't time Prometheus' own scrapes


class PrometheusMiddleware:
    """Times every HTTP request and counts the ones in flight.

    A plain ASGI middleware (not BaseHTTPMiddleware): lower overhead, and it sees the
    real status code even when a route raises an HTTPException.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        status_code = 500  # if the app crashes before answering, that's what the client gets

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        IN_FLIGHT.inc()
        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            IN_FLIGHT.dec()
            route = _route_template(scope)
            if route not in SKIP_ROUTES:
                REQUEST_LATENCY.labels(scope["method"], route, str(status_code)).observe(
                    time.perf_counter() - start
                )


def _route_template(scope: Scope) -> str:
    """The route PATTERN (e.g. "/predict"), never the raw URL.

    Using raw URLs would create a new time series for every random path a bot tries
    (/wp-login.php, /.env, ...): a "cardinality explosion" that can take Prometheus down.
    FastAPI stores the matched route in the scope; anything unmatched shares one label.
    """
    route = scope.get("route")
    return getattr(route, "path", UNMATCHED_ROUTE)


def metrics_endpoint() -> Response:
    """GET /metrics: all metrics in Prometheus' plain-text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
