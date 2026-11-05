"""FastAPI application entry point.

Run locally with:  uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import get_settings
from app.model import load_model
from app.routes import generate, health, predict

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Load the model ONCE at startup, not per request (that would cost ms on every call).
    # If it's missing we still start, so /health can report it; /predict returns 503.
    try:
        app.state.model = load_model(settings.model_path)
    except Exception:
        log.exception("Could not load model from %s; /predict will return 503", settings.model_path)
        app.state.model = None
    app.openapi_schema = None  # rebuild /docs with the loaded model's schema

    # TODO (step 3): create one shared httpx.AsyncClient for Ollama here.
    yield
    # TODO (step 3): close the httpx client on shutdown.


# docs_url=None: we serve our own /docs below (same Swagger UI, plus one style tweak).
app = FastAPI(title="ML Inference Service", version="0.1.0", lifespan=lifespan, docs_url=None)

# In /docs, hide the static "Responses" list of example replies (placeholder values that
# are easy to mistake for real results). The live "Server response" after Execute stays.
# The full response schemas are still in /openapi.json for anyone who needs them.
HIDE_STATIC_RESPONSES_CSS = """
<style>
  table.responses-table:not(.live-responses-table),
  .responses-inner > div > h4:last-child { display: none; }  /* that table's "Responses" title */
</style>
"""


@app.get("/docs", include_in_schema=False)
async def docs() -> HTMLResponse:
    page = get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{app.title} - Docs")
    html = page.body.decode().replace("</head>", HIDE_STATIC_RESPONSES_CSS + "</head>")
    return HTMLResponse(html)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """There's no home page; send people to the interactive docs instead."""
    return RedirectResponse(url="/docs")


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(generate.router)


def custom_openapi() -> dict:
    """Show the metadata-built CustomerFeatures schema in /docs.

    /predict validates its body against a model created at startup, so FastAPI
    can't see it at import time; we patch it into the OpenAPI document here.
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    model = getattr(app.state, "model", None)
    if model is not None:
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        components["CustomerFeatures"] = model.features_model.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        body = schema["paths"]["/predict"]["post"]["requestBody"]
        body["content"]["application/json"]["schema"] = {
            "$ref": "#/components/schemas/CustomerFeatures"
        }
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

# TODO (step 4): add the Prometheus latency middleware and mount /metrics.
