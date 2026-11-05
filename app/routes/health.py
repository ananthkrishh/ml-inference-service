from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Always 200 while the API process is alive; `status` says if a dependency is down.

    200 + "degraded" (not 503) so Docker doesn't restart a healthy API just because
    Ollama is still starting or the model file is missing — restarting wouldn't fix either.
    """
    model = getattr(request.app.state, "model", None)
    ollama = getattr(request.app.state, "ollama", None)
    ollama_ok = await ollama.is_reachable() if ollama is not None else False
    return HealthResponse(
        status="ok" if model is not None and ollama_ok else "degraded",
        model_loaded=model is not None,
        model_version=model.version if model is not None else None,
        ollama_reachable=ollama_ok,
    )
