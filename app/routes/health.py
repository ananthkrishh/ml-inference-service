from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    # TODO (step 3): report ollama_reachable as well.
    model = getattr(request.app.state, "model", None)
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=model.version if model is not None else None,
    )
