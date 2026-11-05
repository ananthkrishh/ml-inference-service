"""POST /predict — churn prediction from the scikit-learn pipeline."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.metrics import INFERENCE_ERRORS, MODEL_INFERENCE
from app.schemas import PredictionResponse

router = APIRouter(tags=["ml"])


# A plain `def` (not `async def`): FastAPI runs it in a thread pool, so the
# CPU-bound predict_proba call doesn't block the event loop for other requests.
@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: Annotated[dict[str, Any], Body()]) -> PredictionResponse:
    model = request.app.state.model
    if model is None:
        INFERENCE_ERRORS.labels(model="sklearn", reason="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate against the schema built from the loaded model's metadata.
    try:
        features = model.features_model.model_validate(payload)
    except ValidationError as exc:
        # Prefix "body" so errors look like FastAPI's own: {"loc": ["body", "tenure"], ...}
        errors = [{**e, "loc": ("body", *e["loc"])} for e in exc.errors(include_url=False)]
        raise RequestValidationError(errors) from exc

    # Time ONLY the model call, so we can compare it with the whole-request latency.
    with MODEL_INFERENCE.labels(model="sklearn").time():
        probability = model.predict_proba_one(features)
    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn=probability >= model.threshold,
        model_version=model.version,
    )
