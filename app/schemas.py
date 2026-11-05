"""Pydantic request/response models.

The /predict request model is NOT hand-written. It is built at startup from the
model bundle's metadata (feature_columns, numeric_features, category_values), so
the API always accepts exactly what the loaded model was trained on.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model

# Numeric features the client may leave out. The pipeline's median imputer fills
# them in (brand-new customers have no TotalCharges in the real data).
OPTIONAL_NUMERIC_FEATURES = {"TotalCharges"}


def _int_to_str(value: Any) -> Any:
    """Let clients send SeniorCitizen as 1 as well as "1" (the model saw strings)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value


def build_customer_features_model(metadata: dict) -> type[BaseModel]:
    """Create the CustomerFeatures request model from model metadata.

    - categoricals become Literal[...] of the values seen in training -> 422 on typos
    - numerics are floats >= 0; those in OPTIONAL_NUMERIC_FEATURES may be omitted
    - unknown fields are rejected (extra="forbid") so misspelled keys fail loudly
    """
    numeric = set(metadata["numeric_features"])
    categories: dict[str, list[str]] = metadata["category_values"]

    fields: dict[str, Any] = {}
    for name in metadata["feature_columns"]:
        if name in numeric:
            if name in OPTIONAL_NUMERIC_FEATURES:
                fields[name] = (float | None, Field(default=None, ge=0))
            else:
                fields[name] = (float, Field(ge=0))
        else:
            allowed = Literal[tuple(categories[name])]  # type: ignore[valid-type]
            fields[name] = (Annotated[allowed, BeforeValidator(_int_to_str)], ...)

    return create_model(
        "CustomerFeatures",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(ge=0, le=1, description="P(customer churns)")
    churn: bool = Field(description="churn_probability >= the model's decision threshold")
    model_version: str


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1, max_length=4000, examples=["Explain customer churn in 2 lines."]
    )
    max_tokens: int | None = Field(
        default=None, ge=1, le=2048, description="Answer length cap; the server cap still applies"
    )
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="0 = predictable, higher = more creative"
    )


class GenerateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_seconds: float = Field(description="Time inside Ollama, including model load")
    load_seconds: float = Field(description="Model load time; ~0 when the model is warm")
    tokens_per_second: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description='"ok" when everything works, "degraded" if a part is down')
    model_loaded: bool
    model_version: str | None = None
    ollama_reachable: bool
