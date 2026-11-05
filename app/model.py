"""Loading the churn model bundle and running a single prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from app.schemas import build_customer_features_model

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChurnModel:
    pipeline: Pipeline
    metadata: dict
    features_model: type[BaseModel]  # the CustomerFeatures request schema

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    def predict_proba_one(self, features: BaseModel) -> float:
        """P(churn) for one customer.

        The pipeline gets RAW values in the exact column order used in training;
        all preprocessing happens inside the pipeline (no training/serving skew).
        """
        columns = self.metadata["feature_columns"]
        row = pd.DataFrame([features.model_dump()], columns=columns)
        # Optional numerics arrive as None; the imputer needs a float NaN.
        numeric = self.metadata["numeric_features"]
        row[numeric] = row[numeric].astype(float)
        return float(self.pipeline.predict_proba(row)[0, 1])


def load_model(path: str | Path) -> ChurnModel:
    """Load {pipeline, metadata} from joblib. Raises if the file is missing or invalid."""
    bundle = joblib.load(path)
    metadata = bundle["metadata"]

    trained_with = metadata.get("sklearn_version")
    if trained_with != sklearn.__version__:
        log.warning(
            "Model was trained with scikit-learn %s but %s is installed; "
            "predictions may differ or fail. Retrain or pin the matching version.",
            trained_with,
            sklearn.__version__,
        )

    model = ChurnModel(
        pipeline=bundle["pipeline"],
        metadata=metadata,
        features_model=build_customer_features_model(metadata),
    )
    log.info("Loaded model %s from %s", model.version, path)
    return model
