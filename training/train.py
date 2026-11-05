"""Step 1: train the customer-churn classifier and save a versioned model bundle.

Usage:
    python -m training.train                       # download data if needed, train, save
    python -m training.train --skip-compare        # only train the default model
    python -m training.train --output models/x.joblib

Output:
    models/churn_model.joblib   {"pipeline": sklearn Pipeline, "metadata": dict}
    models/churn_model.json     the same metadata, human-readable

The pipeline takes RAW feature values (strings for categoricals, numbers for
numerics), so the API never re-implements preprocessing. That is the main
guard against training/serving skew.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger("train")

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_DATA_PATH = Path("data/raw/telco_churn.csv")
DEFAULT_OUTPUT = Path("models/churn_model.joblib")

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- data


def download_data(path: Path, url: str = DATA_URL, timeout: int = 30) -> Path:
    """Download the dataset once and cache it locally."""
    if path.exists():
        log.info("Using cached dataset at %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed URL)
        path.write_bytes(resp.read())
    return path


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the raw CSV into (features, target).

    - TotalCharges arrives as text, with blanks for brand-new customers
      (tenure 0). Blanks become NaN and are imputed inside the pipeline.
    - SeniorCitizen is 0/1 but is really a category, so it is cast to str.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    y = (df[TARGET] == "Yes").astype(int)
    X = df[FEATURE_COLUMNS]
    return X, y


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------ model


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # handle_unknown="ignore": an unseen category at inference time becomes an
    # all-zero one-hot row instead of a 500 error.
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(seed: int) -> dict[str, object]:
    # class_weight="balanced" because only ~27% of customers churn.
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def select_model(
    X: pd.DataFrame, y: pd.Series, seed: int, folds: int, names: list[str]
) -> tuple[str, dict[str, float]]:
    """Pick the candidate with the best mean cross-validated ROC-AUC."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = candidate_models(seed)
    scores: dict[str, float] = {}
    for name in names:
        auc = cross_val_score(build_pipeline(models[name]), X, y, cv=cv, scoring="roc_auc")
        scores[name] = round(float(auc.mean()), 4)
        log.info("CV ROC-AUC %-24s %.4f ± %.4f", name, auc.mean(), auc.std())
    best = max(scores, key=scores.get)
    return best, scores


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
    }


def benchmark_single_row(pipeline: Pipeline, X: pd.DataFrame, n: int = 300) -> dict[str, float]:
    """Latency of one-row predictions — the shape the API will actually see.

    Gives you a baseline to compare against the HTTP latency in step 4.
    """
    rows = [X.iloc[[i % len(X)]] for i in range(n)]
    pipeline.predict_proba(rows[0])  # warm-up
    timings = []
    for row in rows:
        t0 = time.perf_counter()
        pipeline.predict_proba(row)
        timings.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": round(float(np.percentile(timings, 50)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
        "p99_ms": round(float(np.percentile(timings, 99)), 3),
    }


def category_values(X: pd.DataFrame) -> dict[str, list[str]]:
    """Allowed values per categorical feature — the API schema is built from these."""
    return {c: sorted(X[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}


# ------------------------------------------------------------------------- main


def train_and_save(
    df: pd.DataFrame,
    output: Path,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    folds: int = 5,
    compare: bool = True,
    data_sha256: str | None = None,
) -> dict:
    X, y = clean(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    names = list(candidate_models(seed)) if compare else ["logistic_regression"]
    best_name, cv_scores = select_model(X_train, y_train, seed, folds, names)
    log.info("Selected model: %s", best_name)

    pipeline = build_pipeline(candidate_models(seed)[best_name])
    pipeline.fit(X_train, y_train)

    test_metrics = evaluate(pipeline, X_test, y_test)
    latency = benchmark_single_row(pipeline, X_test)

    trained_at = datetime.now(UTC)
    version = trained_at.strftime("%Y%m%d%H%M%S")
    if data_sha256:
        version += f"-{data_sha256[:7]}"

    metadata = {
        "model_name": "churn_classifier",
        "model_version": version,
        "algorithm": best_name,
        "trained_at": trained_at.isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_sha256": data_sha256,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": round(float(y.mean()), 4),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_values": category_values(X),
        "decision_threshold": 0.5,
        "cv_roc_auc": cv_scores,
        "test_metrics": test_metrics,
        "single_row_latency": latency,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, output, compress=3)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s (version %s)", output, version)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the churn model.")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--skip-compare", action="store_true", help="train logistic regression only")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    path = download_data(args.data_path)
    df = pd.read_csv(path)
    meta = train_and_save(
        df,
        args.output,
        seed=args.seed,
        test_size=args.test_size,
        folds=args.cv_folds,
        compare=not args.skip_compare,
        data_sha256=file_sha256(path),
    )
    print(
        json.dumps(
            {
                k: meta[k]
                for k in (
                    "model_version",
                    "algorithm",
                    "cv_roc_auc",
                    "test_metrics",
                    "single_row_latency",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
