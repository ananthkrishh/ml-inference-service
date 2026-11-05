import pytest


def test_predict_returns_probability_label_and_version(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn"] == (body["churn_probability"] >= 0.5)
    assert body["model_version"].endswith("-test123")


def test_total_charges_is_optional(client, customer):
    del customer["TotalCharges"]  # brand-new customer: imputed by the pipeline
    assert client.post("/predict", json=customer).status_code == 200


def test_senior_citizen_accepts_int(client, customer):
    customer["SeniorCitizen"] = 1
    assert client.post("/predict", json=customer).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Weekly"},  # not a category seen in training
        {"tenure": -1},  # numerics must be >= 0
        {"MonthlyCharges": "lots"},  # wrong type
        {"Contrct": "One year"},  # typo'd / unknown field
    ],
)
def test_invalid_payload_is_422(client, customer, change):
    customer.update(change)
    r = client.post("/predict", json=customer)
    assert r.status_code == 422
    assert r.json()["detail"]


def test_missing_required_field_is_422(client, customer):
    del customer["Contract"]
    r = client.post("/predict", json=customer)
    assert r.status_code == 422
    assert any(e["loc"][-1] == "Contract" for e in r.json()["detail"])


def test_predict_503_when_model_missing(client_without_model, customer):
    r = client_without_model.post("/predict", json=customer)
    assert r.status_code == 503


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_version"].endswith("-test123")


def test_health_without_model(client_without_model):
    body = client_without_model.get("/health").json()
    assert body["model_loaded"] is False
    assert body["model_version"] is None
    assert body["status"] == "degraded"


def test_openapi_shows_schema_from_metadata(client):
    schema = client.get("/openapi.json").json()
    features = schema["components"]["schemas"]["CustomerFeatures"]
    assert set(features["properties"]["Contract"]["enum"]) == {
        "Month-to-month",
        "One year",
        "Two year",
    }
    assert "TotalCharges" not in features["required"]


def test_array_input_matches_dataframe_pipeline(raw_churn_df, tmp_path, customer):
    """The fast numpy path must give exactly the same answer as the old DataFrame path."""
    import pandas as pd
    from sklearn.base import clone

    from app.model import load_model
    from training.train import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        build_pipeline,
        candidate_models,
        clean,
        train_and_save,
    )

    path = tmp_path / "m.joblib"
    train_and_save(raw_churn_df, path, folds=3, compare=False)
    fast = load_model(path)

    # Same estimator, but selecting columns by NAME and fed a DataFrame (the old way).
    X, y = clean(raw_churn_df)
    slow = build_pipeline(clone(candidate_models(42)["logistic_regression"]))
    slow.set_params(
        preprocess__transformers=[
            ("num", slow.named_steps["preprocess"].transformers[0][1], NUMERIC_FEATURES),
            ("cat", slow.named_steps["preprocess"].transformers[1][1], CATEGORICAL_FEATURES),
        ]
    )
    # Fit on exactly the rows train_and_save used for training.
    from sklearn.model_selection import train_test_split

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    slow.fit(X_train, y_train)

    for total in (447.5, None):  # None = missing TotalCharges, filled by the imputer
        customer["TotalCharges"] = total
        features = fast.features_model.model_validate(customer)
        row = pd.DataFrame([features.model_dump()], columns=fast.metadata["feature_columns"])
        row[NUMERIC_FEATURES] = row[NUMERIC_FEATURES].astype(float)
        assert fast.predict_proba_one(features) == pytest.approx(
            float(slow.predict_proba(row)[0, 1]), abs=1e-12
        )
