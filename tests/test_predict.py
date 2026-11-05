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
