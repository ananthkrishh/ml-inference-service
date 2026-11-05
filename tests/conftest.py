import numpy as np
import pandas as pd
import pytest

YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["Yes", "No", "No internet service"]


@pytest.fixture
def raw_churn_df() -> pd.DataFrame:
    """Small synthetic frame with the same schema and quirks as the Telco CSV.

    Tests use this instead of the real dataset so CI never downloads data.
    """
    rng = np.random.default_rng(0)
    n = 300
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # the real CSV has blanks for brand-new customers
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    # make churn learnable: month-to-month + short tenure churn more often
    churn_p = 0.15 + 0.4 * (contract == "Month-to-month") + 0.3 * (tenure < 12)
    churn = np.where(rng.random(n) < churn_p, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(YES_NO, n),
            "Dependents": rng.choice(YES_NO, n),
            "tenure": tenure,
            "PhoneService": rng.choice(YES_NO, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(INTERNET_ADDON, n),
            "OnlineBackup": rng.choice(INTERNET_ADDON, n),
            "DeviceProtection": rng.choice(INTERNET_ADDON, n),
            "TechSupport": rng.choice(INTERNET_ADDON, n),
            "StreamingTV": rng.choice(INTERNET_ADDON, n),
            "StreamingMovies": rng.choice(INTERNET_ADDON, n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(YES_NO, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        }
    )


# ---------------------------------------------------------------- API fixtures

# Fake Ollama address for tests. Port 9 refuses connections instantly, so a test that
# forgets to mock Ollama fails fast; tests that need Ollama mock this URL with respx.
OLLAMA_URL = "http://127.0.0.1:9"

VALID_CUSTOMER = {
    "tenure": 5,
    "MonthlyCharges": 89.5,
    "TotalCharges": 447.5,
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


@pytest.fixture
def customer() -> dict:
    """A valid /predict payload (a fresh copy, so tests can mutate it)."""
    return dict(VALID_CUSTOMER)


@pytest.fixture
def model_path(raw_churn_df, tmp_path):
    """Train a tiny model on the synthetic data — no download, a few seconds."""
    from training.train import train_and_save

    path = tmp_path / "model.joblib"
    train_and_save(raw_churn_df, path, folds=3, compare=False, data_sha256="test123abc")
    return path


def _client_with_model_path(monkeypatch, path):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import app

    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_URL)
    monkeypatch.setenv("LLM_MAX_TOKENS", "256")
    get_settings.cache_clear()  # settings are cached; re-read the env var
    with TestClient(app) as client:  # `with` runs the lifespan (loads the model)
        yield client
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch, model_path):
    yield from _client_with_model_path(monkeypatch, model_path)


@pytest.fixture
def client_without_model(monkeypatch, tmp_path):
    yield from _client_with_model_path(monkeypatch, tmp_path / "missing.joblib")
