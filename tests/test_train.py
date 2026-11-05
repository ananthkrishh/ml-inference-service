import json

import joblib
import sklearn

from training.train import FEATURE_COLUMNS, clean, to_model_input, train_and_save


def test_clean_handles_blank_total_charges(raw_churn_df):
    X, y = clean(raw_churn_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X["TotalCharges"].dtype.kind == "f"
    assert X["TotalCharges"].isna().sum() == (raw_churn_df["TotalCharges"] == " ").sum()
    assert set(y.unique()) <= {0, 1}
    assert X["SeniorCitizen"].isin(["0", "1"]).all()


def test_train_and_save_roundtrip(raw_churn_df, tmp_path):
    out = tmp_path / "model.joblib"
    meta = train_and_save(raw_churn_df, out, folds=3, compare=True, data_sha256="abc1234def")

    assert out.exists()
    assert out.with_suffix(".json").exists()
    saved_meta = json.loads(out.with_suffix(".json").read_text())
    assert saved_meta["model_version"] == meta["model_version"]
    assert meta["model_version"].endswith("-abc1234")
    assert meta["sklearn_version"] == sklearn.__version__
    assert meta["algorithm"] in meta["cv_roc_auc"]
    assert 0.0 <= meta["test_metrics"]["roc_auc"] <= 1.0

    bundle = joblib.load(out)
    X, _ = clean(raw_churn_df)
    assert meta["input_format"] == "array"
    proba = bundle["pipeline"].predict_proba(to_model_input(X.head(5)))
    assert proba.shape == (5, 2)


def test_unseen_category_and_missing_numeric_do_not_crash(raw_churn_df, tmp_path):
    out = tmp_path / "model.joblib"
    train_and_save(raw_churn_df, out, folds=3, compare=False)
    pipeline = joblib.load(out)["pipeline"]

    X, _ = clean(raw_churn_df)
    row = X.head(1).copy()
    row["PaymentMethod"] = "Crypto"  # never seen in training
    row["TotalCharges"] = float("nan")
    proba = pipeline.predict_proba(to_model_input(row))
    assert 0.0 <= proba[0, 1] <= 1.0
