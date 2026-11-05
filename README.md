# ML Inference Service

REST inference API serving a scikit-learn customer-churn model and a local Llama 3 model via Ollama.
Built and tested in GitHub Actions; Prometheus latency metrics track and guide performance work.

**Stack:** Python · FastAPI · scikit-learn · Ollama (Llama 3) · Docker · GitHub Actions · Prometheus

## Architecture

```
client ──► FastAPI (Uvicorn)
             ├── POST /predict  ──► scikit-learn pipeline (in-process, joblib)
             ├── POST /generate ──► Ollama container (Llama 3, :11434)
             ├── GET  /health
             └── GET  /metrics  ◄── Prometheus scrape ──► Grafana
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install        # runtime + dev dependencies
make train          # downloads the Telco churn dataset, trains, writes models/churn_model.joblib
make test
make run            # http://localhost:8000/docs
```

## Model (step 1)

- **Data:** IBM Telco Customer Churn — 7,043 customers, 19 features, ~27% churn.
- **Pipeline:** `ColumnTransformer` (median impute + scale numerics, one-hot categoricals with
  `handle_unknown="ignore"`) → classifier. The API sends raw values; all preprocessing lives in
  the saved pipeline, so training and serving can't drift apart.
- **Selection:** logistic regression vs. histogram gradient boosting, 5-fold stratified CV on
  ROC-AUC, both with balanced class weights.
- **Artifact:** `models/churn_model.joblib` holds `{pipeline, metadata}`; `churn_model.json` has
  the same metadata (version, sklearn version, data hash, feature list, allowed categories,
  metrics, single-row latency baseline).

| Metric (held-out 20%) | Value |
|---|---|
| ROC-AUC | 0.841 |
| Recall | 0.783 |
| Precision | 0.503 |
| F1 | 0.613 |
| Single-row predict p95 | ~13 ms |

## Roadmap

- [x] 1. Train and version the model
- [x] 2. FastAPI `/predict`
- [x] 3. Ollama `/generate`
- [ ] 4. Prometheus metrics
- [x] 5. Tests (API, mocked Ollama)
- [ ] 6. Docker + Compose
- [ ] 7. GitHub Actions CI/CD
- [ ] 8. Load test and optimise (before/after latency)
- [ ] 9. Grafana dashboard + docs
