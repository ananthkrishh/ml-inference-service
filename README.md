# ML Inference Service

[![CI](https://github.com/ananthkrishh/ml-inference-service/actions/workflows/ci.yml/badge.svg)](https://github.com/ananthkrishh/ml-inference-service/actions/workflows/ci.yml)

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

Or the whole stack in Docker (API, Prometheus, Grafana):

```bash
make up             # trains the model first if needed, then docker compose up --build
make up-llm         # same + Ollama, pulls llama3.2:3b (~2 GB) once
make down
```

API http://localhost:8000/docs · Prometheus http://localhost:9090 · Grafana http://localhost:3000

Or just the published API image (built and smoke-tested by CI on every push to `main`):

```bash
docker run -p 8000:8000 ghcr.io/ananthkrishh/ml-inference-service:latest
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
| Single-row predict p95 (model only) | 1.1 ms (was 2.9 ms before step 8) |

## Performance (step 8)

Load-tested with Locust against the Docker image (2 CPU limit), on an Apple M4 Pro.

| Change | Load | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|
| Baseline | 20 users | 181 req/s | 7 ms | 23 ms | 34 ms |
| + numpy array input instead of a 1-row DataFrame | 20 users | 188 req/s | 4 ms | 12 ms | 17 ms |
| + 2 uvicorn workers (Prometheus multiprocess mode) | 20 users | 191 req/s | 3 ms | **7 ms** | 10 ms |
| Baseline image | 100 users | 318 req/s | 210 ms | 280 ms | 310 ms |
| Final image | 100 users | **929 req/s** | 5 ms | **17 ms** | 29 ms |

How the bottleneck was found: profiling showed 81% of a prediction was preprocessing a one-row
pandas DataFrame (the classifier itself was 1.4%); selecting columns by position and passing a
numpy array made predictions 2.5x faster with identical outputs. Under concurrent load,
Python's GIL serialises CPU work inside one process, so a second worker process cut tail
latency further. Ruled out by measurement: Docker overhead (<1 ms), BLAS/OpenMP thread
counts, CPU throttling.

```bash
pip install -r requirements-loadtest.txt
make loadtest HOST=http://localhost:8000 NAME=my-run   # 20 users, 60 s, headless
```

## Roadmap

- [x] 1. Train and version the model
- [x] 2. FastAPI `/predict`
- [x] 3. Ollama `/generate`
- [x] 4. Prometheus metrics
- [x] 5. Tests (API, mocked Ollama)
- [x] 6. Docker + Compose
- [x] 7. GitHub Actions CI/CD
- [x] 8. Load test and optimise (before/after latency)
- [ ] 9. Grafana dashboard + docs
