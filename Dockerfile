# syntax=docker/dockerfile:1
#
# Multi-stage build for the inference API.
#   stage 1 "builder": installs Python packages into a virtualenv (needs pip and build tools)
#   stage 2 "runtime": copies ONLY that virtualenv + our code into a clean image
# Result: a smaller image with no pip cache or build leftovers, running as a non-root user.
#
# The model file is NOT trained here: run `make train` first (CI does the same in step 7).
# Python matches the version the model is trained with locally (pickles are version-sensitive).

ARG PYTHON_IMAGE=python:3.13.18-slim

# ---------------------------------------------------------------- stage 1: builder
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements on their own first: Docker caches this layer, so code-only changes
# don't reinstall every package (rebuilds take seconds instead of minutes).
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------- stage 2: runtime
FROM ${PYTHON_IMAGE} AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=models/churn_model.joblib

# A dedicated unprivileged user: if someone ever broke into the app, they wouldn't be root.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
# Named explicitly so the build fails loudly if you forgot `make train`.
COPY models/churn_model.joblib models/churn_model.json ./models/

USER app
EXPOSE 8000

# Docker asks "are you alive?" every 15 s. /health returns 200 whenever the API process
# works (it reports a missing Ollama as "degraded" instead of failing), so this checks
# the API itself. Uses Python's stdlib: no curl needed in the slim image.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
