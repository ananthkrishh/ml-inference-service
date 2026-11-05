"""Load test for the inference API.

Pretend customers hit POST /predict with random but VALID profiles, built from the model's
own metadata (the same allowed values the API checks against). /generate users only run
when LOADTEST_LLM=1, because they need Ollama.

Interactive (web UI at http://localhost:8089):
    locust -f loadtest/locustfile.py --host http://localhost:8000

Headless, the way step 8's numbers were measured:
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 20 -t 60s --only-summary --csv loadtest/results/<name>

Include the LLM endpoint (needs `make up-llm`):
    LOADTEST_LLM=1 locust -f loadtest/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import FastHttpUser, between, task

METADATA = json.loads(
    (Path(__file__).resolve().parent.parent / "models" / "churn_model.json").read_text()
)
CATEGORIES: dict[str, list[str]] = METADATA["category_values"]

PROMPTS = [
    "In one sentence, what is customer churn?",
    "Give two reasons a telecom customer might cancel.",
    "Suggest one retention offer for a month-to-month customer.",
]


def random_customer() -> dict:
    tenure = random.randint(0, 72)
    monthly = round(random.uniform(18, 120), 2)
    customer = {name: random.choice(values) for name, values in CATEGORIES.items()}
    customer.update(
        tenure=tenure,
        MonthlyCharges=monthly,
        # Brand-new customers have no TotalCharges, like the real data.
        TotalCharges=round(tenure * monthly, 2) if tenure else None,
    )
    return customer


class PredictUser(FastHttpUser):
    """A client app asking for churn predictions, with short pauses in between."""

    wait_time = between(0.05, 0.15)

    @task
    def predict(self) -> None:
        with self.client.post("/predict", json=random_customer(), catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")


class GenerateUser(FastHttpUser):
    """Someone chatting with Llama 3. LLM calls take seconds, so they ask less often."""

    abstract = os.environ.get("LOADTEST_LLM") != "1"  # abstract = never spawned
    wait_time = between(1, 3)

    @task
    def generate(self) -> None:
        body = {"prompt": random.choice(PROMPTS), "max_tokens": 64}
        with self.client.post("/generate", json=body, catch_response=True, timeout=120) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
