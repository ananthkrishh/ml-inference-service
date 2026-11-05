.PHONY: install train run test lint fmt up up-llm down logs loadtest

MODEL := models/churn_model.joblib

install:
	pip install -r requirements-dev.txt

train:
	python -m training.train

# Train only if the model file doesn't exist yet (Docker needs it to build the image).
$(MODEL):
	python -m training.train

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

up: $(MODEL)
	docker compose up --build -d

up-llm: $(MODEL)
	docker compose --profile llm up --build -d

down:
	docker compose --profile llm down

logs:
	docker compose logs -f api

# Headless Locust run: make loadtest HOST=http://localhost:8080 NAME=after-fix USERS=20
HOST ?= http://localhost:8000
NAME ?= run
USERS ?= 20
loadtest:
	locust -f loadtest/locustfile.py --host $(HOST) --headless -u $(USERS) -r $(USERS) \
		-t 60s --only-summary --csv loadtest/results/$(NAME)
	python loadtest/summary.py loadtest/results/$(NAME)
