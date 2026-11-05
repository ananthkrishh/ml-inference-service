.PHONY: install train run test lint fmt up up-llm down logs

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
