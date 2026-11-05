.PHONY: install train run test lint fmt up down

install:
	pip install -r requirements-dev.txt

train:
	python -m training.train

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

up:
	docker compose up --build -d

down:
	docker compose down
