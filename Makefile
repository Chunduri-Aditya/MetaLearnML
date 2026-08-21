VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: setup lint test test-integration run web benchmark-fetch benchmark-smoke benchmark-full benchmark-report-check clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTEST) -q --cov=engine --cov=models --cov=utils --cov-report=term-missing --cov-fail-under=70 -m "not neo4j and not slow"

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

test-integration:
	$(PYTEST) -q -m "integration and not neo4j"

run:
	$(PYTHON) main.py

web:
	$(VENV)/bin/uvicorn web.app:app --host 0.0.0.0 --port 8000

benchmark-fetch:
	$(PYTHON) -m benchmarks.run fetch

benchmark-smoke:
	$(PYTHON) -m benchmarks.run smoke

benchmark-full:
	$(PYTHON) -m benchmarks.run full

benchmark-report-check:
	$(PYTHON) -m benchmarks.run report-check

clean:
	rm -rf .pytest_cache __pycache__
