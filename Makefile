VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: setup test run web clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTEST) -q

run:
	$(PYTHON) main.py

web:
	$(VENV)/bin/uvicorn web.app:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache __pycache__
