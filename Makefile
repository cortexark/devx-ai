.PHONY: install dev test lint type-check fmt ci clean docs serve-metrics

# --- Setup ---

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# --- Quality ---

test:
	pytest --cov=src/devx --cov-report=term-missing tests/

test-fast:
	pytest -x -q tests/

lint:
	ruff check src/ tests/

type-check:
	mypy src/devx/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

ci: lint type-check test

# --- Run ---

serve-metrics:
	uvicorn devx.metrics.dashboard:app --reload --port 8000

# --- Maintenance ---

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# --- Help ---

help:
	@echo "devx-ai development commands"
	@echo ""
	@echo "  make install       Install package"
	@echo "  make dev           Install with dev dependencies + pre-commit"
	@echo "  make test          Run tests with coverage"
	@echo "  make test-fast     Run tests, stop on first failure"
	@echo "  make lint          Run ruff linter"
	@echo "  make type-check    Run mypy"
	@echo "  make fmt           Auto-format code"
	@echo "  make ci            Run full CI pipeline (lint + type-check + test)"
	@echo "  make serve-metrics Start metrics dashboard on :8000"
	@echo "  make clean         Remove build artifacts"
