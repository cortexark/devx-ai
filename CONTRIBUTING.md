# Contributing to devx-ai

Thanks for your interest in contributing to devx-ai! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/cortexark/devx-ai.git
cd devx-ai

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test file
pytest tests/test_review.py -v
```

## Code Quality

```bash
# Lint
make lint

# Format
make format

# Type check
make typecheck
```

## Making Changes

1. **Fork** the repo and create a feature branch from `main`
2. **Write tests** for any new functionality
3. **Run the full test suite** before submitting
4. **Follow existing code style** — we use ruff for linting and formatting
5. **Write clear commit messages** describing the change
6. **Submit a PR** with a description of your changes

## Architecture

See [docs/architecture.md](docs/architecture.md) for an overview of the codebase structure. Key decisions are documented in [ADRs](docs/adr/).

## Adding a New Review Rule

1. Create a new analyzer in `src/devx/review/`
2. Implement the analysis pattern (AST-based or LLM-augmented)
3. Register it in the review agent pipeline
4. Add tests in `tests/test_review.py`

## Adding a New Test Generator Template

1. Add your template in `src/devx/testgen/templates.py`
2. Register the template type in the generator
3. Add tests validating generation output

## Adding a New SDLC Automation

1. Create the automation in `src/devx/sdlc/`
2. Wire it to the GitHub client
3. Add tests in `tests/test_sdlc.py`

## Reporting Issues

Use [GitHub Issues](https://github.com/cortexark/devx-ai/issues) with the provided templates for bugs and feature requests.

## Code of Conduct

Be respectful, constructive, and collaborative. We're building tools to make developer workflows smarter.
