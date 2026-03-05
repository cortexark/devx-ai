# devx-ai

[![CI](https://github.com/cortexark/devx-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/cortexark/devx-ai/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**AI-powered developer tooling platform** -- code review, test generation, SDLC automation, and engineering metrics in a single library.

---

## Why devx-ai

Engineering teams drown in toil: reviewing boilerplate PRs, writing test scaffolds, labeling issues, and manually tracking delivery metrics. devx-ai automates the repetitive parts so engineers focus on design and problem-solving.

The platform combines **static analysis** (tree-sitter AST parsing) with **LLM augmentation** (OpenAI / Anthropic) so every feature works offline with deterministic analysis and improves when an LLM is available.

## Architecture

```
+-----------------------------------------------------------+
|                        devx-ai                            |
+-----------------------------------------------------------+
|                                                           |
|  +----------+  +---------+  +------+  +---------+        |
|  |  review   |  | testgen |  | sdlc |  | metrics |        |
|  |           |  |         |  |      |  |         |        |
|  | AST       |  | Extract |  | Label|  | Collect |        |
|  | Agent     |  | Generate|  | Triage  | Analyze |        |
|  | DiffParse |  | Templat |  | GitHub  | API     |        |
|  +-----+-----+  +----+----+  +--+---+  +----+----+        |
|        |              |         |            |             |
|  +-----+--------------+---------+------------+------+      |
|  |                     core                          |      |
|  |  models.py  config.py  llm_client.py              |      |
|  +---------------------------------------------------+      |
+-----------------------------------------------------------+
         |                    |                    |
    tree-sitter          OpenAI / Anthropic    GitHub API
```

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Review a PR

```python
import asyncio
from devx.review import ReviewAgent

agent = ReviewAgent()  # AST-only, no API key needed

async def main():
    diff = open("my_changes.diff").read()
    result = await agent.review_diff(diff)
    for finding in result.findings:
        print(f"[{finding.severity}] {finding.title} at {finding.location}")

asyncio.run(main())
```

### Generate Tests

```python
import asyncio
from devx.testgen import TestGenerator

generator = TestGenerator()  # Template-based, no API key needed

async def main():
    source = open("src/utils.py").read()
    suite = await generator.generate_for_source(source, module="utils")
    for tc in suite.test_cases:
        print(tc.code)

asyncio.run(main())
```

### Classify a PR

```python
import asyncio
from devx.sdlc import PRLabeler

labeler = PRLabeler()  # Heuristic mode

async def main():
    result = await labeler.classify(
        title="Fix null pointer in auth handler",
        changed_files=["src/auth.py", "tests/test_auth.py"],
    )
    print(result.labels)       # [PRLabel.BUG_FIX, PRLabel.TEST]
    print(result.confidence)   # 0.6

asyncio.run(main())
```

### DORA Metrics

```python
from devx.metrics import DORAAnalyzer
from devx.core.models import DeploymentRecord

analyzer = DORAAnalyzer()
metrics = analyzer.calculate(deployments, window_days=30)

print(f"Deploy frequency: {metrics.deployment_frequency}/day ({metrics.deployment_frequency_rating})")
print(f"Lead time: {metrics.lead_time_seconds}s ({metrics.lead_time_rating})")
print(f"Change failure rate: {metrics.change_failure_rate:.1%}")
print(f"MTTR: {metrics.mttr_seconds}s")
```

### Metrics Dashboard

```bash
make serve-metrics
# Open http://localhost:8000/docs for interactive API docs
```

## Features

### Code Review Agent
- **Two-phase pipeline**: tree-sitter AST analysis + LLM semantic review
- Detects complexity, missing docs, parameter overloads, security issues
- Outputs to GitHub comments, terminal (rich), or JSON
- Works offline with AST-only mode

### Test Generator
- Extracts function signatures, docstrings, and type hints
- Generates pytest tests via LLM or template fallback
- Template registry for unit, edge case, and integration patterns
- Supports async functions

### SDLC Automation
- **PR Labeler**: Classifies PRs as bug-fix, feature, refactor, docs, etc.
- **Issue Triage**: Assigns priority (P0-P4) and severity
- GitHub API client with rate-limit awareness

### Engineering Metrics
- DORA metrics: deployment frequency, lead time, change failure rate, MTTR
- Benchmark ratings (elite, high, medium, low)
- Trend analysis and team comparisons
- FastAPI dashboard with REST API

## Configuration

Settings load from environment variables (with `DEVX_` prefix) or a `devx.yaml` file:

```yaml
llm:
  provider: openai        # or "anthropic"
  model: gpt-4o
  temperature: 0.2
  max_tokens: 4096

github:
  token: ${GITHUB_TOKEN}  # use env var reference
  rate_limit_buffer: 100

metrics:
  window_days: 30
  cache_ttl_seconds: 300

log_level: INFO
```

Environment variables:

```bash
export DEVX_LLM_API_KEY="sk-..."
export DEVX_LLM_PROVIDER="openai"
export DEVX_GITHUB_TOKEN="ghp_..."
```

## Development

```bash
make dev          # Install with dev dependencies
make test         # Run tests with coverage
make lint         # Run ruff linter
make type-check   # Run mypy
make ci           # Full CI pipeline
make fmt          # Auto-format code
```

## Architecture Decision Records

- [ADR 001: tree-sitter for AST Analysis](docs/adr/001-tree-sitter-for-ast-analysis.md) -- Why tree-sitter over the ast module
- [ADR 002: LLM-Augmented Code Review](docs/adr/002-llm-augmented-code-review.md) -- Two-phase review pipeline design
- [ADR 003: DORA Metrics Implementation](docs/adr/003-dora-metrics-implementation.md) -- Calculating DORA from GitHub data

## Project Structure

```
devx-ai/
  src/devx/
    core/          Pydantic models, config, LLM client
    review/        Code review agent (AST + LLM)
    testgen/       Test generation engine
    sdlc/          PR labeling, issue triage, GitHub client
    metrics/       DORA metrics, collector, dashboard API
  tests/           Comprehensive test suite
  docs/adr/        Architecture decision records
  examples/        Usage examples
```

## Contributing

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests first (TDD)
4. Ensure `make ci` passes
5. Open a pull request with a clear description

All code must have:
- Type hints on all function signatures
- Docstrings on all public methods
- Test coverage for new functionality
- No hardcoded API keys or secrets

## License

[MIT](LICENSE)
