# Architecture

## System Overview

devx-ai is structured as a modular Python library with four domain modules
and a shared core. Each module can be used independently or composed into
larger workflows.

```
+-----------------------------------------------------------+
|                        devx-ai                            |
+-----------------------------------------------------------+
|                                                           |
|  +----------+  +---------+  +------+  +---------+        |
|  |  review   |  | testgen |  | sdlc |  | metrics |        |
|  |           |  |         |  |      |  |         |        |
|  | Analyzer  |  | Extract |  | Label|  | Collect |        |
|  | Agent     |  | Generat |  | Triag|  | Analyze |        |
|  | DiffParse |  | Templat |  | GitHu|  | Dashbrd |        |
|  +-----+-----+  +----+----+  +--+---+  +----+----+        |
|        |              |         |            |             |
|  +-----+--------------+---------+------------+------+      |
|  |                     core                          |      |
|  |  models.py  config.py  llm_client.py              |      |
|  +---------------------------------------------------+      |
|                                                           |
+-----------------------------------------------------------+
         |                    |                    |
    tree-sitter          OpenAI / Anthropic    GitHub API
```

## Module Responsibilities

### core/
- **models.py**: Pydantic v2 models for all data boundaries (findings, diffs, metrics, etc.)
- **config.py**: YAML + environment variable configuration with validation
- **llm_client.py**: Unified async client for OpenAI and Anthropic with retry logic

### review/
- **diff_parser.py**: Parses unified diff format into structured `FileDiff`/`DiffHunk` models
- **analyzer.py**: tree-sitter-based AST analysis for structural code issues
- **agent.py**: Orchestrates AST + LLM two-phase review pipeline
- **suggestions.py**: Formats findings for GitHub comments, terminal, or JSON

### testgen/
- **extractor.py**: Extracts function signatures, docstrings, and type hints using Python `ast`
- **generator.py**: Generates pytest test cases via LLM or template fallback
- **templates.py**: Registry of reusable test code templates

### sdlc/
- **labeler.py**: PR auto-classification using LLM or filename heuristics
- **triage.py**: Issue priority/severity assignment
- **github_client.py**: Async GitHub REST API wrapper with rate limiting

### metrics/
- **collector.py**: Gathers deployment and PR data from GitHub
- **analyzer.py**: Calculates DORA metrics, trends, and team comparisons
- **dashboard.py**: FastAPI REST API for metrics consumption

## Design Principles

1. **Graceful degradation.** Every module works without an LLM configured. AST analysis, heuristic labeling, and template-based test generation provide baseline functionality.

2. **Structured boundaries.** All data crosses module boundaries through Pydantic models. No raw dicts or unvalidated strings at public interfaces.

3. **Async-first.** External I/O (LLM calls, GitHub API) uses async/await. Internal computation (AST parsing, template rendering) is synchronous for simplicity.

4. **Configuration over code.** LLM provider, model, temperature, and thresholds are configurable via environment variables or YAML. Changing behavior should not require code changes.

5. **Testability.** Every module has a testing seam: LLM clients can be mocked, GitHub responses can be faked, and AST analysis works on string input (no filesystem required).

## Data Flow: Code Review

```
git diff output
     |
     v
DiffParser.parse() --> list[FileDiff]
     |
     v
ReviewAgent.review_diff()
     |
     +---> Phase 1: ASTAnalyzer.analyze_python() --> list[ReviewFinding]
     |
     +---> Phase 2: LLMClient.complete() --> JSON --> list[ReviewFinding]
     |
     v
_merge_findings() --> deduplicated, sorted list[ReviewFinding]
     |
     v
ReviewResult
     |
     v
SuggestionFormatter.to_github_comment() / .to_json() / .print_terminal()
```

## Data Flow: DORA Metrics

```
GitHub Deployments API
     |
     v
MetricsCollector.collect_deployments() --> list[DeploymentRecord]
     |
     v
DORAAnalyzer.calculate() --> DORAMetrics
     |
     v
Dashboard API (FastAPI)
     |
     v
GET /api/v1/dora --> DORAResponse with ratings
```
