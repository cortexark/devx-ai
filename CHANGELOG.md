# Changelog

All notable changes to devx-ai will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-05

### Added
- Core models and configuration with Pydantic v2
- LLM client abstraction for provider-agnostic integration
- Code review agent with tree-sitter AST analysis
- LLM-augmented review suggestions engine
- Diff parser for PR analysis
- Test generation from function signatures and docstrings
- Template-based test scaffolding
- SDLC automation: PR labeler, issue triage
- GitHub client for API integration
- DORA metrics analyzer (deployment frequency, lead time, MTTR, change failure rate)
- Metrics collector and FastAPI dashboard
- GitHub Actions CI pipeline (Python 3.11/3.12/3.13)
- Architecture Decision Records (3 ADRs)
- Example scripts for PR review and test generation
