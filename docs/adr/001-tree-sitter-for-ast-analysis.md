# ADR 001: tree-sitter for AST Analysis

## Status
Accepted

## Date
2026-03-01

## Context

devx-ai needs to parse source code structurally to detect complexity issues, missing documentation, parameter overloads, and other anti-patterns during code review. We evaluated three approaches:

1. **Python `ast` module** -- stdlib, zero dependencies, Python-only
2. **tree-sitter** -- multi-language, incremental parsing, widely adopted
3. **Roslyn / Language Server Protocol** -- language-specific, heavy dependencies

The code review agent must support at least Python and JavaScript, with the architecture open to adding more languages without rewriting the analyzer.

## Decision

We chose **tree-sitter** for AST analysis.

### Why tree-sitter

**Multi-language support.** tree-sitter has grammar packages for 100+ languages. Adding JavaScript, TypeScript, Go, or Rust analysis requires installing a grammar package and writing extraction logic -- not a new parser. The `ast` module is Python-only.

**Incremental parsing.** tree-sitter can re-parse a document after edits in O(change) rather than O(document). This matters when analyzing diffs incrementally or when building editor integrations.

**Error recovery.** tree-sitter produces a valid parse tree even for syntactically broken code. This is critical for analyzing in-progress diffs where the code may not be complete. Python's `ast.parse` raises `SyntaxError` on invalid input.

**Performance.** tree-sitter is written in C with Python bindings. On a 10K-line file:
- tree-sitter: ~5ms
- Python ast: ~15ms
- LSP cold start: ~2s

For a review agent processing dozens of files per PR, this overhead compounds.

**Industry adoption.** GitHub's code navigation, Neovim, Helix, and Zed all use tree-sitter. Choosing it aligns with where the ecosystem is heading.

### Trade-offs

**Dependency weight.** tree-sitter requires compiled C extensions, which complicates installation in environments without a C compiler. We mitigate this by providing pre-built wheels and a fallback regex analyzer.

**API surface.** tree-sitter's Python API is lower-level than `ast`. Walking the tree requires understanding node types and cursor navigation. We encapsulate this complexity in `ASTAnalyzer`.

**Grammar versioning.** Grammar packages must be compatible with the tree-sitter runtime version. We pin both in `pyproject.toml` and test compatibility in CI.

## Consequences

- The `ASTAnalyzer` class wraps tree-sitter and exposes `analyze_python()`, `analyze_javascript()`, etc.
- A `_analyze_with_fallback()` method uses regex when tree-sitter is unavailable (e.g., in minimal Docker images).
- Adding a new language requires: (1) add grammar dependency, (2) write `_extract_*` methods, (3) add tests.
- We do NOT use tree-sitter for type-level analysis; type checking remains a separate concern.

## Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| Python `ast` | Zero deps, simple API | Python-only, no error recovery |
| Roslyn / LSP | Deep semantic analysis | Heavy, language-specific, slow startup |
| regex heuristics | No dependencies | Fragile, false positives, no structure |
