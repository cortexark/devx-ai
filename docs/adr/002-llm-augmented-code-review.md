# ADR 002: LLM-Augmented Code Review

## Status
Accepted

## Date
2026-03-01

## Context

Static analysis catches structural issues (complexity, missing docs, parameter counts) but misses semantic problems: logic errors, security vulnerabilities, design anti-patterns, and naming that misleads. LLMs can reason about code intent but are expensive, slow, and occasionally wrong.

We need an architecture that gets the best of both worlds: fast, deterministic static analysis combined with LLM-powered semantic review.

## Decision

We implement a **two-phase review pipeline**:

```
Phase 1: AST Analysis (tree-sitter)        Phase 2: LLM Analysis
  - Complexity detection                      - Logic error detection
  - Missing docstrings                        - Security vulnerability scan
  - Parameter count violations                - API design review
  - Class size violations                     - Naming quality
  - Import analysis                           - Error handling gaps
       |                                           |
       +-----------> Merge + Deduplicate <---------+
                           |
                    Ranked Findings
```

### Key Design Decisions

**1. AST-first, LLM-second.** AST analysis runs unconditionally and completes in milliseconds. LLM analysis is optional and only called when configured. This means the review agent works offline and in CI environments without API keys.

**2. Deduplication by (file, line, category).** When AST and LLM both flag the same location for the same category, the AST finding wins because it has higher confidence (deterministic). This prevents noisy duplicate findings.

**3. Structured LLM output.** The LLM is instructed to return JSON matching our `ReviewFinding` schema. We parse and validate the output with Pydantic. Invalid findings are silently dropped rather than crashing the review.

**4. Cost optimization.** We send only the diff content to the LLM, not the entire file. For large PRs, we truncate context and focus on changed hunks. Average cost per review: ~$0.02 with GPT-4o.

**5. Confidence scoring.** AST findings default to 0.8 confidence (deterministic, rule-based). LLM findings default to 0.7 confidence (probabilistic). Consumers can filter by confidence threshold.

### Prompt Engineering

The system prompt is explicit about:
- What to look for (bugs, security, performance, error handling)
- What NOT to flag (style issues that linters catch)
- Output format (JSON array with specific fields)
- Focus on "why" not "what" (explain the impact, not just the symptom)

We iterate on the prompt by measuring:
- Precision: % of findings that humans agree with
- Recall: % of known issues that the LLM catches
- Signal-to-noise: ratio of high/critical to info findings

## Consequences

- The `ReviewAgent` class orchestrates both phases and is the public API.
- LLM findings are always labeled with `confidence=0.7` so consumers can distinguish them.
- The system degrades gracefully: LLM failures produce a warning log and return AST-only results.
- Adding new static rules requires adding checks to `ASTAnalyzer._generate_findings()`.
- Changing LLM behavior requires updating the system prompt, not code.

## Trade-offs

| Aspect | Our Approach | Alternative |
|--------|-------------|-------------|
| Latency | AST: ~50ms, LLM: ~2-5s | LLM-only: ~5-10s |
| Cost | ~$0.02/review | LLM-only: ~$0.10/review |
| Offline support | Yes (AST-only) | No (LLM required) |
| False positive rate | Lower (dedup + confidence) | Higher (LLM hallucinations) |
| Multi-language | tree-sitter grammars | Limited by LLM training data |

## Risks

1. **LLM hallucinations.** The model may fabricate findings. Mitigated by confidence scoring and validation against the actual diff.
2. **Prompt injection.** Malicious code in the diff could attempt to manipulate the LLM. Mitigated by treating LLM output as untrusted and validating with Pydantic.
3. **API cost spikes.** Large monorepo PRs could generate expensive prompts. Mitigated by truncating diff context and setting `max_tokens`.
