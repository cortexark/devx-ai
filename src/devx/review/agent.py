"""Code review agent combining AST analysis with LLM-augmented suggestions.

The ReviewAgent orchestrates a two-phase review:
  1. **Static analysis** via tree-sitter AST to catch structural issues
     (complexity, missing docs, parameter counts) with zero latency cost.
  2. **LLM analysis** for semantic issues (logic errors, security,
     naming, design patterns) that require understanding intent.

Findings from both phases are merged, deduplicated, and ranked.
"""

from __future__ import annotations

import json
import logging
import time

from devx.core.config import LLMConfig
from devx.core.llm_client import LLMClient
from devx.core.models import (
    Category,
    CodeLocation,
    FileDiff,
    ReviewFinding,
    ReviewResult,
    Severity,
)
from devx.review.analyzer import ASTAnalyzer
from devx.review.diff_parser import DiffParser

logger = logging.getLogger(__name__)

_REVIEW_SYSTEM_PROMPT = """\
You are an expert code reviewer. Analyze the provided code changes and return
findings as a JSON array. Each finding must have:
- "title": short description (max 200 chars)
- "description": detailed explanation
- "severity": one of "critical", "high", "medium", "low", "info"
- "category": one of "bug", "security", "performance", "style", \
"maintainability", "complexity", "testing", "documentation"
- "file": file path
- "start_line": line number
- "suggestion": optional suggested fix

Focus on:
1. Logic errors and bugs
2. Security vulnerabilities (injection, secrets, auth)
3. Performance issues (N+1, unnecessary allocations)
4. Error handling gaps
5. API design issues

Do NOT flag style issues that a linter would catch.
Return ONLY the JSON array, no markdown fences or explanation.
Return an empty array [] if no significant issues are found.
"""


class ReviewAgent:
    """Orchestrates AST + LLM code review.

    Example::

        agent = ReviewAgent(llm_config=LLMConfig(api_key="sk-..."))
        result = await agent.review_diff(raw_diff_text)
        print(result.summary)
        for f in result.findings:
            print(f.severity, f.title)

    For offline / testing usage, omit the LLM config to get AST-only review::

        agent = ReviewAgent()
        result = await agent.review_diff(raw_diff_text)
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        *,
        enable_llm: bool = True,
    ) -> None:
        self._analyzer = ASTAnalyzer()
        self._diff_parser = DiffParser()
        self._llm: LLMClient | None = None
        self._enable_llm = enable_llm

        if llm_config and enable_llm:
            self._llm = LLMClient(llm_config)

    async def review_diff(self, diff_text: str) -> ReviewResult:
        """Review a unified diff and return aggregated findings.

        Args:
            diff_text: Raw output from ``git diff``.

        Returns:
            ReviewResult with all findings, summary, and metadata.
        """
        start = time.monotonic()
        file_diffs = self._diff_parser.parse(diff_text)

        if not file_diffs:
            return ReviewResult(
                summary="No changes to review.",
                duration_seconds=time.monotonic() - start,
            )

        # Phase 1: AST analysis on added/modified Python files
        ast_findings = self._run_ast_analysis(file_diffs)

        # Phase 2: LLM analysis
        llm_findings = await self._run_llm_analysis(file_diffs) if self._llm else []

        # Merge and deduplicate
        all_findings = self._merge_findings(ast_findings, llm_findings)
        all_findings.sort(key=lambda f: list(Severity).index(f.severity))

        elapsed = time.monotonic() - start
        summary = self._build_summary(all_findings, len(file_diffs))

        return ReviewResult(
            findings=all_findings,
            summary=summary,
            files_analyzed=len(file_diffs),
            duration_seconds=round(elapsed, 2),
        )

    async def review_file(self, source: str, file_path: str = "<stdin>") -> ReviewResult:
        """Review a single file's source code (not a diff).

        Args:
            source: Full file source code.
            file_path: Path for location references.

        Returns:
            ReviewResult with findings.
        """
        start = time.monotonic()
        result = self._analyzer.analyze_python(source, file_path)
        elapsed = time.monotonic() - start

        return ReviewResult(
            findings=result.findings,
            summary=self._build_summary(result.findings, 1),
            files_analyzed=1,
            duration_seconds=round(elapsed, 2),
        )

    def _run_ast_analysis(self, file_diffs: list[FileDiff]) -> list[ReviewFinding]:
        """Run AST analysis on Python files in the diff."""
        findings: list[ReviewFinding] = []

        for fd in file_diffs:
            if fd.is_deleted:
                continue
            path = fd.path
            if not path.endswith(".py"):
                continue

            # Reconstruct added content from hunks for analysis
            added_code = "\n".join(line for hunk in fd.hunks for line in hunk.added_lines)
            if added_code.strip():
                result = self._analyzer.analyze_python(added_code, path)
                findings.extend(result.findings)

        return findings

    async def _run_llm_analysis(self, file_diffs: list[FileDiff]) -> list[ReviewFinding]:
        """Run LLM analysis on the diff."""
        if not self._llm:
            return []

        # Build a concise diff summary for the LLM
        diff_summary = self._build_diff_context(file_diffs)
        if not diff_summary.strip():
            return []

        try:
            response = await self._llm.complete(
                diff_summary,
                system=_REVIEW_SYSTEM_PROMPT,
                temperature=0.1,
            )
            return self._parse_llm_findings(response.content)
        except Exception:
            files = [fd.path for fd in file_diffs if not fd.is_deleted]
            logger.exception(
                "LLM review failed for files %s, continuing with AST-only results", files
            )
            return []

    def _build_diff_context(self, file_diffs: list[FileDiff]) -> str:
        """Build a diff summary string for LLM consumption."""
        parts: list[str] = []
        for fd in file_diffs:
            if fd.is_deleted:
                continue
            parts.append(f"=== {fd.path} ===")
            for hunk in fd.hunks:
                parts.append(hunk.content)
            parts.append("")
        return "\n".join(parts)

    def _parse_llm_findings(self, raw: str) -> list[ReviewFinding]:
        """Parse LLM JSON response into ReviewFinding list."""
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return []

        if not isinstance(data, list):
            return []

        findings: list[ReviewFinding] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                finding = ReviewFinding(
                    title=item.get("title", "Untitled finding"),
                    description=item.get("description", ""),
                    severity=Severity(item.get("severity", "medium")),
                    category=Category(item.get("category", "maintainability")),
                    location=CodeLocation(
                        file=item.get("file", "<unknown>"),
                        start_line=int(item.get("start_line", 1)),
                    ),
                    suggestion=item.get("suggestion"),
                    confidence=0.7,  # LLM findings get lower default confidence
                )
                findings.append(finding)
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping malformed LLM finding: %s", exc)
                continue

        return findings

    def _merge_findings(
        self,
        ast_findings: list[ReviewFinding],
        llm_findings: list[ReviewFinding],
    ) -> list[ReviewFinding]:
        """Merge AST and LLM findings, removing near-duplicates.

        Deduplication is based on matching file + line + category.
        When duplicates exist, the AST finding (higher confidence) is kept.
        """
        seen: set[tuple[str, int, str]] = set()
        merged: list[ReviewFinding] = []

        # AST findings take priority
        for f in ast_findings:
            key = (f.location.file, f.location.start_line, f.category.value)
            seen.add(key)
            merged.append(f)

        for f in llm_findings:
            key = (f.location.file, f.location.start_line, f.category.value)
            if key not in seen:
                seen.add(key)
                merged.append(f)

        return merged

    @staticmethod
    def _build_summary(findings: list[ReviewFinding], file_count: int) -> str:
        """Generate a human-readable summary."""
        if not findings:
            return f"Reviewed {file_count} files. No issues found."

        severity_counts: dict[str, int] = {}
        for f in findings:
            severity_counts[f.severity.value] = severity_counts.get(f.severity.value, 0) + 1

        parts = [f"Reviewed {file_count} files, found {len(findings)} issues:"]
        for sev in ("critical", "high", "medium", "low", "info"):
            count = severity_counts.get(sev, 0)
            if count > 0:
                parts.append(f"  {sev}: {count}")

        return "\n".join(parts)
