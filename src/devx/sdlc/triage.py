"""Issue triage and priority assignment.

Analyzes issue content to assign priority (P0-P4), severity, labels,
and optionally suggest an assignee based on file ownership patterns.
"""

from __future__ import annotations

import json
import logging

from devx.core.config import LLMConfig
from devx.core.llm_client import LLMClient
from devx.core.models import IssuePriority, Severity, TriageResult

logger = logging.getLogger(__name__)

_TRIAGE_SYSTEM_PROMPT = """\
You are an expert at triaging software issues. Given an issue title, description,
and optional metadata, determine the priority and severity.

Priority levels:
- P0: Production outage, data loss, security breach. Immediate response required.
- P1: Major feature broken for many users. Fix within hours.
- P2: Feature degraded or workaround exists. Fix within days.
- P3: Minor issue, cosmetic, or edge case. Fix within a sprint.
- P4: Nice-to-have, tech debt, minor improvement. Backlog.

Severity levels:
- critical: System down, data corruption, security vulnerability
- high: Major feature broken, significant user impact
- medium: Feature degraded, workaround available
- low: Minor inconvenience, cosmetic issue
- info: Question, discussion, or improvement suggestion

Return a JSON object with:
- "priority": one of "P0", "P1", "P2", "P3", "P4"
- "severity": one of "critical", "high", "medium", "low", "info"
- "labels": array of suggested label strings
- "reasoning": brief explanation

Return ONLY the JSON object, no markdown fences.
"""

# Keyword-based priority signals for heuristic fallback
_URGENCY_KEYWORDS: dict[str, IssuePriority] = {
    "outage": IssuePriority.P0,
    "production down": IssuePriority.P0,
    "data loss": IssuePriority.P0,
    "security vulnerability": IssuePriority.P0,
    "crash": IssuePriority.P1,
    "broken": IssuePriority.P1,
    "cannot": IssuePriority.P1,
    "error": IssuePriority.P2,
    "slow": IssuePriority.P2,
    "wrong": IssuePriority.P2,
    "improve": IssuePriority.P3,
    "enhance": IssuePriority.P3,
    "typo": IssuePriority.P4,
    "cosmetic": IssuePriority.P4,
    "nice to have": IssuePriority.P4,
}

_SEVERITY_KEYWORDS: dict[str, Severity] = {
    "outage": Severity.CRITICAL,
    "data loss": Severity.CRITICAL,
    "security": Severity.CRITICAL,
    "crash": Severity.HIGH,
    "broken": Severity.HIGH,
    "cannot": Severity.HIGH,
    "error": Severity.MEDIUM,
    "slow": Severity.MEDIUM,
    "wrong": Severity.MEDIUM,
    "improve": Severity.LOW,
    "typo": Severity.INFO,
    "question": Severity.INFO,
}


class IssueTriage:
    """Triage issues by analyzing title and description.

    Example::

        triage = IssueTriage(llm_config=LLMConfig(api_key="sk-..."))
        result = await triage.triage(
            title="Login page returns 500 for SSO users",
            description="Since the last deploy, all SSO users see a 500 error...",
        )
        print(result.priority, result.severity)  # P1 high
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        *,
        team_members: list[str] | None = None,
    ) -> None:
        self._llm: LLMClient | None = None
        if llm_config:
            self._llm = LLMClient(llm_config)
        self._team_members = team_members or []

    async def triage(
        self,
        *,
        title: str,
        description: str = "",
        existing_labels: list[str] | None = None,
        reporter: str = "",
    ) -> TriageResult:
        """Triage an issue and return priority assignment.

        Args:
            title: Issue title.
            description: Issue body / description.
            existing_labels: Labels already on the issue.
            reporter: Username of the issue reporter.

        Returns:
            TriageResult with priority, severity, and labels.
        """
        if self._llm:
            return await self._triage_with_llm(
                title=title,
                description=description,
                existing_labels=existing_labels or [],
                reporter=reporter,
            )
        return self._triage_with_heuristics(
            title=title,
            description=description,
        )

    async def _triage_with_llm(
        self,
        *,
        title: str,
        description: str,
        existing_labels: list[str],
        reporter: str,
    ) -> TriageResult:
        """Use LLM for sophisticated triage."""
        if not self._llm:
            return self._triage_with_heuristics(title=title, description=description)

        prompt_parts = [
            f"Issue Title: {title}",
            f"Issue Description: {description or '(none)'}",
        ]
        if existing_labels:
            prompt_parts.append(f"Existing Labels: {', '.join(existing_labels)}")
        if reporter:
            prompt_parts.append(f"Reporter: {reporter}")

        prompt = "\n\n".join(prompt_parts)

        try:
            response = await self._llm.complete(
                prompt,
                system=_TRIAGE_SYSTEM_PROMPT,
                temperature=0.1,
            )
            return self._parse_llm_response(response.content)
        except Exception:
            logger.exception("LLM triage failed, falling back to heuristics")
            return self._triage_with_heuristics(title=title, description=description)

    def _parse_llm_response(self, raw: str) -> TriageResult:
        """Parse LLM JSON response into TriageResult."""
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
            logger.warning("Failed to parse LLM triage response")
            return TriageResult(
                priority=IssuePriority.P3,
                severity=Severity.MEDIUM,
                reasoning="Failed to parse LLM response",
            )

        try:
            return TriageResult(
                priority=IssuePriority(data.get("priority", "P3")),
                severity=Severity(data.get("severity", "medium")),
                labels=data.get("labels", []),
                reasoning=data.get("reasoning", ""),
            )
        except (ValueError, KeyError):
            return TriageResult(
                priority=IssuePriority.P3,
                severity=Severity.MEDIUM,
                reasoning="Failed to validate LLM response values",
            )

    def _triage_with_heuristics(
        self,
        *,
        title: str,
        description: str,
    ) -> TriageResult:
        """Triage using keyword matching."""
        combined = f"{title} {description}".lower()
        reasons: list[str] = []

        # Find highest priority keyword match
        priority = IssuePriority.P3  # default
        priority_order = list(IssuePriority)
        for keyword, p in _URGENCY_KEYWORDS.items():
            if keyword in combined and priority_order.index(p) < priority_order.index(priority):
                priority = p
                reasons.append(f"Keyword '{keyword}' suggests {p.value}")

        # Find highest severity keyword match
        severity = Severity.MEDIUM  # default
        for keyword, s in _SEVERITY_KEYWORDS.items():
            if keyword in combined and list(Severity).index(s) < list(Severity).index(severity):
                severity = s
                reasons.append(f"Keyword '{keyword}' suggests {s.value}")

        # Generate labels
        labels: list[str] = []
        if "bug" in combined or "error" in combined or "crash" in combined:
            labels.append("bug")
        if "feature" in combined or "request" in combined:
            labels.append("enhancement")
        if "question" in combined or "how to" in combined:
            labels.append("question")
        if "security" in combined:
            labels.append("security")

        if not reasons:
            reasons.append("Default classification (no strong keyword signals)")

        return TriageResult(
            priority=priority,
            severity=severity,
            labels=labels,
            reasoning="; ".join(reasons),
        )
