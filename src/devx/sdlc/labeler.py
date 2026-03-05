"""PR auto-labeling with LLM classification.

Analyzes pull request content (title, description, file changes) to
automatically assign labels like bug-fix, feature, refactor, docs, etc.
Uses LLM for nuanced classification that goes beyond filename heuristics.
"""

from __future__ import annotations

import json
import logging

from devx.core.config import LLMConfig
from devx.core.llm_client import LLMClient
from devx.core.models import LabelClassification, PRLabel

logger = logging.getLogger(__name__)

_LABELER_SYSTEM_PROMPT = """\
You are a PR classification expert. Given a pull request's title, description,
and list of changed files, determine the appropriate labels.

Available labels:
- bug-fix: Fixes a bug or incorrect behavior
- feature: Adds new functionality
- refactor: Code restructuring without behavior change
- docs: Documentation only changes
- test: Test additions or modifications
- chore: Build, CI, dependency updates
- security: Security-related changes
- performance: Performance improvements
- breaking-change: Changes that break backward compatibility

Return a JSON object with:
- "labels": array of label strings from the list above
- "confidence": float between 0 and 1
- "reasoning": brief explanation of why these labels apply

Return ONLY the JSON object, no markdown fences.
"""

# Heuristic rules for fast labeling without LLM
_FILE_PATTERN_LABELS: dict[str, PRLabel] = {
    "test": PRLabel.TEST,
    "spec": PRLabel.TEST,
    "README": PRLabel.DOCS,
    ".md": PRLabel.DOCS,
    "docs/": PRLabel.DOCS,
    "Makefile": PRLabel.CHORE,
    "Dockerfile": PRLabel.CHORE,
    ".yml": PRLabel.CHORE,
    ".yaml": PRLabel.CHORE,
    "requirements": PRLabel.CHORE,
    "pyproject.toml": PRLabel.CHORE,
    "package.json": PRLabel.CHORE,
}


class PRLabeler:
    """Classify pull requests and assign labels.

    Supports two modes:
    - **LLM mode**: Uses language model for nuanced classification.
    - **Heuristic mode**: Uses filename patterns when no LLM is configured.

    Example::

        labeler = PRLabeler(llm_config=LLMConfig(api_key="sk-..."))
        result = await labeler.classify(
            title="Fix null pointer in user auth",
            description="Handles the case where session token is expired",
            changed_files=["src/auth/session.py", "tests/test_session.py"],
        )
        print(result.labels)  # [PRLabel.BUG_FIX, PRLabel.TEST]
    """

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self._llm: LLMClient | None = None
        if llm_config:
            self._llm = LLMClient(llm_config)

    async def classify(
        self,
        *,
        title: str,
        description: str = "",
        changed_files: list[str] | None = None,
        diff_summary: str = "",
    ) -> LabelClassification:
        """Classify a pull request and return suggested labels.

        Args:
            title: PR title.
            description: PR description / body.
            changed_files: List of changed file paths.
            diff_summary: Optional abbreviated diff for additional context.

        Returns:
            LabelClassification with labels, confidence, and reasoning.
        """
        if self._llm:
            return await self._classify_with_llm(
                title=title,
                description=description,
                changed_files=changed_files or [],
                diff_summary=diff_summary,
            )
        return self._classify_with_heuristics(
            title=title,
            description=description,
            changed_files=changed_files or [],
        )

    async def _classify_with_llm(
        self,
        *,
        title: str,
        description: str,
        changed_files: list[str],
        diff_summary: str,
    ) -> LabelClassification:
        """Use LLM for classification."""
        if not self._llm:
            return LabelClassification(confidence=0.0, reasoning="No LLM configured")

        prompt_parts = [
            f"PR Title: {title}",
            f"PR Description: {description or '(none)'}",
            f"Changed Files: {', '.join(changed_files[:50]) if changed_files else '(none)'}",
        ]
        if diff_summary:
            prompt_parts.append(f"Diff Summary:\n{diff_summary[:2000]}")

        prompt = "\n\n".join(prompt_parts)

        try:
            response = await self._llm.complete(
                prompt,
                system=_LABELER_SYSTEM_PROMPT,
                temperature=0.1,
            )
            return self._parse_llm_response(response.content)
        except Exception:
            logger.exception("LLM labeling failed, falling back to heuristics")
            return self._classify_with_heuristics(
                title=title,
                description=description,
                changed_files=changed_files,
            )

    def _parse_llm_response(self, raw: str) -> LabelClassification:
        """Parse LLM response into LabelClassification."""
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
            logger.warning("Failed to parse LLM labeling response")
            return LabelClassification(confidence=0.0, reasoning="Parse error")

        labels: list[PRLabel] = []
        for label_str in data.get("labels", []):
            try:
                labels.append(PRLabel(label_str))
            except ValueError:
                logger.debug("Unknown label from LLM: %s", label_str)
                continue

        return LabelClassification(
            labels=labels,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

    def _classify_with_heuristics(
        self,
        *,
        title: str,
        description: str,
        changed_files: list[str],
    ) -> LabelClassification:
        """Classify using filename patterns and title keywords."""
        labels: set[PRLabel] = set()
        reasons: list[str] = []

        # Title-based heuristics
        title_lower = title.lower()
        title_keywords: dict[str, PRLabel] = {
            "fix": PRLabel.BUG_FIX,
            "bug": PRLabel.BUG_FIX,
            "feat": PRLabel.FEATURE,
            "add": PRLabel.FEATURE,
            "refactor": PRLabel.REFACTOR,
            "doc": PRLabel.DOCS,
            "readme": PRLabel.DOCS,
            "test": PRLabel.TEST,
            "chore": PRLabel.CHORE,
            "ci": PRLabel.CHORE,
            "security": PRLabel.SECURITY,
            "perf": PRLabel.PERFORMANCE,
            "breaking": PRLabel.BREAKING,
        }

        for keyword, label in title_keywords.items():
            if keyword in title_lower:
                labels.add(label)
                reasons.append(f"Title contains '{keyword}'")

        # File-based heuristics
        for file_path in changed_files:
            for pattern, label in _FILE_PATTERN_LABELS.items():
                if pattern in file_path:
                    labels.add(label)
                    reasons.append(f"File '{file_path}' matches pattern '{pattern}'")
                    break

        # Default to feature if no labels matched
        if not labels:
            labels.add(PRLabel.FEATURE)
            reasons.append("Default classification")

        return LabelClassification(
            labels=sorted(labels, key=lambda lbl: lbl.value),
            confidence=0.6 if len(reasons) > 1 else 0.4,
            reasoning="; ".join(reasons[:5]),
        )
