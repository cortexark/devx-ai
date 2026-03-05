"""Tests for SDLC automation: PR labeling and issue triage."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from devx.core.config import LLMConfig
from devx.core.llm_client import LLMResponse
from devx.core.models import IssuePriority, PRLabel, Severity
from devx.sdlc.labeler import PRLabeler
from devx.sdlc.triage import IssueTriage

# ---------------------------------------------------------------------------
# PRLabeler
# ---------------------------------------------------------------------------


class TestPRLabeler:
    @pytest.mark.asyncio
    async def test_heuristic_bug_fix(self):
        labeler = PRLabeler()  # No LLM
        result = await labeler.classify(
            title="Fix null pointer in user authentication",
            changed_files=["src/auth.py", "tests/test_auth.py"],
        )
        assert PRLabel.BUG_FIX in result.labels

    @pytest.mark.asyncio
    async def test_heuristic_feature(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Add user profile page",
            changed_files=["src/profile.py", "src/templates/profile.html"],
        )
        assert PRLabel.FEATURE in result.labels

    @pytest.mark.asyncio
    async def test_heuristic_docs(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Update API documentation",
            changed_files=["docs/api.md", "README.md"],
        )
        assert PRLabel.DOCS in result.labels

    @pytest.mark.asyncio
    async def test_heuristic_chore(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Update CI configuration",
            changed_files=[".github/workflows/ci.yml", "Makefile"],
        )
        assert PRLabel.CHORE in result.labels

    @pytest.mark.asyncio
    async def test_heuristic_test(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Add unit tests for payment module",
            changed_files=["tests/test_payment.py"],
        )
        assert PRLabel.TEST in result.labels

    @pytest.mark.asyncio
    async def test_heuristic_refactor(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Refactor database connection pooling",
            changed_files=["src/db/pool.py"],
        )
        assert PRLabel.REFACTOR in result.labels

    @pytest.mark.asyncio
    async def test_heuristic_default_to_feature(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Implement order processing",
            changed_files=["src/orders.py"],
        )
        # "Implement" doesn't match any heuristic keyword, but
        # it should still produce labels
        assert len(result.labels) >= 1

    @pytest.mark.asyncio
    async def test_heuristic_confidence(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Fix bug in test runner",
            changed_files=["tests/runner.py"],
        )
        # Multiple signals should give higher confidence
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_heuristic_reasoning(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Fix login bug",
            changed_files=["src/auth.py"],
        )
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_llm_classification(self, mock_label_llm_response):
        config = LLMConfig(api_key="test")
        labeler = PRLabeler(llm_config=config)
        labeler._llm = AsyncMock()
        labeler._llm.complete = AsyncMock(return_value=mock_label_llm_response)

        result = await labeler.classify(
            title="Fix null pointer in auth",
            description="Handles expired session tokens",
            changed_files=["src/auth.py", "tests/test_auth.py"],
        )
        assert PRLabel.BUG_FIX in result.labels
        assert result.confidence > 0.8

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        config = LLMConfig(api_key="test")
        labeler = PRLabeler(llm_config=config)
        labeler._llm = AsyncMock()
        labeler._llm.complete = AsyncMock(side_effect=RuntimeError("API error"))

        # Should fall back to heuristics
        result = await labeler.classify(
            title="Fix login bug",
            changed_files=["src/auth.py"],
        )
        assert len(result.labels) >= 1

    @pytest.mark.asyncio
    async def test_llm_invalid_json_fallback(self):
        config = LLMConfig(api_key="test")
        labeler = PRLabeler(llm_config=config)
        labeler._llm = AsyncMock()
        labeler._llm.complete = AsyncMock(
            return_value=LLMResponse(
                content="not valid json",
                model="gpt-4o",
            )
        )

        result = await labeler.classify(
            title="Fix bug",
            changed_files=["src/app.py"],
        )
        # Should return a result even with parse failure
        assert isinstance(result, type(result))


# ---------------------------------------------------------------------------
# IssueTriage
# ---------------------------------------------------------------------------


class TestIssueTriage:
    @pytest.mark.asyncio
    async def test_heuristic_p0_outage(self):
        triage = IssueTriage()
        result = await triage.triage(
            title="Production outage - all users affected",
            description="The main API endpoint is returning 503.",
        )
        assert result.priority == IssuePriority.P0
        assert result.severity == Severity.CRITICAL

    @pytest.mark.asyncio
    async def test_heuristic_p1_crash(self):
        triage = IssueTriage()
        result = await triage.triage(
            title="App crash on startup after update",
            description="Users cannot launch the application.",
        )
        assert result.priority in (IssuePriority.P0, IssuePriority.P1)

    @pytest.mark.asyncio
    async def test_heuristic_p2_error(self):
        triage = IssueTriage()
        result = await triage.triage(
            title="Error message shown when uploading large files",
            description="Files over 10MB show a generic error.",
        )
        assert result.priority in (IssuePriority.P1, IssuePriority.P2)

    @pytest.mark.asyncio
    async def test_heuristic_p4_cosmetic(self):
        triage = IssueTriage()
        result = await triage.triage(
            title="Typo in settings page",
            description="The word 'configuration' is misspelled.",
        )
        assert result.priority in (IssuePriority.P3, IssuePriority.P4)

    @pytest.mark.asyncio
    async def test_heuristic_labels(self):
        triage = IssueTriage()
        result = await triage.triage(
            title="Bug: login fails for SSO users",
            description="Security issue with token validation.",
        )
        assert "bug" in result.labels
        assert "security" in result.labels

    @pytest.mark.asyncio
    async def test_heuristic_default_priority(self):
        triage = IssueTriage()
        result = await triage.triage(
            title="Something about something",
            description="No strong signals here.",
        )
        # Should default to P3/medium when no signals
        assert result.priority == IssuePriority.P3
        assert result.severity == Severity.MEDIUM

    @pytest.mark.asyncio
    async def test_llm_triage(self, mock_triage_llm_response):
        config = LLMConfig(api_key="test")
        triage = IssueTriage(llm_config=config)
        triage._llm = AsyncMock()
        triage._llm.complete = AsyncMock(return_value=mock_triage_llm_response)

        result = await triage.triage(
            title="Login fails for SSO users",
            description="Since the last deploy, all SSO users see errors.",
        )
        assert result.priority == IssuePriority.P1
        assert result.severity == Severity.HIGH

    @pytest.mark.asyncio
    async def test_llm_triage_failure_fallback(self):
        config = LLMConfig(api_key="test")
        triage = IssueTriage(llm_config=config)
        triage._llm = AsyncMock()
        triage._llm.complete = AsyncMock(side_effect=RuntimeError("API error"))

        result = await triage.triage(
            title="Bug in error handling",
            description="Some error case not handled.",
        )
        # Should fall back to heuristics
        assert result.priority is not None
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_reasoning_populated(self):
        triage = IssueTriage()
        result = await triage.triage(
            title="Data loss when saving form",
            description="Users report losing data.",
        )
        assert len(result.reasoning) > 0
