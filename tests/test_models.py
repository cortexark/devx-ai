"""Tests for Pydantic models in devx.core.models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from devx.core.models import (
    Category,
    CodeLocation,
    DeploymentRecord,
    DiffHunk,
    DORAMetrics,
    FileDiff,
    FunctionSignature,
    IssuePriority,
    LabelClassification,
    PRLabel,
    ReviewFinding,
    ReviewResult,
    Severity,
    TestCase,
    TestSuite,
    TriageResult,
)

# ---------------------------------------------------------------------------
# CodeLocation
# ---------------------------------------------------------------------------


class TestCodeLocation:
    def test_str_single_line(self):
        loc = CodeLocation(file="src/main.py", start_line=10)
        assert str(loc) == "src/main.py:10"

    def test_str_line_range(self):
        loc = CodeLocation(file="src/main.py", start_line=10, end_line=20)
        assert str(loc) == "src/main.py:10-20"

    def test_str_same_start_end(self):
        loc = CodeLocation(file="src/main.py", start_line=10, end_line=10)
        assert str(loc) == "src/main.py:10"

    def test_invalid_line_number(self):
        with pytest.raises(ValidationError):
            CodeLocation(file="src/main.py", start_line=0)

    def test_negative_line_number(self):
        with pytest.raises(ValidationError):
            CodeLocation(file="src/main.py", start_line=-1)


# ---------------------------------------------------------------------------
# Severity and Category enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_severity_values(self):
        assert Severity.CRITICAL == "critical"
        assert Severity.HIGH == "high"
        assert Severity.MEDIUM == "medium"
        assert Severity.LOW == "low"
        assert Severity.INFO == "info"

    def test_category_values(self):
        assert Category.BUG == "bug"
        assert Category.SECURITY == "security"
        assert Category.PERFORMANCE == "performance"

    def test_pr_label_values(self):
        assert PRLabel.BUG_FIX == "bug-fix"
        assert PRLabel.FEATURE == "feature"
        assert PRLabel.REFACTOR == "refactor"

    def test_issue_priority_values(self):
        assert IssuePriority.P0 == "P0"
        assert IssuePriority.P4 == "P4"


# ---------------------------------------------------------------------------
# ReviewFinding
# ---------------------------------------------------------------------------


class TestReviewFinding:
    def test_create_valid_finding(self, sample_finding):
        assert sample_finding.title == "Function too complex"
        assert sample_finding.severity == Severity.MEDIUM
        assert sample_finding.category == Category.COMPLEXITY

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="Test",
                description="Test",
                severity=Severity.LOW,
                category=Category.STYLE,
                location=CodeLocation(file="f.py", start_line=1),
                confidence=1.5,
            )

    def test_title_max_length(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="x" * 201,
                description="Test",
                severity=Severity.LOW,
                category=Category.STYLE,
                location=CodeLocation(file="f.py", start_line=1),
            )

    def test_default_confidence(self):
        finding = ReviewFinding(
            title="Test",
            description="Test",
            severity=Severity.LOW,
            category=Category.STYLE,
            location=CodeLocation(file="f.py", start_line=1),
        )
        assert finding.confidence == 0.8


# ---------------------------------------------------------------------------
# DiffHunk
# ---------------------------------------------------------------------------


class TestDiffHunk:
    def test_added_lines(self):
        hunk = DiffHunk(
            old_start=1, old_count=3, new_start=1, new_count=5,
            content="+line1\n line2\n-line3\n+line4\n+line5",
        )
        assert hunk.added_lines == ["line1", "line4", "line5"]

    def test_removed_lines(self):
        hunk = DiffHunk(
            old_start=1, old_count=3, new_start=1, new_count=5,
            content="+line1\n line2\n-line3\n+line4\n-line5",
        )
        assert hunk.removed_lines == ["line3", "line5"]

    def test_empty_content(self):
        hunk = DiffHunk(
            old_start=1, old_count=0, new_start=1, new_count=0,
            content="",
        )
        assert hunk.added_lines == []
        assert hunk.removed_lines == []


# ---------------------------------------------------------------------------
# FileDiff
# ---------------------------------------------------------------------------


class TestFileDiff:
    def test_path_prefers_new(self):
        fd = FileDiff(old_path="old.py", new_path="new.py")
        assert fd.path == "new.py"

    def test_path_fallback_to_old(self):
        fd = FileDiff(old_path="old.py", is_deleted=True)
        assert fd.path == "old.py"

    def test_path_unknown(self):
        fd = FileDiff()
        assert fd.path == "<unknown>"

    def test_total_additions(self):
        fd = FileDiff(
            new_path="test.py",
            hunks=[
                DiffHunk(old_start=1, old_count=0, new_start=1, new_count=2,
                         content="+line1\n+line2"),
                DiffHunk(old_start=5, old_count=0, new_start=5, new_count=1,
                         content="+line3"),
            ],
        )
        assert fd.total_additions == 3

    def test_total_deletions(self):
        fd = FileDiff(
            old_path="test.py",
            hunks=[
                DiffHunk(old_start=1, old_count=2, new_start=1, new_count=0,
                         content="-line1\n-line2"),
            ],
        )
        assert fd.total_deletions == 2


# ---------------------------------------------------------------------------
# ReviewResult
# ---------------------------------------------------------------------------


class TestReviewResult:
    def test_empty_result(self):
        result = ReviewResult()
        assert result.findings == []
        assert result.files_analyzed == 0

    def test_with_findings(self, sample_finding):
        result = ReviewResult(
            findings=[sample_finding],
            summary="Found 1 issue",
            files_analyzed=3,
            duration_seconds=1.5,
        )
        assert len(result.findings) == 1
        assert result.files_analyzed == 3


# ---------------------------------------------------------------------------
# FunctionSignature
# ---------------------------------------------------------------------------


class TestFunctionSignature:
    def test_basic_signature(self):
        sig = FunctionSignature(
            name="add",
            parameters=[
                {"name": "a", "type": "int"},
                {"name": "b", "type": "int"},
            ],
            return_type="int",
        )
        assert sig.name == "add"
        assert len(sig.parameters) == 2
        assert sig.is_async is False

    def test_async_signature(self):
        sig = FunctionSignature(name="fetch", is_async=True)
        assert sig.is_async is True


# ---------------------------------------------------------------------------
# TestCase and TestSuite
# ---------------------------------------------------------------------------


class TestTestModels:
    def test_create_test_case(self):
        tc = TestCase(
            name="test_add",
            code="def test_add(): assert add(1, 2) == 3",
            target_function="add",
        )
        assert tc.category == "unit"

    def test_create_test_suite(self):
        suite = TestSuite(
            module="utils",
            test_cases=[
                TestCase(
                    name="test_add",
                    code="def test_add(): pass",
                    target_function="add",
                )
            ],
        )
        assert suite.framework == "pytest"
        assert len(suite.test_cases) == 1


# ---------------------------------------------------------------------------
# SDLC Models
# ---------------------------------------------------------------------------


class TestLabelClassification:
    def test_valid_classification(self):
        lc = LabelClassification(
            labels=[PRLabel.BUG_FIX, PRLabel.TEST],
            confidence=0.9,
            reasoning="Fixes a bug and adds tests",
        )
        assert len(lc.labels) == 2
        assert lc.confidence == 0.9


class TestTriageResult:
    def test_valid_triage(self):
        tr = TriageResult(
            priority=IssuePriority.P1,
            severity=Severity.HIGH,
            labels=["bug", "auth"],
            reasoning="Login broken for all users",
        )
        assert tr.priority == IssuePriority.P1


# ---------------------------------------------------------------------------
# DORA Metrics
# ---------------------------------------------------------------------------


class TestDORAMetrics:
    def test_deployment_frequency_rating_elite(self):
        m = DORAMetrics(
            deployment_frequency=2.0,
            lead_time_seconds=3600,
            change_failure_rate=0.05,
            mttr_seconds=1800,
        )
        assert m.deployment_frequency_rating == "elite"

    def test_deployment_frequency_rating_high(self):
        m = DORAMetrics(
            deployment_frequency=0.5,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=3600,
        )
        assert m.deployment_frequency_rating == "high"

    def test_deployment_frequency_rating_medium(self):
        m = DORAMetrics(
            deployment_frequency=0.05,
            lead_time_seconds=3600,
            change_failure_rate=0.2,
            mttr_seconds=7200,
        )
        assert m.deployment_frequency_rating == "medium"

    def test_deployment_frequency_rating_low(self):
        m = DORAMetrics(
            deployment_frequency=0.01,
            lead_time_seconds=3600,
            change_failure_rate=0.3,
            mttr_seconds=86400,
        )
        assert m.deployment_frequency_rating == "low"

    def test_lead_time_rating_elite(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,  # 1 hour
            change_failure_rate=0.05,
            mttr_seconds=1800,
        )
        assert m.lead_time_rating == "elite"

    def test_lead_time_rating_low(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=5_000_000,  # > 1 month
            change_failure_rate=0.05,
            mttr_seconds=1800,
        )
        assert m.lead_time_rating == "low"

    def test_change_failure_rate_bounds(self):
        with pytest.raises(ValidationError):
            DORAMetrics(
                deployment_frequency=1.0,
                lead_time_seconds=3600,
                change_failure_rate=1.5,  # > 1.0
                mttr_seconds=1800,
            )


class TestDeploymentRecord:
    def test_create_deployment(self):
        dep = DeploymentRecord(
            id="dep-1",
            repo="org/app",
            sha="abc123",
            deployed_at=datetime.now(tz=UTC),
        )
        assert dep.status == "success"
        assert dep.environment == "production"
