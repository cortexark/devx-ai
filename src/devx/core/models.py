"""Pydantic v2 domain models used across the devx platform.

Every data boundary in devx-ai is modeled here so that validation,
serialization, and documentation stay in one place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    """Severity level for review findings and issue triage."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(StrEnum):
    """Categories for code review findings."""

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    COMPLEXITY = "complexity"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


class PRLabel(StrEnum):
    """Standard labels for pull request classification."""

    BUG_FIX = "bug-fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    CHORE = "chore"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BREAKING = "breaking-change"


class IssuePriority(StrEnum):
    """Priority levels for triaged issues."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


# ---------------------------------------------------------------------------
# Code Review
# ---------------------------------------------------------------------------


class CodeLocation(BaseModel):
    """A precise location in source code."""

    file: str = Field(description="Relative file path")
    start_line: int = Field(ge=1, description="Starting line number (1-indexed)")
    end_line: int | None = Field(default=None, ge=1, description="Ending line number, inclusive")

    def __str__(self) -> str:
        if self.end_line and self.end_line != self.start_line:
            return f"{self.file}:{self.start_line}-{self.end_line}"
        return f"{self.file}:{self.start_line}"


class ReviewFinding(BaseModel):
    """A single finding from the code review agent."""

    title: str = Field(max_length=200, description="Short description of the finding")
    description: str = Field(description="Detailed explanation")
    severity: Severity
    category: Category
    location: CodeLocation
    suggestion: str | None = Field(
        default=None, description="Suggested fix (code snippet or prose)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.8, description="Model confidence in this finding"
    )


class ReviewResult(BaseModel):
    """Aggregated result of a code review."""

    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = Field(default="", description="High-level summary of the review")
    files_analyzed: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Diff Parsing
# ---------------------------------------------------------------------------


class DiffHunk(BaseModel):
    """A single hunk within a file diff."""

    old_start: int = Field(ge=0)
    old_count: int = Field(ge=0)
    new_start: int = Field(ge=0)
    new_count: int = Field(ge=0)
    header: str = Field(default="", description="@@ header line")
    content: str = Field(description="Raw hunk content including +/- lines")

    @property
    def added_lines(self) -> list[str]:
        """Return only the added lines (without the '+' prefix)."""
        return [line[1:] for line in self.content.splitlines() if line.startswith("+")]

    @property
    def removed_lines(self) -> list[str]:
        """Return only the removed lines (without the '-' prefix)."""
        return [line[1:] for line in self.content.splitlines() if line.startswith("-")]


class FileDiff(BaseModel):
    """Parsed diff for a single file."""

    old_path: str | None = Field(default=None, description="Path before rename/delete")
    new_path: str | None = Field(default=None, description="Path after rename/add")
    hunks: list[DiffHunk] = Field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False

    @property
    def path(self) -> str:
        """Canonical path for this file (prefers new_path)."""
        return self.new_path or self.old_path or "<unknown>"

    @property
    def total_additions(self) -> int:
        """Total number of added lines across all hunks."""
        return sum(len(h.added_lines) for h in self.hunks)

    @property
    def total_deletions(self) -> int:
        """Total number of deleted lines across all hunks."""
        return sum(len(h.removed_lines) for h in self.hunks)


# ---------------------------------------------------------------------------
# Test Generation
# ---------------------------------------------------------------------------


class FunctionSignature(BaseModel):
    """Extracted metadata about a Python function."""

    name: str
    module: str = ""
    docstring: str | None = None
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    return_type: str | None = None
    decorators: list[str] = Field(default_factory=list)
    is_async: bool = False
    source: str = Field(default="", description="Raw source code of the function")


class TestCase(BaseModel):
    """A generated test case."""

    name: str = Field(description="Test function name, e.g. test_add_returns_sum")
    description: str = Field(default="", description="What this test verifies")
    code: str = Field(description="Complete test function source")
    target_function: str = Field(description="Name of the function under test")
    category: str = Field(default="unit", description="unit | integration | edge_case")


class TestSuite(BaseModel):
    """Collection of generated test cases for a module."""

    module: str = Field(description="Module path being tested")
    imports: list[str] = Field(default_factory=list)
    test_cases: list[TestCase] = Field(default_factory=list)
    framework: str = Field(default="pytest")


# ---------------------------------------------------------------------------
# SDLC Automation
# ---------------------------------------------------------------------------


class LabelClassification(BaseModel):
    """Result of PR label classification."""

    labels: list[PRLabel] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Why these labels were chosen")


class TriageResult(BaseModel):
    """Result of issue triage."""

    priority: IssuePriority
    severity: Severity
    labels: list[str] = Field(default_factory=list)
    assignee_suggestion: str | None = None
    reasoning: str = Field(default="")


# ---------------------------------------------------------------------------
# Engineering Metrics
# ---------------------------------------------------------------------------


class DeploymentRecord(BaseModel):
    """A single deployment event."""

    id: str
    repo: str
    environment: str = "production"
    sha: str
    deployed_at: datetime
    status: str = Field(default="success", description="success | failure | rollback")
    lead_time_seconds: float | None = Field(
        default=None, description="Seconds from first commit to deploy"
    )


class DORAMetrics(BaseModel):
    """DORA metrics snapshot for a team or repository."""

    deployment_frequency: float = Field(description="Deployments per day (averaged over window)")
    lead_time_seconds: float = Field(description="Median seconds from commit to production")
    change_failure_rate: float = Field(
        ge=0.0, le=1.0, description="Fraction of deployments causing failure"
    )
    mttr_seconds: float = Field(description="Mean time to recovery in seconds")
    window_days: int = Field(default=30, description="Measurement window in days")
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def deployment_frequency_rating(self) -> str:
        """Classify deployment frequency per DORA benchmarks."""
        if self.deployment_frequency >= 1.0:
            return "elite"
        if self.deployment_frequency >= 1 / 7:
            return "high"
        if self.deployment_frequency >= 1 / 30:
            return "medium"
        return "low"

    @property
    def lead_time_rating(self) -> str:
        """Classify lead time per DORA benchmarks."""
        if self.lead_time_seconds < 86400:  # < 1 day
            return "elite"
        if self.lead_time_seconds < 604800:  # < 1 week
            return "high"
        if self.lead_time_seconds < 2592000:  # < 1 month
            return "medium"
        return "low"
