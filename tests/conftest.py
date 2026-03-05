"""Shared pytest fixtures and mock data for devx-ai tests.

All fixtures that need to be shared across test modules live here.
LLM responses are mocked so tests run without API keys.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from devx.core.config import GitHubConfig, LLMConfig, Settings
from devx.core.llm_client import LLMClient, LLMResponse
from devx.core.models import (
    Category,
    CodeLocation,
    DeploymentRecord,
    ReviewFinding,
    Severity,
)

# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_config() -> LLMConfig:
    """LLM config with test defaults."""
    return LLMConfig(provider="openai", model="gpt-4o", api_key="test-key")


@pytest.fixture
def github_config() -> GitHubConfig:
    """GitHub config with test defaults."""
    return GitHubConfig(token="ghp_test_token")


@pytest.fixture
def settings() -> Settings:
    """Full settings with test defaults."""
    return Settings.default()


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """A generic successful LLM response."""
    return LLMResponse(
        content="Test response content",
        model="gpt-4o",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )


@pytest.fixture
def mock_review_llm_response() -> LLMResponse:
    """LLM response simulating code review findings."""
    findings = [
        {
            "title": "Potential null pointer dereference",
            "description": "The variable `user` could be None when accessed on line 15.",
            "severity": "high",
            "category": "bug",
            "file": "src/auth.py",
            "start_line": 15,
            "suggestion": "Add a None check before accessing user.name",
        },
        {
            "title": "SQL injection vulnerability",
            "description": "User input is interpolated directly into SQL query.",
            "severity": "critical",
            "category": "security",
            "file": "src/db.py",
            "start_line": 42,
            "suggestion": "Use parameterized queries instead of string formatting.",
        },
    ]
    return LLMResponse(
        content=json.dumps(findings),
        model="gpt-4o",
        usage={"prompt_tokens": 500, "completion_tokens": 200},
    )


@pytest.fixture
def mock_testgen_llm_response() -> LLMResponse:
    """LLM response simulating test generation."""
    tests = [
        {
            "name": "test_add_returns_sum",
            "description": "Test basic addition",
            "code": (
                "def test_add_returns_sum():\n"
                '    """Test that add returns correct sum."""\n'
                "    assert add(2, 3) == 5\n"
            ),
            "target_function": "add",
            "category": "unit",
        },
        {
            "name": "test_add_handles_negative",
            "description": "Test addition with negative numbers",
            "code": (
                "def test_add_handles_negative():\n"
                '    """Test add with negative numbers."""\n'
                "    assert add(-1, 1) == 0\n"
            ),
            "target_function": "add",
            "category": "edge_case",
        },
    ]
    return LLMResponse(
        content=json.dumps(tests),
        model="gpt-4o",
        usage={"prompt_tokens": 300, "completion_tokens": 150},
    )


@pytest.fixture
def mock_label_llm_response() -> LLMResponse:
    """LLM response simulating PR label classification."""
    result = {
        "labels": ["bug-fix", "test"],
        "confidence": 0.92,
        "reasoning": "The PR fixes a null pointer bug and adds tests for the fix.",
    }
    return LLMResponse(
        content=json.dumps(result),
        model="gpt-4o",
        usage={"prompt_tokens": 200, "completion_tokens": 50},
    )


@pytest.fixture
def mock_triage_llm_response() -> LLMResponse:
    """LLM response simulating issue triage."""
    result = {
        "priority": "P1",
        "severity": "high",
        "labels": ["bug", "auth"],
        "reasoning": "Login failures affect all SSO users - high impact.",
    }
    return LLMResponse(
        content=json.dumps(result),
        model="gpt-4o",
        usage={"prompt_tokens": 150, "completion_tokens": 60},
    )


@pytest.fixture
def mock_llm_client(mock_llm_response: LLMResponse) -> LLMClient:
    """LLMClient with mocked complete method."""
    client = LLMClient(LLMConfig(api_key="test"))
    client.complete = AsyncMock(return_value=mock_llm_response)  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_diff() -> str:
    """A realistic unified diff for testing."""
    return """\
diff --git a/src/auth.py b/src/auth.py
index abc1234..def5678 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,15 @@ class AuthService:
     def __init__(self, db):
         self.db = db

+    def authenticate(self, username, password):
+        user = self.db.find_user(username)
+        if user and user.check_password(password):
+            return self._create_token(user)
+        return None
+
+    def _create_token(self, user):
+        return {"token": "abc", "user_id": user.id}
+
     def logout(self, token):
         self.db.invalidate_token(token)

diff --git a/tests/test_auth.py b/tests/test_auth.py
new file mode 100644
--- /dev/null
+++ b/tests/test_auth.py
@@ -0,0 +1,12 @@
+import pytest
+from src.auth import AuthService
+
+
+def test_authenticate_valid_user():
+    db = MockDB()
+    auth = AuthService(db)
+    result = auth.authenticate("admin", "password")
+    assert result is not None
+
+def test_authenticate_invalid_user():
+    db = MockDB()
"""


@pytest.fixture
def sample_python_source() -> str:
    """Sample Python source for AST analysis and test generation."""
    return '''\
"""Math utilities module."""


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        Sum of a and b.
    """
    return a + b


def divide(numerator: float, denominator: float) -> float:
    """Divide two numbers.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Result of division.

    Raises:
        ZeroDivisionError: If denominator is zero.
    """
    if denominator == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return numerator / denominator


async def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetch data from a URL.

    Args:
        url: Target URL.
        timeout: Request timeout in seconds.

    Returns:
        Response data as dictionary.
    """
    return {"url": url, "data": "mock"}
'''


@pytest.fixture
def sample_deployments() -> list[DeploymentRecord]:
    """Sample deployment records for DORA metrics testing."""
    now = datetime.now(tz=UTC)
    return [
        DeploymentRecord(
            id="dep-1",
            repo="org/app",
            sha="aaa111",
            deployed_at=now - timedelta(days=1),
            status="success",
            lead_time_seconds=3600,  # 1 hour
        ),
        DeploymentRecord(
            id="dep-2",
            repo="org/app",
            sha="bbb222",
            deployed_at=now - timedelta(days=3),
            status="success",
            lead_time_seconds=7200,  # 2 hours
        ),
        DeploymentRecord(
            id="dep-3",
            repo="org/app",
            sha="ccc333",
            deployed_at=now - timedelta(days=5),
            status="failure",
            lead_time_seconds=1800,
        ),
        DeploymentRecord(
            id="dep-4",
            repo="org/app",
            sha="ddd444",
            deployed_at=now - timedelta(days=4),
            status="success",
            lead_time_seconds=5400,  # 1.5 hours
        ),
        DeploymentRecord(
            id="dep-5",
            repo="org/app",
            sha="eee555",
            deployed_at=now - timedelta(days=7),
            status="success",
            lead_time_seconds=10800,  # 3 hours
        ),
    ]


@pytest.fixture
def sample_finding() -> ReviewFinding:
    """A single review finding for testing formatters."""
    return ReviewFinding(
        title="Function too complex",
        description="Cyclomatic complexity of 15 exceeds threshold of 10.",
        severity=Severity.MEDIUM,
        category=Category.COMPLEXITY,
        location=CodeLocation(file="src/utils.py", start_line=42, end_line=90),
        suggestion="Extract helper functions to reduce complexity.",
    )
