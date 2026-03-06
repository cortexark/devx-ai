"""Tests for the code review module: diff parsing, AST analysis, and agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from devx.core.config import LLMConfig
from devx.core.models import ReviewResult, Severity
from devx.review.agent import ReviewAgent
from devx.review.analyzer import ASTAnalyzer
from devx.review.diff_parser import DiffParser
from devx.review.suggestions import SuggestionFormatter

# ---------------------------------------------------------------------------
# DiffParser
# ---------------------------------------------------------------------------


class TestDiffParser:
    def test_empty_diff(self):
        parser = DiffParser()
        assert parser.parse("") == []
        assert parser.parse("   \n  ") == []

    def test_single_file_diff(self, sample_diff):
        parser = DiffParser()
        diffs = parser.parse(sample_diff)
        assert len(diffs) == 2  # auth.py and test_auth.py

    def test_file_paths(self, sample_diff):
        parser = DiffParser()
        diffs = parser.parse(sample_diff)
        paths = {d.path for d in diffs}
        assert "src/auth.py" in paths
        assert "tests/test_auth.py" in paths

    def test_new_file_detection(self, sample_diff):
        parser = DiffParser()
        diffs = parser.parse(sample_diff)
        test_file = next(d for d in diffs if "test_auth" in d.path)
        assert test_file.is_new is True

    def test_hunk_content(self, sample_diff):
        parser = DiffParser()
        diffs = parser.parse(sample_diff)
        auth_file = next(d for d in diffs if d.path == "src/auth.py")
        assert len(auth_file.hunks) == 1
        assert auth_file.total_additions > 0

    def test_additions_count(self, sample_diff):
        parser = DiffParser()
        diffs = parser.parse(sample_diff)
        auth_file = next(d for d in diffs if d.path == "src/auth.py")
        # The diff adds authenticate and _create_token methods
        assert auth_file.total_additions >= 5

    def test_rename_detection(self):
        parser = DiffParser()
        diff = """\
diff --git a/old_name.py b/new_name.py
similarity index 95%
rename from old_name.py
rename to new_name.py
--- a/old_name.py
+++ b/new_name.py
@@ -1,3 +1,3 @@
-old content
+new content
"""
        diffs = parser.parse(diff)
        assert len(diffs) == 1
        assert diffs[0].is_rename is True
        assert diffs[0].old_path == "old_name.py"
        assert diffs[0].new_path == "new_name.py"

    def test_deleted_file(self):
        parser = DiffParser()
        diff = """\
diff --git a/removed.py b/removed.py
deleted file mode 100644
--- a/removed.py
+++ /dev/null
@@ -1,5 +0,0 @@
-line1
-line2
-line3
-line4
-line5
"""
        diffs = parser.parse(diff)
        assert len(diffs) == 1
        assert diffs[0].is_deleted is True
        assert diffs[0].total_deletions == 5

    def test_multiple_hunks(self):
        parser = DiffParser()
        diff = """\
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line1
+added_line1
 line2
 line3
@@ -10,3 +11,4 @@
 line10
+added_line2
 line11
 line12
"""
        diffs = parser.parse(diff)
        assert len(diffs) == 1
        assert len(diffs[0].hunks) == 2


# ---------------------------------------------------------------------------
# ASTAnalyzer
# ---------------------------------------------------------------------------


class TestASTAnalyzer:
    def test_analyze_python_functions(self, sample_python_source):
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(sample_python_source, "utils.py")
        func_names = [f.name for f in result.functions]
        assert "add" in func_names
        assert "divide" in func_names

    def test_analyze_detects_missing_docstring(self):
        source = """\
def undocumented_function(x):
    return x * 2
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "test.py")
        doc_findings = [f for f in result.findings if "docstring" in f.title.lower()]
        assert len(doc_findings) >= 1

    def test_analyze_long_function(self):
        # Create a function with > 50 lines
        lines = ["def long_function():"]
        lines.append('    """Does many things."""')
        for i in range(60):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_0")
        source = "\n".join(lines)

        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "test.py")
        complexity_findings = [
            f for f in result.findings if "long" in f.title.lower() or "too" in f.title.lower()
        ]
        assert len(complexity_findings) >= 1

    def test_analyze_too_many_parameters(self):
        source = """\
def many_params(a, b, c, d, e, f, g):
    \"\"\"Too many params.\"\"\"
    return a + b + c + d + e + f + g
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "test.py")
        param_findings = [f for f in result.findings if "parameter" in f.title.lower()]
        assert len(param_findings) >= 1

    def test_analyze_empty_source(self):
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python("", "empty.py")
        assert result.functions == []
        assert result.findings == []

    def test_analyze_class_detection(self):
        source = """\
class MyService:
    \"\"\"A service class.\"\"\"

    def method_one(self):
        pass

    def method_two(self):
        pass
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "service.py")
        assert len(result.classes) >= 1
        assert result.classes[0].name == "MyService"


# ---------------------------------------------------------------------------
# ReviewAgent
# ---------------------------------------------------------------------------


class TestReviewAgent:
    @pytest.mark.asyncio
    async def test_review_empty_diff(self):
        agent = ReviewAgent(enable_llm=False)
        result = await agent.review_diff("")
        assert isinstance(result, ReviewResult)
        assert result.findings == []

    @pytest.mark.asyncio
    async def test_review_diff_ast_only(self, sample_diff):
        agent = ReviewAgent(enable_llm=False)
        result = await agent.review_diff(sample_diff)
        assert isinstance(result, ReviewResult)
        assert result.files_analyzed >= 1

    @pytest.mark.asyncio
    async def test_review_file(self, sample_python_source):
        agent = ReviewAgent(enable_llm=False)
        result = await agent.review_file(sample_python_source, "utils.py")
        assert isinstance(result, ReviewResult)
        assert result.files_analyzed == 1

    @pytest.mark.asyncio
    async def test_review_with_mock_llm(self, sample_diff, mock_review_llm_response):
        config = LLMConfig(api_key="test")
        agent = ReviewAgent(llm_config=config, enable_llm=True)
        agent._llm = AsyncMock()
        agent._llm.complete = AsyncMock(return_value=mock_review_llm_response)

        result = await agent.review_diff(sample_diff)
        assert isinstance(result, ReviewResult)
        assert len(result.findings) >= 1

    @pytest.mark.asyncio
    async def test_review_deduplicates_findings(self, sample_diff, mock_review_llm_response):
        config = LLMConfig(api_key="test")
        agent = ReviewAgent(llm_config=config, enable_llm=True)
        agent._llm = AsyncMock()
        agent._llm.complete = AsyncMock(return_value=mock_review_llm_response)

        result = await agent.review_diff(sample_diff)
        # Findings should be sorted by severity
        if len(result.findings) >= 2:
            severity_order = list(Severity)
            for i in range(len(result.findings) - 1):
                idx_a = severity_order.index(result.findings[i].severity)
                idx_b = severity_order.index(result.findings[i + 1].severity)
                assert idx_a <= idx_b

    @pytest.mark.asyncio
    async def test_review_handles_llm_failure(self, sample_diff):
        config = LLMConfig(api_key="test")
        agent = ReviewAgent(llm_config=config, enable_llm=True)
        agent._llm = AsyncMock()
        agent._llm.complete = AsyncMock(side_effect=RuntimeError("API error"))

        # Should not raise, falls back to AST-only
        result = await agent.review_diff(sample_diff)
        assert isinstance(result, ReviewResult)


# ---------------------------------------------------------------------------
# SuggestionFormatter
# ---------------------------------------------------------------------------


class TestSuggestionFormatter:
    def test_github_comment_with_findings(self, sample_finding):
        result = ReviewResult(
            findings=[sample_finding],
            summary="Found 1 issue",
            files_analyzed=1,
            duration_seconds=0.5,
        )
        formatter = SuggestionFormatter()
        comment = formatter.to_github_comment(result)
        assert "Code Review Summary" in comment
        assert "Function too complex" in comment

    def test_github_comment_empty(self):
        result = ReviewResult(files_analyzed=3, duration_seconds=0.2)
        formatter = SuggestionFormatter()
        comment = formatter.to_github_comment(result)
        assert "No issues found" in comment

    def test_inline_comments(self, sample_finding):
        result = ReviewResult(findings=[sample_finding])
        formatter = SuggestionFormatter()
        comments = formatter.to_inline_comments(result)
        assert len(comments) == 1
        assert comments[0]["path"] == "src/utils.py"
        assert comments[0]["line"] == 42

    def test_to_json(self, sample_finding):
        result = ReviewResult(
            findings=[sample_finding],
            summary="Test",
            files_analyzed=1,
        )
        formatter = SuggestionFormatter()
        data = formatter.to_json(result)
        assert "findings" in data
        assert len(data["findings"]) == 1
