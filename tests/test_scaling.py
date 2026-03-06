"""Scaling Tests for devx-ai.

Tests that core components handle large inputs gracefully: large diffs,
large source files, many deployment records, many file paths.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from devx.core.models import DeploymentRecord, DORAMetrics, LabelClassification, PRLabel, ReviewResult
from devx.metrics.analyzer import DORAAnalyzer
from devx.metrics.dashboard import MetricsStore
from devx.review.analyzer import ASTAnalyzer
from devx.review.diff_parser import DiffParser
from devx.review.agent import ReviewAgent
from devx.review.suggestions import SuggestionFormatter
from devx.sdlc.labeler import PRLabeler
from devx.testgen.extractor import SignatureExtractor
from devx.testgen.generator import TestGenerator


# ===========================================================================
# 1. DiffParser with Very Large Diffs (1000+ lines)
# ===========================================================================


class TestDiffParserScaling:
    """DiffParser handles very large diffs without failure."""

    def _generate_large_diff(self, num_added_lines: int) -> str:
        """Generate a synthetic unified diff with many added lines."""
        lines = [
            "diff --git a/large_file.py b/large_file.py",
            "--- a/large_file.py",
            "+++ b/large_file.py",
            "@@ -1,0 +1,{} @@".format(num_added_lines),
        ]
        for i in range(num_added_lines):
            lines.append("+    x_{} = {}  # added line {}".format(i, i, i))
        return "\n".join(lines)

    def test_parse_1000_line_diff(self):
        parser = DiffParser()
        diff = self._generate_large_diff(1000)
        result = parser.parse(diff)
        assert len(result) == 1
        assert result[0].total_additions == 1000

    def test_parse_5000_line_diff(self):
        parser = DiffParser()
        diff = self._generate_large_diff(5000)
        result = parser.parse(diff)
        assert len(result) == 1
        assert result[0].total_additions == 5000

    def test_parse_10000_line_diff(self):
        parser = DiffParser()
        diff = self._generate_large_diff(10000)
        start = time.monotonic()
        result = parser.parse(diff)
        elapsed = time.monotonic() - start
        assert len(result) == 1
        assert result[0].total_additions == 10000
        # Should complete within 5 seconds
        assert elapsed < 5.0, "Parsing 10k-line diff took {:.2f}s".format(elapsed)

    def test_many_files_in_diff(self):
        """Diff with 100 files."""
        parser = DiffParser()
        parts = []
        for i in range(100):
            parts.append("diff --git a/file_{}.py b/file_{}.py".format(i, i))
            parts.append("--- a/file_{}.py".format(i))
            parts.append("+++ b/file_{}.py".format(i))
            parts.append("@@ -1,1 +1,2 @@")
            parts.append(" existing line")
            parts.append("+new line in file {}".format(i))
        diff = "\n".join(parts)
        result = parser.parse(diff)
        assert len(result) == 100

    def test_multiple_hunks_per_file(self):
        """Single file with 50 hunks."""
        parser = DiffParser()
        parts = [
            "diff --git a/multi_hunk.py b/multi_hunk.py",
            "--- a/multi_hunk.py",
            "+++ b/multi_hunk.py",
        ]
        for i in range(50):
            start = i * 20 + 1
            parts.append("@@ -{},3 +{},4 @@".format(start, start))
            parts.append(" context_line_{}".format(i))
            parts.append("+added_line_{}".format(i))
            parts.append(" another_context_{}".format(i))
        diff = "\n".join(parts)
        result = parser.parse(diff)
        assert len(result) == 1
        assert len(result[0].hunks) == 50

    def test_diff_with_large_context(self):
        """Diff with many context lines between hunks."""
        parser = DiffParser()
        parts = [
            "diff --git a/ctx.py b/ctx.py",
            "--- a/ctx.py",
            "+++ b/ctx.py",
            "@@ -1,500 +1,501 @@",
        ]
        for i in range(500):
            parts.append(" context_line_{}".format(i))
        parts.append("+added_line")
        diff = "\n".join(parts)
        result = parser.parse(diff)
        assert len(result) == 1
        assert result[0].total_additions == 1


# ===========================================================================
# 2. ASTAnalyzer with Large Python Files (100+ functions)
# ===========================================================================


class TestASTAnalyzerScaling:
    """ASTAnalyzer handles large Python files."""

    def _generate_large_module(self, num_functions: int) -> str:
        """Generate a Python module with many functions."""
        lines = ['"""Large module."""\n']
        for i in range(num_functions):
            lines.append(
                "def function_{}(x: int) -> int:\n"
                '    """Function {}."""\n'
                "    return x + {}\n\n".format(i, i, i)
            )
        return "\n".join(lines)

    def test_100_functions(self):
        analyzer = ASTAnalyzer()
        source = self._generate_large_module(100)
        result = analyzer.analyze_python(source, "large.py")
        assert len(result.functions) >= 100

    def test_200_functions(self):
        analyzer = ASTAnalyzer()
        source = self._generate_large_module(200)
        result = analyzer.analyze_python(source, "large.py")
        assert len(result.functions) >= 200

    def test_500_functions_performance(self):
        analyzer = ASTAnalyzer()
        source = self._generate_large_module(500)
        start = time.monotonic()
        result = analyzer.analyze_python(source, "large.py")
        elapsed = time.monotonic() - start
        assert len(result.functions) >= 500
        # Should complete within 5 seconds
        assert elapsed < 5.0, "Analyzing 500 functions took {:.2f}s".format(elapsed)

    def test_large_file_with_classes(self):
        """Module with 50 classes, each with 5 methods."""
        lines = ['"""Module with many classes."""\n']
        for i in range(50):
            lines.append("class Class_{}:".format(i))
            lines.append('    """Class {}."""'.format(i))
            for j in range(5):
                lines.append("    def method_{}(self):".format(j))
                lines.append('        """Method {}."""'.format(j))
                lines.append("        return {}".format(j))
            lines.append("")
        source = "\n".join(lines)
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "classes.py")
        assert len(result.classes) >= 50

    def test_functions_with_many_parameters(self):
        """Functions that each have 10+ parameters."""
        lines = ['"""Module with complex signatures."""\n']
        for i in range(50):
            params = ", ".join("p{}: int = {}".format(j, j) for j in range(10))
            lines.append("def complex_func_{}({}):".format(i, params))
            lines.append('    """Complex function {}."""'.format(i))
            lines.append("    return p0 + p1")
            lines.append("")
        source = "\n".join(lines)
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "complex.py")
        assert len(result.functions) >= 50
        # Each function should generate a "too many parameters" finding
        param_findings = [
            f for f in result.findings if "parameter" in f.title.lower()
        ]
        assert len(param_findings) >= 50

    def test_deeply_nested_function_bodies(self):
        """Functions with deeply nested control flow."""
        lines = ['"""Deeply nested module."""\n']
        for i in range(20):
            lines.append("def nested_func_{}():".format(i))
            lines.append('    """Nested function {}."""'.format(i))
            indent = "    "
            for d in range(8):
                lines.append("{}{}if True:".format(indent, "    " * d))
            lines.append("{}{}pass".format(indent, "    " * 8))
            lines.append("")
        source = "\n".join(lines)
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "nested.py")
        assert len(result.functions) >= 20

    def test_long_function_detection_at_scale(self):
        """Multiple long functions should all be detected."""
        lines = ['"""Module with long functions."""\n']
        for i in range(10):
            lines.append("def long_func_{}():".format(i))
            lines.append('    """Long function {}."""'.format(i))
            for j in range(60):
                lines.append("    x_{} = {}".format(j, j))
            lines.append("    return x_0")
            lines.append("")
        source = "\n".join(lines)
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "long.py")
        long_findings = [
            f for f in result.findings if "too long" in f.title.lower()
        ]
        assert len(long_findings) >= 10


# ===========================================================================
# 3. DORAAnalyzer with 1000+ Deployment Records
# ===========================================================================


class TestDORAAnalyzerScaling:
    """DORAAnalyzer handles large deployment datasets."""

    def _generate_deployments(
        self, count: int, failure_rate: float = 0.1
    ) -> list[DeploymentRecord]:
        """Generate a list of deployment records."""
        now = datetime.now(tz=UTC)
        deployments = []
        failure_interval = max(int(1 / failure_rate), 1)
        for i in range(count):
            status = "failure" if (i % failure_interval == 0) else "success"
            deployments.append(
                DeploymentRecord(
                    id="dep-{}".format(i),
                    repo="org/app",
                    sha="sha{:04x}".format(i),
                    deployed_at=now - timedelta(hours=i),
                    status=status,
                    lead_time_seconds=3600 + (i * 10),
                )
            )
        return deployments

    def test_1000_deployments(self):
        analyzer = DORAAnalyzer()
        deployments = self._generate_deployments(1000)
        metrics = analyzer.calculate(deployments, window_days=30)
        assert metrics.deployment_frequency > 0
        assert metrics.lead_time_seconds > 0
        assert 0.0 <= metrics.change_failure_rate <= 1.0

    def test_5000_deployments(self):
        analyzer = DORAAnalyzer()
        deployments = self._generate_deployments(5000)
        metrics = analyzer.calculate(deployments, window_days=30)
        assert isinstance(metrics, DORAMetrics)
        assert metrics.deployment_frequency > 0

    def test_10000_deployments_performance(self):
        analyzer = DORAAnalyzer()
        deployments = self._generate_deployments(10000)
        start = time.monotonic()
        metrics = analyzer.calculate(deployments, window_days=30)
        elapsed = time.monotonic() - start
        assert isinstance(metrics, DORAMetrics)
        # Should complete within 5 seconds
        assert elapsed < 5.0, "Calculating DORA for 10k deploys took {:.2f}s".format(elapsed)

    def test_all_failures(self):
        """All deployments are failures."""
        now = datetime.now(tz=UTC)
        deployments = [
            DeploymentRecord(
                id="d{}".format(i), repo="r", sha="s{}".format(i),
                deployed_at=now - timedelta(hours=i),
                status="failure",
                lead_time_seconds=3600,
            )
            for i in range(100)
        ]
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(deployments, window_days=30)
        assert metrics.change_failure_rate == 1.0
        assert metrics.deployment_frequency == 0.0  # No successful deploys

    def test_all_successes(self):
        """All deployments are successful."""
        now = datetime.now(tz=UTC)
        deployments = [
            DeploymentRecord(
                id="d{}".format(i), repo="r", sha="s{}".format(i),
                deployed_at=now - timedelta(hours=i),
                status="success",
                lead_time_seconds=3600,
            )
            for i in range(100)
        ]
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(deployments, window_days=30)
        assert metrics.change_failure_rate == 0.0
        assert metrics.deployment_frequency > 0

    def test_no_lead_times(self):
        """Deployments with no lead_time_seconds."""
        now = datetime.now(tz=UTC)
        deployments = [
            DeploymentRecord(
                id="d{}".format(i), repo="r", sha="s{}".format(i),
                deployed_at=now - timedelta(hours=i),
                status="success",
                lead_time_seconds=None,
            )
            for i in range(50)
        ]
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(deployments, window_days=30)
        assert metrics.lead_time_seconds == 0.0

    def test_mixed_lead_times(self):
        """Some deployments have lead time, some do not."""
        now = datetime.now(tz=UTC)
        deployments = [
            DeploymentRecord(
                id="d{}".format(i), repo="r", sha="s{}".format(i),
                deployed_at=now - timedelta(hours=i),
                status="success",
                lead_time_seconds=3600 if i % 2 == 0 else None,
            )
            for i in range(100)
        ]
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(deployments, window_days=30)
        assert metrics.lead_time_seconds == 3600.0  # median of all 3600s

    def test_trend_with_large_numbers(self):
        """Trend calculation with extreme values."""
        analyzer = DORAAnalyzer()
        current = DORAMetrics(
            deployment_frequency=100.0,
            lead_time_seconds=100,
            change_failure_rate=0.01,
            mttr_seconds=60,
        )
        previous = DORAMetrics(
            deployment_frequency=0.001,
            lead_time_seconds=1_000_000,
            change_failure_rate=0.99,
            mttr_seconds=1_000_000,
        )
        trend = analyzer.trend(current, previous)
        assert trend["deployment_frequency"]["direction"] == "improving"
        assert trend["lead_time_seconds"]["direction"] == "improving"
        assert trend["change_failure_rate"]["direction"] == "improving"
        assert trend["mttr_seconds"]["direction"] == "improving"

    def test_trend_from_zero_previous(self):
        """Previous values are all zero (edge case for division)."""
        analyzer = DORAAnalyzer()
        current = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        previous = DORAMetrics(
            deployment_frequency=0.0,
            lead_time_seconds=0.0,
            change_failure_rate=0.0,
            mttr_seconds=0.0,
        )
        trend = analyzer.trend(current, previous)
        # When previous is 0 and current > 0, change_percent should be 100
        assert trend["deployment_frequency"]["change_percent"] == 100.0

    def test_team_comparison_many_teams(self):
        """Compare 20 teams."""
        analyzer = DORAAnalyzer()
        teams = {}
        for i in range(20):
            teams["team_{}".format(i)] = DORAMetrics(
                deployment_frequency=float(i + 1),
                lead_time_seconds=float(3600 * (21 - i)),
                change_failure_rate=round(i * 0.04, 2),
                mttr_seconds=float(1800 * (21 - i)),
            )
        comparison = analyzer.team_comparison(teams)
        assert comparison["deployment_frequency"]["best_team"] == "team_19"
        assert len(comparison["deployment_frequency"]["rankings"]) == 20


# ===========================================================================
# 4. PRLabeler with Many File Paths
# ===========================================================================


class TestPRLabelerScaling:
    """PRLabeler handles large numbers of changed files."""

    async def test_100_changed_files(self):
        labeler = PRLabeler()
        files = ["src/module_{}/handler.py".format(i) for i in range(100)]
        result = await labeler.classify(
            title="Fix authentication bug",
            changed_files=files,
        )
        assert isinstance(result, LabelClassification)
        assert len(result.labels) >= 1

    async def test_500_changed_files(self):
        labeler = PRLabeler()
        files = ["src/component_{}.py".format(i) for i in range(500)]
        result = await labeler.classify(
            title="Refactor entire codebase",
            changed_files=files,
        )
        assert isinstance(result, LabelClassification)

    async def test_mixed_file_types(self):
        """Many files of different types should produce multiple labels."""
        labeler = PRLabeler()
        files = (
            ["tests/test_{}.py".format(i) for i in range(50)]
            + ["docs/page_{}.md".format(i) for i in range(50)]
            + ["src/module_{}.py".format(i) for i in range(50)]
            + ["Dockerfile", "Makefile", "pyproject.toml"]
        )
        result = await labeler.classify(
            title="Major release prep",
            changed_files=files,
        )
        assert isinstance(result, LabelClassification)
        # Should detect at least TEST, DOCS, and CHORE
        assert PRLabel.TEST in result.labels
        assert PRLabel.DOCS in result.labels
        assert PRLabel.CHORE in result.labels

    async def test_performance_with_many_files(self):
        labeler = PRLabeler()
        files = ["path/to/deep/nested/file_{}.py".format(i) for i in range(1000)]
        start = time.monotonic()
        result = await labeler.classify(
            title="Fix bug",
            changed_files=files,
        )
        elapsed = time.monotonic() - start
        assert isinstance(result, LabelClassification)
        # Should complete within 1 second
        assert elapsed < 1.0, "Classifying 1000 files took {:.2f}s".format(elapsed)


# ===========================================================================
# 5. SignatureExtractor Scaling
# ===========================================================================


class TestSignatureExtractorScaling:
    """SignatureExtractor handles large source files."""

    def test_extract_from_100_functions(self):
        lines = []
        for i in range(100):
            lines.append("def func_{}(x: int, y: str = 'default') -> bool:".format(i))
            lines.append('    """Function {}."""'.format(i))
            lines.append("    return True")
            lines.append("")
        source = "\n".join(lines)
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(source, module="large_mod")
        assert len(sigs) == 100

    def test_extract_from_500_functions_performance(self):
        lines = []
        for i in range(500):
            lines.append("def func_{}(a: int, b: float, c: str) -> dict:".format(i))
            lines.append('    """Function {}."""'.format(i))
            lines.append("    return {}")
            lines.append("")
        source = "\n".join(lines)
        extractor = SignatureExtractor()
        start = time.monotonic()
        sigs = extractor.extract_from_source(source, module="large_mod")
        elapsed = time.monotonic() - start
        assert len(sigs) == 500
        # Should complete within 2 seconds
        assert elapsed < 2.0, "Extracting 500 signatures took {:.2f}s".format(elapsed)


# ===========================================================================
# 6. TestGenerator Scaling
# ===========================================================================


class TestTestGeneratorScaling:
    """TestGenerator handles modules with many functions."""

    async def test_generate_for_50_functions(self):
        lines = []
        for i in range(50):
            lines.append("def func_{}(x: int) -> int:".format(i))
            lines.append('    """Function {}."""'.format(i))
            lines.append("    return x + {}".format(i))
            lines.append("")
        source = "\n".join(lines)
        gen = TestGenerator()
        suite = await gen.generate_for_source(source, module="big_mod")
        # At least one test per function
        assert len(suite.test_cases) >= 50

    async def test_generate_for_100_functions_performance(self):
        lines = []
        for i in range(100):
            lines.append("def func_{}(x: int, y: str = 'test') -> bool:".format(i))
            lines.append('    """Function {}."""'.format(i))
            lines.append("    return True")
            lines.append("")
        source = "\n".join(lines)
        gen = TestGenerator()
        start = time.monotonic()
        suite = await gen.generate_for_source(source, module="perf_mod")
        elapsed = time.monotonic() - start
        assert len(suite.test_cases) >= 100
        # Should complete within 3 seconds
        assert elapsed < 3.0, "Generating tests for 100 functions took {:.2f}s".format(elapsed)


# ===========================================================================
# 7. MetricsStore Scaling
# ===========================================================================


class TestMetricsStoreScaling:
    """MetricsStore handles large amounts of data."""

    def test_1000_deployments_retrieval(self):
        store = MetricsStore()
        now = datetime.now(tz=UTC)
        for i in range(1000):
            store.add_deployment(
                DeploymentRecord(
                    id="d{}".format(i), repo="org/app", sha="s{}".format(i),
                    deployed_at=now - timedelta(minutes=i),
                )
            )
        # Default limit is 50
        result = store.get_deployments()
        assert len(result) == 50

        # Get all
        result = store.get_deployments(limit=1000)
        assert len(result) == 1000

    def test_100_dora_snapshots(self):
        store = MetricsStore()
        for i in range(100):
            store.add_dora_snapshot(
                DORAMetrics(
                    deployment_frequency=float(i),
                    lead_time_seconds=float(3600 - i),
                    change_failure_rate=min(i * 0.01, 1.0),
                    mttr_seconds=float(1800 - i),
                )
            )
        latest = store.get_latest_dora()
        assert latest is not None
        assert latest.deployment_frequency == 99.0

    def test_deployment_sorting_performance(self):
        store = MetricsStore()
        now = datetime.now(tz=UTC)
        # Add deployments in random-ish order
        for i in range(500):
            offset = (i * 7) % 500  # pseudo-random ordering
            store.add_deployment(
                DeploymentRecord(
                    id="d{}".format(i), repo="org/app", sha="s{}".format(i),
                    deployed_at=now - timedelta(minutes=offset),
                )
            )
        start = time.monotonic()
        result = store.get_deployments(limit=500)
        elapsed = time.monotonic() - start
        assert len(result) == 500
        # Verify sorted (most recent first)
        for i in range(len(result) - 1):
            assert result[i].deployed_at >= result[i + 1].deployed_at
        # Should complete quickly
        assert elapsed < 1.0


# ===========================================================================
# 8. ReviewAgent Scaling with Large Diffs
# ===========================================================================


class TestReviewAgentScaling:
    """ReviewAgent handles large diffs efficiently."""

    async def test_review_large_python_diff(self):
        """Review a diff with 500+ added Python lines."""
        lines = [
            "diff --git a/big.py b/big.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/big.py",
            "@@ -0,0 +1,500 @@",
        ]
        for i in range(500):
            lines.append("+def func_{}():".format(i))
            lines.append('+    """Function {}."""'.format(i))
            lines.append("+    return {}".format(i))
            lines.append("+")
        diff = "\n".join(lines)

        agent = ReviewAgent(enable_llm=False)
        start = time.monotonic()
        result = await agent.review_diff(diff)
        elapsed = time.monotonic() - start

        assert isinstance(result, ReviewResult)
        assert result.files_analyzed >= 1
        # Should complete within 5 seconds
        assert elapsed < 5.0, "Reviewing large diff took {:.2f}s".format(elapsed)

    async def test_review_many_files_diff(self):
        """Review a diff spanning 50 Python files."""
        parts = []
        for i in range(50):
            parts.append("diff --git a/mod_{}.py b/mod_{}.py".format(i, i))
            parts.append("new file mode 100644")
            parts.append("--- /dev/null")
            parts.append("+++ b/mod_{}.py".format(i))
            parts.append("@@ -0,0 +1,5 @@")
            parts.append("+def func_{}():".format(i))
            parts.append('+    """Function {}."""'.format(i))
            parts.append("+    return {}".format(i))
            parts.append("+")
        diff = "\n".join(parts)

        agent = ReviewAgent(enable_llm=False)
        result = await agent.review_diff(diff)
        assert result.files_analyzed == 50
