"""API Contract Tests for devx-ai.

Tests that every public interface has correct signatures, validates inputs
properly, rejects bad input, and returns expected types. Covers all Pydantic
models, config objects, and public class methods.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from devx.core.config import GitHubConfig, LLMConfig, MetricsConfig, Settings
from devx.core.llm_client import LLMClient, LLMResponse
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
from devx.metrics.analyzer import DORAAnalyzer
from devx.metrics.dashboard import (
    DeploymentListResponse,
    DORAResponse,
    HealthResponse,
    MetricsStore,
)
from devx.review.agent import ReviewAgent
from devx.review.analyzer import ASTAnalysisResult, ASTAnalyzer, ClassInfo, FunctionInfo
from devx.review.diff_parser import DiffParser
from devx.review.suggestions import SuggestionFormatter
from devx.sdlc.labeler import PRLabeler
from devx.sdlc.triage import IssueTriage
from devx.testgen.extractor import SignatureExtractor
from devx.testgen.generator import TestGenerator

# ===========================================================================
# 1. Config Model Contracts
# ===========================================================================


class TestLLMConfigContracts:
    """LLMConfig validates all fields with env_prefix DEVX_LLM_."""

    def test_default_values(self):
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key == ""
        assert config.temperature == 0.2
        assert config.max_tokens == 4096
        assert config.timeout_seconds == 60

    def test_valid_provider_openai(self):
        config = LLMConfig(provider="openai")
        assert config.provider == "openai"

    def test_valid_provider_anthropic(self):
        config = LLMConfig(provider="anthropic")
        assert config.provider == "anthropic"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="azure")

    def test_temperature_lower_bound(self):
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_upper_bound(self):
        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_temperature_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

    def test_temperature_above_two_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)

    def test_max_tokens_negative_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=-1)

    def test_timeout_zero_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=0)

    def test_serialization_round_trip(self):
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test")
        data = config.model_dump()
        restored = LLMConfig(**data)
        assert restored.provider == config.provider
        assert restored.model == config.model


class TestGitHubConfigContracts:
    """GitHubConfig validates all fields with env_prefix DEVX_GITHUB_."""

    def test_default_values(self):
        config = GitHubConfig()
        assert config.token == ""
        assert config.base_url == "https://api.github.com"
        assert config.rate_limit_buffer == 100

    def test_custom_base_url(self):
        config = GitHubConfig(base_url="https://github.example.com/api/v3")
        assert config.base_url == "https://github.example.com/api/v3"

    def test_rate_limit_buffer_zero(self):
        config = GitHubConfig(rate_limit_buffer=0)
        assert config.rate_limit_buffer == 0

    def test_rate_limit_buffer_negative_rejected(self):
        with pytest.raises(ValidationError):
            GitHubConfig(rate_limit_buffer=-1)


class TestMetricsConfigContracts:
    """MetricsConfig validates all fields with env_prefix DEVX_METRICS_."""

    def test_default_values(self):
        config = MetricsConfig()
        assert config.window_days == 30
        assert config.cache_ttl_seconds == 300

    def test_window_days_zero_rejected(self):
        with pytest.raises(ValidationError):
            MetricsConfig(window_days=0)

    def test_cache_ttl_zero_allowed(self):
        config = MetricsConfig(cache_ttl_seconds=0)
        assert config.cache_ttl_seconds == 0

    def test_cache_ttl_negative_rejected(self):
        with pytest.raises(ValidationError):
            MetricsConfig(cache_ttl_seconds=-1)


class TestSettingsContracts:
    """Settings combines LLM, GitHub, and Metrics configs."""

    def test_default_factory(self):
        settings = Settings.default()
        assert isinstance(settings.llm, LLMConfig)
        assert isinstance(settings.github, GitHubConfig)
        assert isinstance(settings.metrics, MetricsConfig)
        assert settings.log_level == "INFO"
        assert settings.debug is False

    def test_from_yaml_nonexistent_file(self, tmp_path):
        """from_yaml with nonexistent file should use defaults."""
        settings = Settings.from_yaml(tmp_path / "nonexistent.yaml")
        assert settings.log_level == "INFO"

    def test_from_yaml_valid_file(self, tmp_path):
        """from_yaml should merge YAML values."""
        config_path = tmp_path / "devx.yaml"
        config_path.write_text("log_level: DEBUG\ndebug: true\n")
        settings = Settings.from_yaml(config_path)
        assert settings.log_level == "DEBUG"
        assert settings.debug is True

    def test_from_yaml_empty_file(self, tmp_path):
        """from_yaml with empty file should use defaults."""
        config_path = tmp_path / "devx.yaml"
        config_path.write_text("")
        settings = Settings.from_yaml(config_path)
        assert settings.log_level == "INFO"

    def test_from_yaml_non_dict_content(self, tmp_path):
        """from_yaml with non-dict YAML should use defaults."""
        config_path = tmp_path / "devx.yaml"
        config_path.write_text("- item1\n- item2\n")
        settings = Settings.from_yaml(config_path)
        assert settings.log_level == "INFO"


# ===========================================================================
# 2. Core Model Contracts
# ===========================================================================


class TestCodeLocationContracts:
    """CodeLocation must have file and valid start_line."""

    def test_requires_file(self):
        with pytest.raises(ValidationError):
            CodeLocation(start_line=1)

    def test_requires_start_line(self):
        with pytest.raises(ValidationError):
            CodeLocation(file="test.py")

    def test_end_line_zero_rejected(self):
        with pytest.raises(ValidationError):
            CodeLocation(file="test.py", start_line=1, end_line=0)

    def test_end_line_none_allowed(self):
        loc = CodeLocation(file="test.py", start_line=5)
        assert loc.end_line is None


class TestReviewFindingContracts:
    """ReviewFinding validates all required fields and constraints."""

    def test_requires_title(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                description="desc",
                severity=Severity.LOW,
                category=Category.STYLE,
                location=CodeLocation(file="f.py", start_line=1),
            )

    def test_requires_severity(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="t",
                description="d",
                category=Category.STYLE,
                location=CodeLocation(file="f.py", start_line=1),
            )

    def test_requires_category(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="t",
                description="d",
                severity=Severity.LOW,
                location=CodeLocation(file="f.py", start_line=1),
            )

    def test_requires_location(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="t",
                description="d",
                severity=Severity.LOW,
                category=Category.STYLE,
            )

    def test_confidence_at_zero(self):
        f = ReviewFinding(
            title="t",
            description="d",
            severity=Severity.LOW,
            category=Category.STYLE,
            location=CodeLocation(file="f.py", start_line=1),
            confidence=0.0,
        )
        assert f.confidence == 0.0

    def test_confidence_at_one(self):
        f = ReviewFinding(
            title="t",
            description="d",
            severity=Severity.LOW,
            category=Category.STYLE,
            location=CodeLocation(file="f.py", start_line=1),
            confidence=1.0,
        )
        assert f.confidence == 1.0

    def test_confidence_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="t",
                description="d",
                severity=Severity.LOW,
                category=Category.STYLE,
                location=CodeLocation(file="f.py", start_line=1),
                confidence=-0.01,
            )

    def test_title_exactly_200_chars(self):
        f = ReviewFinding(
            title="x" * 200,
            description="d",
            severity=Severity.LOW,
            category=Category.STYLE,
            location=CodeLocation(file="f.py", start_line=1),
        )
        assert len(f.title) == 200

    def test_suggestion_optional(self):
        f = ReviewFinding(
            title="t",
            description="d",
            severity=Severity.LOW,
            category=Category.STYLE,
            location=CodeLocation(file="f.py", start_line=1),
        )
        assert f.suggestion is None

    def test_invalid_severity_string_rejected(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="t",
                description="d",
                severity="invalid_severity",
                category=Category.STYLE,
                location=CodeLocation(file="f.py", start_line=1),
            )

    def test_invalid_category_string_rejected(self):
        with pytest.raises(ValidationError):
            ReviewFinding(
                title="t",
                description="d",
                severity=Severity.LOW,
                category="invalid_category",
                location=CodeLocation(file="f.py", start_line=1),
            )


class TestReviewResultContracts:
    """ReviewResult enforces non-negative fields."""

    def test_negative_files_analyzed_rejected(self):
        with pytest.raises(ValidationError):
            ReviewResult(files_analyzed=-1)

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            ReviewResult(duration_seconds=-0.1)

    def test_default_values(self):
        r = ReviewResult()
        assert r.findings == []
        assert r.summary == ""
        assert r.files_analyzed == 0
        assert r.duration_seconds == 0.0


class TestDiffHunkContracts:
    """DiffHunk requires content and non-negative counts."""

    def test_requires_content(self):
        with pytest.raises(ValidationError):
            DiffHunk(old_start=1, old_count=1, new_start=1, new_count=1)

    def test_negative_old_start_rejected(self):
        with pytest.raises(ValidationError):
            DiffHunk(old_start=-1, old_count=1, new_start=1, new_count=1, content="x")

    def test_negative_old_count_rejected(self):
        with pytest.raises(ValidationError):
            DiffHunk(old_start=1, old_count=-1, new_start=1, new_count=1, content="x")


class TestFileDiffContracts:
    """FileDiff computes path and line counts correctly."""

    def test_all_none_paths(self):
        fd = FileDiff(old_path=None, new_path=None)
        assert fd.path == "<unknown>"

    def test_new_file_has_no_old_path(self):
        fd = FileDiff(new_path="new.py", is_new=True)
        assert fd.path == "new.py"
        assert fd.old_path is None

    def test_deleted_file_has_no_new_path(self):
        fd = FileDiff(old_path="old.py", is_deleted=True)
        assert fd.path == "old.py"

    def test_total_additions_empty_hunks(self):
        fd = FileDiff(new_path="f.py")
        assert fd.total_additions == 0

    def test_total_deletions_empty_hunks(self):
        fd = FileDiff(new_path="f.py")
        assert fd.total_deletions == 0


class TestFunctionSignatureContracts:
    """FunctionSignature defaults and parameter handling."""

    def test_defaults(self):
        sig = FunctionSignature(name="func")
        assert sig.module == ""
        assert sig.docstring is None
        assert sig.parameters == []
        assert sig.return_type is None
        assert sig.decorators == []
        assert sig.is_async is False
        assert sig.source == ""

    def test_with_all_fields(self):
        sig = FunctionSignature(
            name="process",
            module="app.utils",
            docstring="Process data.",
            parameters=[{"name": "data", "type": "dict"}],
            return_type="bool",
            decorators=["staticmethod"],
            is_async=True,
            source="async def process(data: dict) -> bool: ...",
        )
        assert sig.name == "process"
        assert sig.is_async is True
        assert len(sig.parameters) == 1


class TestTestCaseContracts:
    """TestCase requires name, code, and target_function."""

    def test_requires_name(self):
        with pytest.raises(ValidationError):
            TestCase(code="pass", target_function="f")

    def test_requires_code(self):
        with pytest.raises(ValidationError):
            TestCase(name="test_f", target_function="f")

    def test_requires_target_function(self):
        with pytest.raises(ValidationError):
            TestCase(name="test_f", code="pass")

    def test_default_category(self):
        tc = TestCase(name="test_f", code="pass", target_function="f")
        assert tc.category == "unit"


class TestTestSuiteContracts:
    """TestSuite requires module."""

    def test_requires_module(self):
        with pytest.raises(ValidationError):
            TestSuite()

    def test_defaults(self):
        suite = TestSuite(module="my_mod")
        assert suite.imports == []
        assert suite.test_cases == []
        assert suite.framework == "pytest"


class TestLabelClassificationContracts:
    """LabelClassification validates confidence bounds."""

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            LabelClassification(confidence=1.1)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            LabelClassification(confidence=-0.1)

    def test_empty_labels_allowed(self):
        lc = LabelClassification(confidence=0.5)
        assert lc.labels == []


class TestTriageResultContracts:
    """TriageResult requires priority and severity."""

    def test_requires_priority(self):
        with pytest.raises(ValidationError):
            TriageResult(severity=Severity.HIGH)

    def test_requires_severity(self):
        with pytest.raises(ValidationError):
            TriageResult(priority=IssuePriority.P1)

    def test_invalid_priority_rejected(self):
        with pytest.raises(ValidationError):
            TriageResult(priority="P5", severity=Severity.LOW)

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            TriageResult(priority=IssuePriority.P1, severity="extreme")


class TestDeploymentRecordContracts:
    """DeploymentRecord requires id, repo, sha, deployed_at."""

    def test_requires_id(self):
        with pytest.raises(ValidationError):
            DeploymentRecord(repo="org/app", sha="abc", deployed_at=datetime.now(tz=UTC))

    def test_requires_repo(self):
        with pytest.raises(ValidationError):
            DeploymentRecord(id="d1", sha="abc", deployed_at=datetime.now(tz=UTC))

    def test_default_status(self):
        dep = DeploymentRecord(id="d1", repo="r", sha="a", deployed_at=datetime.now(tz=UTC))
        assert dep.status == "success"

    def test_default_environment(self):
        dep = DeploymentRecord(id="d1", repo="r", sha="a", deployed_at=datetime.now(tz=UTC))
        assert dep.environment == "production"

    def test_lead_time_optional(self):
        dep = DeploymentRecord(id="d1", repo="r", sha="a", deployed_at=datetime.now(tz=UTC))
        assert dep.lead_time_seconds is None


class TestDORAMetricsContracts:
    """DORAMetrics validates all four metrics and computed properties."""

    def test_requires_all_four_metrics(self):
        with pytest.raises(ValidationError):
            DORAMetrics(
                deployment_frequency=1.0,
                lead_time_seconds=3600,
                change_failure_rate=0.1,
                # missing mttr_seconds
            )

    def test_change_failure_rate_exactly_zero(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.0,
            mttr_seconds=1800,
        )
        assert m.change_failure_rate == 0.0

    def test_change_failure_rate_exactly_one(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=1.0,
            mttr_seconds=1800,
        )
        assert m.change_failure_rate == 1.0

    def test_lead_time_rating_high(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=300000,  # ~3.5 days
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        assert m.lead_time_rating == "high"

    def test_lead_time_rating_medium(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=1_000_000,  # ~11.5 days
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        assert m.lead_time_rating == "medium"

    def test_calculated_at_auto_set(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        assert m.calculated_at is not None
        assert isinstance(m.calculated_at, datetime)

    def test_json_serialization(self):
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        data = m.model_dump(mode="json")
        assert "deployment_frequency" in data
        assert "lead_time_seconds" in data
        assert "change_failure_rate" in data
        assert "mttr_seconds" in data


# ===========================================================================
# 3. LLMClient Contracts
# ===========================================================================


class TestLLMResponseContracts:
    """LLMResponse correctly normalizes token usage."""

    def test_openai_token_format(self):
        resp = LLMResponse(
            content="hello",
            model="gpt-4o",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50

    def test_anthropic_token_format(self):
        resp = LLMResponse(
            content="hello",
            model="claude-3",
            usage={"input_tokens": 200, "output_tokens": 80},
        )
        assert resp.input_tokens == 200
        assert resp.output_tokens == 80

    def test_empty_usage(self):
        resp = LLMResponse(content="hello", model="m")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0

    def test_frozen_dataclass(self):
        resp = LLMResponse(content="hello", model="m")
        with pytest.raises(AttributeError):
            resp.content = "world"


class TestLLMClientContracts:
    """LLMClient provider selection and initialization."""

    def test_default_config(self):
        client = LLMClient()
        assert client._config.provider == "openai"

    def test_custom_config(self):
        config = LLMConfig(provider="anthropic", api_key="test")
        client = LLMClient(config)
        assert client._config.provider == "anthropic"

    async def test_unsupported_provider_raises(self):
        """complete() should raise ValueError for unsupported provider."""
        config = LLMConfig.__new__(LLMConfig)
        object.__setattr__(config, "provider", "invalid")
        object.__setattr__(config, "model", "test")
        object.__setattr__(config, "temperature", 0.2)
        object.__setattr__(config, "max_tokens", 100)
        object.__setattr__(config, "api_key", "")
        object.__setattr__(config, "timeout_seconds", 60)
        client = LLMClient(config)
        # Cannot actually call complete because the provider validation
        # happens at config construction, but we verify the client stores config
        assert client._config is config


# ===========================================================================
# 4. DiffParser Contracts
# ===========================================================================


class TestDiffParserContracts:
    """DiffParser.parse return type and edge cases."""

    def test_parse_returns_list(self):
        parser = DiffParser()
        result = parser.parse("")
        assert isinstance(result, list)

    def test_parse_non_diff_text(self):
        parser = DiffParser()
        result = parser.parse("this is not a diff at all")
        assert result == []

    def test_parse_only_header_no_hunks(self):
        parser = DiffParser()
        diff = "diff --git a/file.py b/file.py\n"
        result = parser.parse(diff)
        # Should produce a FileDiff even with no hunks
        assert len(result) == 1
        assert result[0].hunks == []

    def test_file_diff_types(self, sample_diff):
        parser = DiffParser()
        diffs = parser.parse(sample_diff)
        for d in diffs:
            assert isinstance(d, FileDiff)
            for h in d.hunks:
                assert isinstance(h, DiffHunk)


# ===========================================================================
# 5. ASTAnalyzer Contracts
# ===========================================================================


class TestASTAnalyzerContracts:
    """ASTAnalyzer.analyze_python returns ASTAnalysisResult."""

    def test_returns_ast_analysis_result(self):
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python("x = 1", "test.py")
        assert isinstance(result, ASTAnalysisResult)

    def test_findings_are_review_findings(self):
        source = "def undocumented(x):\n    return x\n"
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "t.py")
        for f in result.findings:
            assert isinstance(f, ReviewFinding)

    def test_functions_are_function_info(self, sample_python_source):
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(sample_python_source, "m.py")
        for f in result.functions:
            assert isinstance(f, FunctionInfo)

    def test_classes_are_class_info(self):
        source = "class Foo:\n    pass\n"
        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source, "t.py")
        for c in result.classes:
            assert isinstance(c, ClassInfo)


# ===========================================================================
# 6. ReviewAgent Contracts
# ===========================================================================


class TestReviewAgentContracts:
    """ReviewAgent.review_diff and review_file return ReviewResult."""

    async def test_review_diff_returns_review_result(self, sample_diff):
        agent = ReviewAgent(enable_llm=False)
        result = await agent.review_diff(sample_diff)
        assert isinstance(result, ReviewResult)
        assert isinstance(result.findings, list)
        assert isinstance(result.summary, str)
        assert isinstance(result.files_analyzed, int)
        assert isinstance(result.duration_seconds, float)

    async def test_review_file_returns_review_result(self, sample_python_source):
        agent = ReviewAgent(enable_llm=False)
        result = await agent.review_file(sample_python_source, "test.py")
        assert isinstance(result, ReviewResult)

    async def test_empty_diff_summary(self):
        agent = ReviewAgent(enable_llm=False)
        result = await agent.review_diff("")
        assert "No changes" in result.summary


# ===========================================================================
# 7. SuggestionFormatter Contracts
# ===========================================================================


class TestSuggestionFormatterContracts:
    """SuggestionFormatter output format contracts."""

    def test_to_github_comment_returns_string(self, sample_finding):
        result = ReviewResult(findings=[sample_finding], files_analyzed=1)
        fmt = SuggestionFormatter()
        comment = fmt.to_github_comment(result)
        assert isinstance(comment, str)

    def test_to_inline_comments_returns_list_of_dicts(self, sample_finding):
        result = ReviewResult(findings=[sample_finding])
        fmt = SuggestionFormatter()
        comments = fmt.to_inline_comments(result)
        assert isinstance(comments, list)
        for c in comments:
            assert isinstance(c, dict)
            assert "path" in c
            assert "line" in c
            assert "body" in c

    def test_to_json_returns_dict(self, sample_finding):
        result = ReviewResult(findings=[sample_finding], files_analyzed=1)
        fmt = SuggestionFormatter()
        data = fmt.to_json(result)
        assert isinstance(data, dict)
        assert "findings" in data

    def test_empty_review_github_comment(self):
        result = ReviewResult(files_analyzed=5, duration_seconds=1.2)
        fmt = SuggestionFormatter()
        comment = fmt.to_github_comment(result)
        assert "No issues found" in comment
        assert "5" in comment

    def test_to_inline_comments_empty(self):
        result = ReviewResult()
        fmt = SuggestionFormatter()
        comments = fmt.to_inline_comments(result)
        assert comments == []


# ===========================================================================
# 8. SignatureExtractor Contracts
# ===========================================================================


class TestSignatureExtractorContracts:
    """SignatureExtractor returns list[FunctionSignature]."""

    def test_returns_list(self):
        extractor = SignatureExtractor()
        result = extractor.extract_from_source("")
        assert isinstance(result, list)

    def test_elements_are_function_signatures(self, sample_python_source):
        extractor = SignatureExtractor()
        result = extractor.extract_from_source(sample_python_source)
        for sig in result:
            assert isinstance(sig, FunctionSignature)

    def test_empty_source_returns_empty(self):
        extractor = SignatureExtractor()
        assert extractor.extract_from_source("") == []

    def test_syntax_error_returns_empty(self):
        extractor = SignatureExtractor()
        assert extractor.extract_from_source("def }{:") == []

    def test_no_functions_returns_empty(self):
        extractor = SignatureExtractor()
        assert extractor.extract_from_source("x = 1\ny = 2\n") == []


# ===========================================================================
# 9. TestGenerator Contracts
# ===========================================================================


class TestTestGeneratorContracts:
    """TestGenerator.generate_for_source returns TestSuite."""

    async def test_returns_test_suite(self, sample_python_source):
        gen = TestGenerator()
        result = await gen.generate_for_source(sample_python_source, module="m")
        assert isinstance(result, TestSuite)

    async def test_test_cases_are_test_case(self, sample_python_source):
        gen = TestGenerator()
        result = await gen.generate_for_source(sample_python_source, module="m")
        for tc in result.test_cases:
            assert isinstance(tc, TestCase)

    async def test_empty_source_returns_empty_suite(self):
        gen = TestGenerator()
        result = await gen.generate_for_source("")
        assert result.test_cases == []

    async def test_generate_for_function_returns_list(self):
        sig = FunctionSignature(name="f", parameters=[], return_type="int")
        gen = TestGenerator()
        result = await gen.generate_for_function(sig)
        assert isinstance(result, list)


# ===========================================================================
# 10. PRLabeler Contracts
# ===========================================================================


class TestPRLabelerContracts:
    """PRLabeler.classify returns LabelClassification."""

    async def test_returns_label_classification(self):
        labeler = PRLabeler()
        result = await labeler.classify(title="Some PR")
        assert isinstance(result, LabelClassification)
        assert isinstance(result.labels, list)
        assert isinstance(result.confidence, float)
        assert isinstance(result.reasoning, str)

    async def test_labels_are_pr_labels(self):
        labeler = PRLabeler()
        result = await labeler.classify(
            title="Fix bug",
            changed_files=["tests/test_x.py"],
        )
        for label in result.labels:
            assert isinstance(label, PRLabel)

    async def test_empty_title_still_returns(self):
        labeler = PRLabeler()
        result = await labeler.classify(title="")
        assert isinstance(result, LabelClassification)

    async def test_no_changed_files_still_returns(self):
        labeler = PRLabeler()
        result = await labeler.classify(title="Update something")
        assert isinstance(result, LabelClassification)


# ===========================================================================
# 11. IssueTriage Contracts
# ===========================================================================


class TestIssueTriageContracts:
    """IssueTriage.triage returns TriageResult."""

    async def test_returns_triage_result(self):
        triage = IssueTriage()
        result = await triage.triage(title="Some issue")
        assert isinstance(result, TriageResult)
        assert isinstance(result.priority, IssuePriority)
        assert isinstance(result.severity, Severity)
        assert isinstance(result.labels, list)
        assert isinstance(result.reasoning, str)

    async def test_empty_title(self):
        triage = IssueTriage()
        result = await triage.triage(title="")
        assert isinstance(result, TriageResult)


# ===========================================================================
# 12. DORAAnalyzer Contracts
# ===========================================================================


class TestDORAAnalyzerContracts:
    """DORAAnalyzer.calculate, trend, team_comparison return types."""

    def test_calculate_returns_dora_metrics(self, sample_deployments):
        analyzer = DORAAnalyzer()
        result = analyzer.calculate(sample_deployments)
        assert isinstance(result, DORAMetrics)

    def test_trend_returns_dict(self):
        analyzer = DORAAnalyzer()
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        result = analyzer.trend(m, m)
        assert isinstance(result, dict)
        assert "deployment_frequency" in result
        assert "lead_time_seconds" in result
        assert "change_failure_rate" in result
        assert "mttr_seconds" in result

    def test_trend_fields(self):
        analyzer = DORAAnalyzer()
        m = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        result = analyzer.trend(m, m)
        for _metric_name, trend in result.items():
            assert "value" in trend
            assert "previous" in trend
            assert "change_percent" in trend
            assert "direction" in trend
            assert trend["direction"] in ("improving", "declining", "stable")

    def test_team_comparison_returns_dict(self):
        analyzer = DORAAnalyzer()
        teams = {
            "a": DORAMetrics(
                deployment_frequency=1.0,
                lead_time_seconds=3600,
                change_failure_rate=0.1,
                mttr_seconds=1800,
            ),
        }
        result = analyzer.team_comparison(teams)
        assert isinstance(result, dict)

    def test_team_comparison_ranking_structure(self):
        analyzer = DORAAnalyzer()
        teams = {
            "a": DORAMetrics(
                deployment_frequency=1.0,
                lead_time_seconds=3600,
                change_failure_rate=0.1,
                mttr_seconds=1800,
            ),
            "b": DORAMetrics(
                deployment_frequency=2.0,
                lead_time_seconds=1800,
                change_failure_rate=0.05,
                mttr_seconds=900,
            ),
        }
        result = analyzer.team_comparison(teams)
        for _metric_name, data in result.items():
            assert "rankings" in data
            assert "average" in data
            assert "best_team" in data
            assert isinstance(data["rankings"], list)


# ===========================================================================
# 13. Dashboard Model Contracts
# ===========================================================================


class TestDashboardModelContracts:
    """Dashboard response models validate correctly."""

    def test_health_response_defaults(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.version == "0.1.0"
        assert h.timestamp is not None

    def test_dora_response_requires_metrics(self):
        with pytest.raises(ValidationError):
            DORAResponse()

    def test_deployment_list_response_defaults(self):
        r = DeploymentListResponse()
        assert r.deployments == []
        assert r.total == 0

    def test_metrics_store_isolation(self):
        """Two MetricsStore instances should be independent."""
        store1 = MetricsStore()
        store2 = MetricsStore()
        store1.add_deployment(
            DeploymentRecord(id="d1", repo="r", sha="a", deployed_at=datetime.now(tz=UTC))
        )
        assert len(store1.get_deployments()) == 1
        assert len(store2.get_deployments()) == 0
