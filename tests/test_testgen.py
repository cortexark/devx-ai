"""Tests for the test generation module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from devx.core.config import LLMConfig
from devx.core.models import FunctionSignature, TestSuite
from devx.testgen.extractor import SignatureExtractor
from devx.testgen.generator import TestGenerator
from devx.testgen.templates import TestTemplate, TestTemplateRegistry

# ---------------------------------------------------------------------------
# SignatureExtractor
# ---------------------------------------------------------------------------


class TestSignatureExtractor:
    def test_extract_basic_function(self, sample_python_source):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(sample_python_source, module="utils")
        names = [s.name for s in sigs]
        assert "add" in names
        assert "divide" in names
        assert "fetch_data" in names

    def test_extract_parameters(self, sample_python_source):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(sample_python_source, module="utils")
        add_sig = next(s for s in sigs if s.name == "add")
        assert len(add_sig.parameters) == 2
        assert add_sig.parameters[0]["name"] == "a"
        assert add_sig.parameters[0]["type"] == "int"

    def test_extract_return_type(self, sample_python_source):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(sample_python_source, module="utils")
        add_sig = next(s for s in sigs if s.name == "add")
        assert add_sig.return_type == "int"

    def test_extract_docstring(self, sample_python_source):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(sample_python_source, module="utils")
        add_sig = next(s for s in sigs if s.name == "add")
        assert add_sig.docstring is not None
        assert "Add two numbers" in add_sig.docstring

    def test_extract_async_function(self, sample_python_source):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(sample_python_source, module="utils")
        fetch_sig = next(s for s in sigs if s.name == "fetch_data")
        assert fetch_sig.is_async is True

    def test_extract_default_parameter(self, sample_python_source):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(sample_python_source, module="utils")
        fetch_sig = next(s for s in sigs if s.name == "fetch_data")
        timeout_param = next(
            p for p in fetch_sig.parameters if p["name"] == "timeout"
        )
        assert timeout_param["default"] == "30"

    def test_skip_private_functions(self):
        source = """\
def public_func():
    pass

def _private_func():
    pass
"""
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(source)
        names = [s.name for s in sigs]
        assert "public_func" in names
        assert "_private_func" not in names

    def test_include_private_functions(self):
        source = """\
def public_func():
    pass

def _private_func():
    pass
"""
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(source, include_private=True)
        names = [s.name for s in sigs]
        assert "_private_func" in names

    def test_extract_from_invalid_source(self):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source("this is not valid python }{}{")
        assert sigs == []

    def test_extract_module_name(self, sample_python_source):
        extractor = SignatureExtractor()
        sigs = extractor.extract_from_source(
            sample_python_source, module="mypackage.utils"
        )
        assert all(s.module == "mypackage.utils" for s in sigs)


# ---------------------------------------------------------------------------
# TestTemplateRegistry
# ---------------------------------------------------------------------------


class TestTestTemplateRegistry:
    def test_default_templates_registered(self):
        registry = TestTemplateRegistry()
        templates = registry.get_templates()
        assert len(templates) > 0

    def test_get_by_category(self):
        registry = TestTemplateRegistry()
        unit_templates = registry.get_templates("unit")
        assert len(unit_templates) > 0
        assert all(t.category == "unit" for t in unit_templates)

    def test_get_edge_case_templates(self):
        registry = TestTemplateRegistry()
        edge_templates = registry.get_templates("edge_case")
        assert len(edge_templates) > 0

    def test_register_custom_template(self):
        registry = TestTemplateRegistry()
        custom = TestTemplate(
            name="custom_test",
            category="custom",
            description="A custom test template",
            template="def test_custom(): pass\n",
        )
        registry.register(custom)
        assert "custom" in registry.categories
        assert len(registry.get_templates("custom")) == 1

    def test_template_render(self):
        template = TestTemplate(
            name="test_render",
            category="unit",
            description="Test rendering",
            template="def test_{func_name}(): assert {func_name}() == {expected}\n",
        )
        rendered = template.render({"func_name": "add", "expected": "5"})
        assert "test_add" in rendered
        assert "assert add() == 5" in rendered

    def test_categories_property(self):
        registry = TestTemplateRegistry()
        cats = registry.categories
        assert "unit" in cats
        assert "edge_case" in cats
        assert "integration" in cats


# ---------------------------------------------------------------------------
# TestGenerator
# ---------------------------------------------------------------------------


class TestTestGenerator:
    @pytest.mark.asyncio
    async def test_generate_from_templates(self, sample_python_source):
        generator = TestGenerator()  # No LLM
        suite = await generator.generate_for_source(
            sample_python_source, module="utils"
        )
        assert isinstance(suite, TestSuite)
        assert len(suite.test_cases) >= 3  # At least one per function
        assert suite.framework == "pytest"

    @pytest.mark.asyncio
    async def test_generate_includes_imports(self, sample_python_source):
        generator = TestGenerator()
        suite = await generator.generate_for_source(
            sample_python_source, module="utils"
        )
        assert any("import pytest" in imp for imp in suite.imports)
        assert any("from utils import" in imp for imp in suite.imports)

    @pytest.mark.asyncio
    async def test_generate_with_category_filter(self, sample_python_source):
        generator = TestGenerator()
        suite = await generator.generate_for_source(
            sample_python_source, module="utils", categories=["unit"]
        )
        assert all(tc.category == "unit" for tc in suite.test_cases)

    @pytest.mark.asyncio
    async def test_generate_empty_source(self):
        generator = TestGenerator()
        suite = await generator.generate_for_source("", module="empty")
        assert suite.test_cases == []

    @pytest.mark.asyncio
    async def test_generate_with_mock_llm(
        self, sample_python_source, mock_testgen_llm_response
    ):
        config = LLMConfig(api_key="test")
        generator = TestGenerator(llm_config=config)
        generator._llm = AsyncMock()
        generator._llm.complete = AsyncMock(return_value=mock_testgen_llm_response)

        suite = await generator.generate_for_source(
            sample_python_source, module="utils"
        )
        assert len(suite.test_cases) >= 1

    @pytest.mark.asyncio
    async def test_generate_for_single_function(self):
        sig = FunctionSignature(
            name="multiply",
            parameters=[
                {"name": "a", "type": "int"},
                {"name": "b", "type": "int"},
            ],
            return_type="int",
        )
        generator = TestGenerator()
        tests = await generator.generate_for_function(sig)
        assert len(tests) >= 1
        assert all(tc.target_function == "multiply" for tc in tests)

    @pytest.mark.asyncio
    async def test_generate_handles_llm_failure(self, sample_python_source):
        config = LLMConfig(api_key="test")
        generator = TestGenerator(llm_config=config)
        generator._llm = AsyncMock()
        generator._llm.complete = AsyncMock(side_effect=RuntimeError("API error"))

        # Should fall back to templates
        suite = await generator.generate_for_source(
            sample_python_source, module="utils"
        )
        assert len(suite.test_cases) >= 1  # Template-based fallback

    @pytest.mark.asyncio
    async def test_generate_async_function_test(self):
        source = """\
async def fetch(url: str) -> dict:
    \"\"\"Fetch data.\"\"\"
    return {}
"""
        generator = TestGenerator()
        suite = await generator.generate_for_source(source, module="api")
        assert any("fetch" in tc.target_function for tc in suite.test_cases)
