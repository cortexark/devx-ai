"""Test generation engine combining signature extraction with LLM.

The generator uses extracted function metadata (signatures, docstrings,
type hints) to build context for LLM-driven test generation.  It falls
back to template-based generation when no LLM is available.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from devx.core.config import LLMConfig
from devx.core.llm_client import LLMClient
from devx.core.models import FunctionSignature, TestCase, TestSuite
from devx.testgen.extractor import SignatureExtractor
from devx.testgen.templates import TestTemplateRegistry

logger = logging.getLogger(__name__)

_TESTGEN_SYSTEM_PROMPT = """\
You are a test generation expert. Given a Python function's signature, docstring,
and source code, generate comprehensive pytest test cases.

For each function, generate:
1. At least one happy-path test with realistic inputs
2. Edge case tests (empty inputs, None, boundary values)
3. Error case tests where appropriate

Return a JSON array of test cases. Each must have:
- "name": test function name (e.g. "test_calculate_total_with_discount")
- "description": what the test verifies
- "code": complete test function source code (valid Python)
- "target_function": name of the function being tested
- "category": one of "unit", "integration", "edge_case"

Use pytest conventions. Include proper assertions.
Return ONLY the JSON array, no markdown fences.
"""


class TestGenerator:
    """Generate test cases from Python source code.

    Combines static extraction (signatures, types, docstrings) with
    LLM-powered test generation.  When no LLM is configured, produces
    template-based tests.

    Example::

        generator = TestGenerator(llm_config=LLMConfig(api_key="sk-..."))
        suite = await generator.generate_for_source(source_code, module="utils")
        for tc in suite.test_cases:
            print(tc.code)
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        *,
        template_registry: TestTemplateRegistry | None = None,
    ) -> None:
        self._extractor = SignatureExtractor()
        self._templates = template_registry or TestTemplateRegistry()
        self._llm: LLMClient | None = None

        if llm_config:
            self._llm = LLMClient(llm_config)

    async def generate_for_source(
        self,
        source: str,
        *,
        module: str = "",
        include_private: bool = False,
        categories: list[str] | None = None,
    ) -> TestSuite:
        """Generate a test suite for all functions in source code.

        Args:
            source: Python source code.
            module: Module path (e.g. ``mypackage.utils``).
            include_private: Whether to generate tests for private functions.
            categories: Filter test categories (unit, edge_case, integration).

        Returns:
            TestSuite with generated test cases.
        """
        signatures = self._extractor.extract_from_source(
            source, module=module, include_private=include_private
        )

        if not signatures:
            return TestSuite(module=module)

        if self._llm:
            test_cases = await self._generate_with_llm(signatures)
        else:
            test_cases = self._generate_from_templates(signatures)

        if categories:
            test_cases = [tc for tc in test_cases if tc.category in categories]

        imports = self._infer_imports(module, signatures)

        return TestSuite(
            module=module,
            imports=imports,
            test_cases=test_cases,
            framework="pytest",
        )

    async def generate_for_function(
        self,
        signature: FunctionSignature,
    ) -> list[TestCase]:
        """Generate tests for a single function.

        Args:
            signature: Extracted function signature.

        Returns:
            List of generated test cases.
        """
        if self._llm:
            return await self._generate_with_llm([signature])
        return self._generate_from_templates([signature])

    async def _generate_with_llm(
        self,
        signatures: list[FunctionSignature],
    ) -> list[TestCase]:
        """Use LLM to generate test cases from function signatures."""
        if not self._llm:
            return []

        # Build prompt with function details
        functions_desc = []
        for sig in signatures:
            desc: dict[str, Any] = {
                "name": sig.name,
                "parameters": sig.parameters,
                "return_type": sig.return_type,
                "is_async": sig.is_async,
            }
            if sig.docstring:
                desc["docstring"] = sig.docstring
            if sig.source:
                desc["source"] = sig.source
            functions_desc.append(desc)

        prompt = (
            "Generate pytest test cases for these Python functions:\n\n"
            f"{json.dumps(functions_desc, indent=2)}"
        )

        try:
            response = await self._llm.complete(
                prompt,
                system=_TESTGEN_SYSTEM_PROMPT,
                temperature=0.3,
            )
            return self._parse_llm_tests(response.content)
        except Exception:
            logger.exception("LLM test generation failed, falling back to templates")
            return self._generate_from_templates(signatures)

    def _parse_llm_tests(self, raw: str) -> list[TestCase]:
        """Parse LLM response into TestCase objects."""
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
            logger.warning("Failed to parse LLM test response as JSON")
            return []

        if not isinstance(data, list):
            return []

        test_cases: list[TestCase] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                tc = TestCase(
                    name=item.get("name", "test_unnamed"),
                    description=item.get("description", ""),
                    code=item.get("code", ""),
                    target_function=item.get("target_function", ""),
                    category=item.get("category", "unit"),
                )
                if tc.code:  # Only include tests with actual code
                    test_cases.append(tc)
            except (ValueError, KeyError):
                continue

        return test_cases

    def _generate_from_templates(
        self,
        signatures: list[FunctionSignature],
    ) -> list[TestCase]:
        """Generate tests using templates when no LLM is available."""
        test_cases: list[TestCase] = []

        for sig in signatures:
            # Basic unit test
            call_args = ", ".join(
                self._default_value_for_param(p) for p in sig.parameters
            )
            call_expr = (
                f"await {sig.name}({call_args})" if sig.is_async
                else f"{sig.name}({call_args})"
            )

            test_cases.append(TestCase(
                name=f"test_{sig.name}_returns_expected_result",
                description=f"Test that {sig.name} returns a valid result",
                code=(
                    f"def test_{sig.name}_returns_expected_result():\n"
                    f'    """Test that {sig.name} returns a valid result."""\n'
                    f"    result = {call_expr}\n"
                    f"    assert result is not None\n"
                ),
                target_function=sig.name,
                category="unit",
            ))

            # Type check test (if return type is annotated)
            if sig.return_type:
                test_cases.append(TestCase(
                    name=f"test_{sig.name}_returns_correct_type",
                    description=f"Test that {sig.name} returns {sig.return_type}",
                    code=(
                        f"def test_{sig.name}_returns_correct_type():\n"
                        f'    """Test that {sig.name} returns {sig.return_type}."""\n'
                        f"    result = {call_expr}\n"
                        f"    assert isinstance(result, {sig.return_type})\n"
                    ),
                    target_function=sig.name,
                    category="unit",
                ))

            # Edge case: test with no args if function has optional params
            has_defaults = any("default" in p for p in sig.parameters)
            if has_defaults and sig.parameters:
                test_cases.append(TestCase(
                    name=f"test_{sig.name}_with_defaults",
                    description=f"Test that {sig.name} works with default parameters",
                    code=(
                        f"def test_{sig.name}_with_defaults():\n"
                        f'    """Test {sig.name} with default parameter values."""\n'
                        f"    # Call with only required parameters\n"
                        f"    result = {sig.name}()\n"
                        f"    assert result is not None\n"
                    ),
                    target_function=sig.name,
                    category="edge_case",
                ))

        return test_cases

    @staticmethod
    def _default_value_for_param(param: dict[str, Any]) -> str:
        """Generate a sensible default value based on type annotation."""
        type_hint = param.get("type", "")
        name = param.get("name", "")

        if name.startswith("*"):
            return ""

        type_defaults: dict[str, str] = {
            "str": '"test_value"',
            "int": "42",
            "float": "3.14",
            "bool": "True",
            "list": "[]",
            "dict": "{}",
            "set": "set()",
            "bytes": 'b"test"',
            "None": "None",
        }

        for type_name, default in type_defaults.items():
            if type_name in type_hint:
                return default

        if "default" in param:
            return str(param["default"])

        return '"test_value"'

    @staticmethod
    def _infer_imports(module: str, signatures: list[FunctionSignature]) -> list[str]:
        """Infer necessary import statements for generated tests."""
        imports = ["import pytest"]

        if module:
            func_names = ", ".join(s.name for s in signatures)
            imports.append(f"from {module} import {func_names}")

        if any(s.is_async for s in signatures):
            imports.append("import pytest_asyncio")

        return imports
