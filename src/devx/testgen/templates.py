"""Test templates for different testing patterns.

Templates are Jinja-like string templates (using plain str.format for
zero dependencies) that produce pytest test code from function metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TestTemplate:
    """A reusable test code template.

    Attributes:
        name: Template identifier (e.g. "unit_basic", "edge_null").
        category: Test category (unit, integration, edge_case).
        description: What this template tests.
        template: Python code template with ``{placeholders}``.
    """

    name: str
    category: str
    description: str
    template: str

    def render(self, context: dict[str, Any]) -> str:
        """Render the template with the provided context.

        Args:
            context: Dictionary of placeholder values.

        Returns:
            Rendered test code string.
        """
        return self.template.format(**context)


class TestTemplateRegistry:
    """Registry of test templates organized by category.

    Provides built-in templates for common test patterns and allows
    registration of custom templates.

    Example::

        registry = TestTemplateRegistry()
        templates = registry.get_templates("unit")
        for t in templates:
            code = t.render({"func_name": "add", "module": "math_utils"})
    """

    def __init__(self) -> None:
        self._templates: dict[str, list[TestTemplate]] = {}
        self._register_defaults()

    def register(self, template: TestTemplate) -> None:
        """Register a new test template.

        Args:
            template: TestTemplate instance to register.
        """
        self._templates.setdefault(template.category, []).append(template)

    def get_templates(self, category: str | None = None) -> list[TestTemplate]:
        """Get templates, optionally filtered by category.

        Args:
            category: Filter by category. None returns all templates.

        Returns:
            List of matching templates.
        """
        if category is None:
            return [t for templates in self._templates.values() for t in templates]
        return list(self._templates.get(category, []))

    @property
    def categories(self) -> list[str]:
        """Return all registered categories."""
        return list(self._templates.keys())

    def _register_defaults(self) -> None:
        """Register built-in test templates."""
        # --- Unit tests ---
        self.register(TestTemplate(
            name="unit_basic",
            category="unit",
            description="Basic unit test for a function with known inputs and outputs",
            template=(
                "def test_{func_name}_returns_expected_result():\n"
                '    """Test that {func_name} returns the expected result for valid input."""\n'
                "    # Arrange\n"
                "    {arrange}\n"
                "\n"
                "    # Act\n"
                "    result = {call_expression}\n"
                "\n"
                "    # Assert\n"
                "    {assertion}\n"
            ),
        ))

        self.register(TestTemplate(
            name="unit_return_type",
            category="unit",
            description="Verify return type matches annotation",
            template=(
                "def test_{func_name}_returns_correct_type():\n"
                '    """Test that {func_name} returns the annotated type."""\n'
                "    result = {call_expression}\n"
                "    assert isinstance(result, {return_type})\n"
            ),
        ))

        # --- Edge case tests ---
        self.register(TestTemplate(
            name="edge_none_input",
            category="edge_case",
            description="Test behavior with None input",
            template=(
                "def test_{func_name}_handles_none_input():\n"
                '    """Test that {func_name} handles None input gracefully."""\n'
                "    with pytest.raises({expected_exception}):\n"
                "        {func_name}(None)\n"
            ),
        ))

        self.register(TestTemplate(
            name="edge_empty_input",
            category="edge_case",
            description="Test behavior with empty input",
            template=(
                "def test_{func_name}_handles_empty_input():\n"
                '    """Test that {func_name} handles empty input."""\n'
                "    result = {func_name}({empty_value})\n"
                "    assert result == {expected_empty_result}\n"
            ),
        ))

        self.register(TestTemplate(
            name="edge_boundary",
            category="edge_case",
            description="Test boundary conditions",
            template=(
                "def test_{func_name}_boundary_values():\n"
                '    """Test {func_name} with boundary values."""\n'
                "    # Test minimum boundary\n"
                "    {min_test}\n"
                "\n"
                "    # Test maximum boundary\n"
                "    {max_test}\n"
            ),
        ))

        # --- Integration tests ---
        self.register(TestTemplate(
            name="integration_basic",
            category="integration",
            description="Integration test verifying component interaction",
            template=(
                "def test_{func_name}_integration():\n"
                '    """Integration test for {func_name} with real dependencies."""\n'
                "    # Setup\n"
                "    {setup}\n"
                "\n"
                "    # Execute\n"
                "    result = {call_expression}\n"
                "\n"
                "    # Verify\n"
                "    {verification}\n"
            ),
        ))

        # --- Async tests ---
        self.register(TestTemplate(
            name="async_basic",
            category="unit",
            description="Async unit test",
            template=(
                "@pytest.mark.asyncio\n"
                "async def test_{func_name}_async():\n"
                '    """Test that async {func_name} completes successfully."""\n'
                "    result = await {call_expression}\n"
                "    {assertion}\n"
            ),
        ))

        # --- Exception tests ---
        self.register(TestTemplate(
            name="unit_raises",
            category="unit",
            description="Test that function raises expected exception for invalid input",
            template=(
                "def test_{func_name}_raises_on_invalid_input():\n"
                '    """Test that {func_name} raises {expected_exception} for invalid input."""\n'
                "    with pytest.raises({expected_exception}):\n"
                "        {invalid_call}\n"
            ),
        ))
