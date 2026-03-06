"""AST analysis engine using tree-sitter.

Provides structural code analysis that complements LLM-based review.
tree-sitter gives us language-agnostic parsing with incremental update
support, making it suitable for large diffs and multi-language repos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from devx.core.models import Category, CodeLocation, ReviewFinding, Severity

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Extracted metadata about a function definition."""

    name: str
    start_line: int
    end_line: int
    parameter_count: int
    body_lines: int
    has_docstring: bool
    has_return_type: bool
    nested_depth: int = 0
    complexity: int = 1  # Cyclomatic complexity approximation


@dataclass
class ClassInfo:
    """Extracted metadata about a class definition."""

    name: str
    start_line: int
    end_line: int
    method_count: int
    has_docstring: bool
    base_classes: list[str] = field(default_factory=list)


@dataclass
class ASTAnalysisResult:
    """Complete AST analysis result for a file."""

    file_path: str
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    findings: list[ReviewFinding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Complexity / style thresholds
# ---------------------------------------------------------------------------

MAX_FUNCTION_LINES = 50
MAX_PARAMETERS = 5
MAX_NESTING_DEPTH = 4
MAX_CLASS_METHODS = 20


class ASTAnalyzer:
    """Analyze source code structure using tree-sitter.

    The analyzer currently supports Python.  It extracts function and class
    metadata, detects complexity issues, and generates review findings for
    structural anti-patterns.

    Example::

        analyzer = ASTAnalyzer()
        result = analyzer.analyze_python(source_code, "src/foo.py")
        for finding in result.findings:
            print(finding.title, finding.severity)
    """

    def __init__(self) -> None:
        self._parser: Any = None
        self._python_language: Any = None

    def _ensure_parser(self) -> None:
        """Lazily initialize the tree-sitter parser with Python grammar."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser

            self._python_language = Language(tspython.language())
            self._parser = Parser(self._python_language)
        except (ImportError, OSError, TypeError) as exc:
            logger.warning("tree-sitter unavailable, falling back to regex analysis: %s", exc)
            self._parser = None

    def analyze_python(self, source: str, file_path: str = "<stdin>") -> ASTAnalysisResult:
        """Analyze Python source code and return structural findings.

        Args:
            source: Python source code as a string.
            file_path: File path for location references in findings.

        Returns:
            ASTAnalysisResult with extracted metadata and findings.
        """
        self._ensure_parser()
        if self._parser is not None:
            return self._analyze_with_tree_sitter(source, file_path)
        return self._analyze_with_fallback(source, file_path)

    def _analyze_with_tree_sitter(self, source: str, file_path: str) -> ASTAnalysisResult:
        """Full analysis using tree-sitter AST."""
        tree = self._parser.parse(source.encode("utf-8"))
        root = tree.root_node

        result = ASTAnalysisResult(file_path=file_path)

        for node in self._walk(root):
            if node.type == "function_definition":
                func_info = self._extract_function_info(node, source)
                result.functions.append(func_info)
            elif node.type == "class_definition":
                class_info = self._extract_class_info(node, source)
                result.classes.append(class_info)
            elif node.type in ("import_statement", "import_from_statement"):
                result.imports.append(source[node.start_byte : node.end_byte])

        result.findings = self._generate_findings(result, file_path)
        return result

    def _walk(self, node: Any) -> list[Any]:
        """Depth-first walk of tree-sitter node, returning top-level definitions."""
        results = []
        cursor = node.walk()

        reached_root = False
        while True:
            if cursor.node.type in (
                "function_definition",
                "class_definition",
                "import_statement",
                "import_from_statement",
            ):
                results.append(cursor.node)

            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue

            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True
                elif cursor.goto_next_sibling():
                    retracing = False

            if reached_root:
                break

        return results

    def _extract_function_info(self, node: Any, source: str) -> FunctionInfo:
        """Extract FunctionInfo from a tree-sitter function_definition node."""
        name = ""
        param_count = 0
        has_docstring = False
        has_return_type = False

        for child in node.children:
            if child.type == "identifier":
                name = source[child.start_byte : child.end_byte]
            elif child.type == "parameters":
                # Count parameters excluding 'self' and 'cls'
                params = [
                    c
                    for c in child.children
                    if c.type
                    in (
                        "identifier",
                        "typed_parameter",
                        "default_parameter",
                        "typed_default_parameter",
                        "list_splat_pattern",
                        "dictionary_splat_pattern",
                    )
                ]
                param_names = []
                for p in params:
                    text = source[p.start_byte : p.end_byte]
                    param_name = text.split(":")[0].split("=")[0].strip().lstrip("*")
                    param_names.append(param_name)
                param_count = len([p for p in param_names if p not in ("self", "cls")])
            elif child.type == "type":
                has_return_type = True
            elif child.type == "block":
                # Check first statement for docstring
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr in stmt.children:
                            if expr.type == "string":
                                has_docstring = True
                        break
                    elif stmt.type != "comment":
                        break

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        body_lines = end_line - start_line

        return FunctionInfo(
            name=name,
            start_line=start_line,
            end_line=end_line,
            parameter_count=param_count,
            body_lines=body_lines,
            has_docstring=has_docstring,
            has_return_type=has_return_type,
        )

    def _extract_class_info(self, node: Any, source: str) -> ClassInfo:
        """Extract ClassInfo from a tree-sitter class_definition node."""
        name = ""
        method_count = 0
        has_docstring = False
        base_classes: list[str] = []

        for child in node.children:
            if child.type == "identifier":
                name = source[child.start_byte : child.end_byte]
            elif child.type == "argument_list":
                for arg in child.children:
                    if arg.type in ("identifier", "attribute"):
                        base_classes.append(source[arg.start_byte : arg.end_byte])
            elif child.type == "block":
                for stmt in child.children:
                    if stmt.type == "function_definition":
                        method_count += 1
                    elif stmt.type == "expression_statement" and method_count == 0:
                        for expr in stmt.children:
                            if expr.type == "string":
                                has_docstring = True

        return ClassInfo(
            name=name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            method_count=method_count,
            has_docstring=has_docstring,
            base_classes=base_classes,
        )

    def _analyze_with_fallback(self, source: str, file_path: str) -> ASTAnalysisResult:
        """Lightweight regex-based analysis when tree-sitter is unavailable."""
        import re

        result = ASTAnalysisResult(file_path=file_path)
        lines = source.splitlines()

        func_pattern = re.compile(r"^\s*(async\s+)?def\s+(\w+)\s*\(")
        class_pattern = re.compile(r"^\s*class\s+(\w+)")

        for i, line in enumerate(lines):
            func_match = func_pattern.match(line)
            if func_match:
                name = func_match.group(2)
                # Rough body length: lines until next def/class at same indent
                indent = len(line) - len(line.lstrip())
                end = i + 1
                for j in range(i + 1, len(lines)):
                    stripped = lines[j]
                    if stripped.strip() and len(stripped) - len(stripped.lstrip()) <= indent:
                        if func_pattern.match(stripped) or class_pattern.match(stripped):
                            break
                    end = j + 1
                result.functions.append(
                    FunctionInfo(
                        name=name,
                        start_line=i + 1,
                        end_line=end,
                        parameter_count=0,
                        body_lines=end - i,
                        has_docstring=False,
                        has_return_type="->" in line,
                    )
                )

            class_match = class_pattern.match(line)
            if class_match:
                result.classes.append(
                    ClassInfo(
                        name=class_match.group(1),
                        start_line=i + 1,
                        end_line=i + 1,
                        method_count=0,
                        has_docstring=False,
                    )
                )

        result.findings = self._generate_findings(result, file_path)
        return result

    def _generate_findings(self, result: ASTAnalysisResult, file_path: str) -> list[ReviewFinding]:
        """Generate review findings from extracted metadata."""
        findings: list[ReviewFinding] = []

        for func in result.functions:
            if func.body_lines > MAX_FUNCTION_LINES:
                findings.append(
                    ReviewFinding(
                        title=f"Function `{func.name}` is too long ({func.body_lines} lines)",
                        description=(
                            f"Functions longer than {MAX_FUNCTION_LINES} lines are harder to "
                            f"understand and test. Consider extracting helper functions."
                        ),
                        severity=Severity.MEDIUM,
                        category=Category.COMPLEXITY,
                        location=CodeLocation(
                            file=file_path,
                            start_line=func.start_line,
                            end_line=func.end_line,
                        ),
                        suggestion="Break this function into smaller, focused functions.",
                    )
                )

            if func.parameter_count > MAX_PARAMETERS:
                findings.append(
                    ReviewFinding(
                        title=(
                            f"Function `{func.name}` has too many "
                            f"parameters ({func.parameter_count})"
                        ),
                        description=(
                            f"Functions with more than {MAX_PARAMETERS} parameters suggest the "
                            f"need for a parameter object or decomposition."
                        ),
                        severity=Severity.LOW,
                        category=Category.MAINTAINABILITY,
                        location=CodeLocation(file=file_path, start_line=func.start_line),
                        suggestion="Group related parameters into a dataclass or Pydantic model.",
                    )
                )

            if not func.has_docstring and not func.name.startswith("_"):
                findings.append(
                    ReviewFinding(
                        title=f"Public function `{func.name}` lacks a docstring",
                        description=(
                            "All public functions should have docstrings for discoverability."
                        ),
                        severity=Severity.INFO,
                        category=Category.DOCUMENTATION,
                        location=CodeLocation(file=file_path, start_line=func.start_line),
                    )
                )

        for cls in result.classes:
            if cls.method_count > MAX_CLASS_METHODS:
                findings.append(
                    ReviewFinding(
                        title=f"Class `{cls.name}` has too many methods ({cls.method_count})",
                        description=(
                            f"Classes with more than {MAX_CLASS_METHODS} methods may violate "
                            f"the Single Responsibility Principle."
                        ),
                        severity=Severity.MEDIUM,
                        category=Category.MAINTAINABILITY,
                        location=CodeLocation(file=file_path, start_line=cls.start_line),
                        suggestion="Consider splitting into multiple focused classes.",
                    )
                )

        return findings
