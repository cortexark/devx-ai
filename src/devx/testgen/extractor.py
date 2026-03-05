"""Extract function signatures, docstrings, and type hints from Python source.

Uses the ``ast`` module (stdlib) for reliable Python-specific extraction.
This complements tree-sitter which is used for multi-language AST analysis
in the review module.
"""

from __future__ import annotations

import ast
from typing import Any

from devx.core.models import FunctionSignature


class SignatureExtractor:
    """Extract function metadata from Python source code.

    Example::

        extractor = SignatureExtractor()
        signatures = extractor.extract_from_source(source_code, module="my_module")
        for sig in signatures:
            print(sig.name, sig.parameters, sig.return_type)
    """

    def extract_from_source(
        self,
        source: str,
        *,
        module: str = "",
        include_private: bool = False,
    ) -> list[FunctionSignature]:
        """Extract all function signatures from Python source code.

        Args:
            source: Python source code string.
            module: Module path for context (e.g. ``mypackage.utils``).
            include_private: Whether to include ``_``-prefixed functions.

        Returns:
            List of extracted function signatures.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        signatures: list[FunctionSignature] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not include_private and node.name.startswith("_"):
                    continue
                sig = self._extract_function(node, source, module)
                signatures.append(sig)

        return signatures

    def extract_from_file(
        self,
        file_path: str,
        *,
        include_private: bool = False,
    ) -> list[FunctionSignature]:
        """Extract signatures from a Python file on disk.

        Args:
            file_path: Path to the Python source file.
            include_private: Whether to include ``_``-prefixed functions.

        Returns:
            List of extracted function signatures.
        """
        with open(file_path) as f:
            source = f.read()

        # Derive module name from file path
        module = file_path.replace("/", ".").replace("\\", ".").removesuffix(".py")

        return self.extract_from_source(
            source,
            module=module,
            include_private=include_private,
        )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        module: str,
    ) -> FunctionSignature:
        """Extract a single function's metadata from its AST node."""
        # Parameters
        parameters = self._extract_parameters(node.args)

        # Return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Docstring
        docstring = ast.get_docstring(node)

        # Decorators
        decorators = [ast.unparse(d) for d in node.decorator_list]

        # Source code
        try:
            func_source = ast.get_source_segment(source, node) or ""
        except (TypeError, ValueError):
            func_source = ""

        return FunctionSignature(
            name=node.name,
            module=module,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            source=func_source,
        )

    def _extract_parameters(self, args: ast.arguments) -> list[dict[str, Any]]:
        """Extract parameter info from function arguments."""
        params: list[dict[str, Any]] = []

        # Calculate default offset: defaults align to the end of args
        num_args = len(args.args)
        num_defaults = len(args.defaults)
        default_offset = num_args - num_defaults

        for i, arg in enumerate(args.args):
            if arg.arg in ("self", "cls"):
                continue

            param: dict[str, Any] = {"name": arg.arg}

            # Type annotation
            if arg.annotation:
                param["type"] = ast.unparse(arg.annotation)

            # Default value
            default_idx = i - default_offset
            if default_idx >= 0 and default_idx < len(args.defaults):
                param["default"] = ast.unparse(args.defaults[default_idx])

            params.append(param)

        # *args
        if args.vararg:
            param = {"name": f"*{args.vararg.arg}"}
            if args.vararg.annotation:
                param["type"] = ast.unparse(args.vararg.annotation)
            params.append(param)

        # **kwargs
        if args.kwarg:
            param = {"name": f"**{args.kwarg.arg}"}
            if args.kwarg.annotation:
                param["type"] = ast.unparse(args.kwarg.annotation)
            params.append(param)

        return params
