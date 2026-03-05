"""Example: Generate tests for a Python module using devx-ai.

Usage:
    export DEVX_LLM_API_KEY="sk-..."
    python examples/generate_tests.py path/to/module.py

    # Without LLM (template-based):
    python examples/generate_tests.py --templates-only path/to/module.py

    # Save output to file:
    python examples/generate_tests.py path/to/module.py -o tests/test_module.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from devx.core.config import LLMConfig
from devx.testgen.generator import TestGenerator


async def generate_tests(
    source_path: str,
    *,
    templates_only: bool = False,
    output_path: str | None = None,
) -> None:
    """Generate test cases for a Python source file."""

    path = Path(source_path)
    if not path.exists():
        print(f"Error: File not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    source = path.read_text()
    module = path.stem

    # Configure generator
    if templates_only:
        generator = TestGenerator()
        print(f"Generating template-based tests for {path.name}...")
    else:
        llm_config = LLMConfig()
        generator = TestGenerator(llm_config=llm_config)
        print(f"Generating LLM-powered tests for {path.name}...")

    # Generate
    suite = await generator.generate_for_source(source, module=module)

    if not suite.test_cases:
        print("No test cases generated. The file may not contain public functions.")
        return

    # Build output
    lines: list[str] = []
    lines.append(f'"""Generated tests for {module}."""\n')
    for imp in suite.imports:
        lines.append(imp)
    lines.append("\n")

    for tc in suite.test_cases:
        lines.append(f"\n{tc.code}\n")

    output = "\n".join(lines)

    # Write or print
    if output_path:
        Path(output_path).write_text(output)
        print(f"Wrote {len(suite.test_cases)} tests to {output_path}")
    else:
        print(f"\n--- Generated {len(suite.test_cases)} test cases ---\n")
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pytest tests for a Python module"
    )
    parser.add_argument("source", help="Path to the Python source file")
    parser.add_argument(
        "--templates-only",
        action="store_true",
        help="Use templates only (no LLM)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    asyncio.run(
        generate_tests(
            args.source,
            templates_only=args.templates_only,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
