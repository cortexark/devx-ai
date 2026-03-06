"""Suggestion models and formatters for code review output.

Formats review findings into different output targets: GitHub PR comments,
terminal output (via rich), or structured JSON for API consumers.
"""

from __future__ import annotations

from typing import ClassVar

from devx.core.models import ReviewFinding, ReviewResult, Severity


class SuggestionFormatter:
    """Format review findings for different output targets.

    Example::

        formatter = SuggestionFormatter()
        markdown = formatter.to_github_comment(result)
        formatter.print_terminal(result)
    """

    # Severity to emoji mapping for GitHub comments
    _SEVERITY_ICONS: ClassVar[dict[Severity, str]] = {
        Severity.CRITICAL: "[CRITICAL]",
        Severity.HIGH: "[HIGH]",
        Severity.MEDIUM: "[MEDIUM]",
        Severity.LOW: "[LOW]",
        Severity.INFO: "[INFO]",
    }

    _SEVERITY_COLORS: ClassVar[dict[Severity, str]] = {
        Severity.CRITICAL: "red",
        Severity.HIGH: "bright_red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "cyan",
        Severity.INFO: "dim",
    }

    def to_github_comment(self, result: ReviewResult) -> str:
        """Format review result as a GitHub PR comment in Markdown.

        Args:
            result: Aggregated review result.

        Returns:
            Markdown-formatted string suitable for GitHub API.
        """
        if not result.findings:
            return self._empty_review_message(result)

        parts: list[str] = []
        parts.append("## Code Review Summary\n")
        parts.append(f"Analyzed **{result.files_analyzed}** files ")
        parts.append(f"in {result.duration_seconds:.1f}s. ")
        parts.append(f"Found **{len(result.findings)}** items.\n")

        if result.summary:
            parts.append(f"\n{result.summary}\n")

        # Group by severity
        by_severity = self._group_by_severity(result.findings)
        all_severities = (
            Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
            Severity.LOW, Severity.INFO,
        )
        for severity in all_severities:
            findings = by_severity.get(severity, [])
            if not findings:
                continue

            icon = self._SEVERITY_ICONS[severity]
            parts.append(f"\n### {icon} {severity.value.title()} ({len(findings)})\n")

            for f in findings:
                parts.append(f"\n**{f.title}**")
                parts.append(f"  `{f.location}`\n")
                parts.append(f"  {f.description}\n")
                if f.suggestion:
                    parts.append(f"\n  > Suggestion: {f.suggestion}\n")

        return "\n".join(parts)

    def to_inline_comments(self, result: ReviewResult) -> list[dict[str, str | int]]:
        """Convert findings to GitHub inline review comment format.

        Returns:
            List of dicts with 'path', 'line', 'body' keys.
        """
        comments: list[dict[str, str | int]] = []
        for f in result.findings:
            body_parts = [
                f"**{self._SEVERITY_ICONS[f.severity]} {f.title}**\n",
                f"{f.description}\n",
            ]
            if f.suggestion:
                body_parts.append(f"\n**Suggestion:** {f.suggestion}")

            comments.append({
                "path": f.location.file,
                "line": f.location.start_line,
                "body": "\n".join(body_parts),
            })
        return comments

    def to_json(self, result: ReviewResult) -> dict[str, object]:
        """Serialize review result to a JSON-compatible dict.

        Returns:
            Dictionary representation of the full result.
        """
        return result.model_dump(mode="json")

    def print_terminal(self, result: ReviewResult) -> None:
        """Print review findings to the terminal using rich.

        Args:
            result: Aggregated review result.
        """
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            # Fallback to plain print if rich is not installed
            for f in result.findings:
                print(f"[{f.severity.value}] {f.title} at {f.location}")
            return

        console = Console()
        if not result.findings:
            console.print("[green]No issues found.[/green]")
            return

        table = Table(title="Code Review Findings", show_lines=True)
        table.add_column("Severity", style="bold", width=10)
        table.add_column("Category", width=15)
        table.add_column("Title", min_width=30)
        table.add_column("Location", width=25)

        for f in result.findings:
            color = self._SEVERITY_COLORS[f.severity]
            table.add_row(
                f"[{color}]{f.severity.value}[/{color}]",
                f.category.value,
                f.title,
                str(f.location),
            )

        console.print(table)
        console.print(
            f"\n[bold]{len(result.findings)}[/bold] findings in "
            f"[bold]{result.files_analyzed}[/bold] files "
            f"({result.duration_seconds:.1f}s)"
        )

    @staticmethod
    def _group_by_severity(findings: list[ReviewFinding]) -> dict[Severity, list[ReviewFinding]]:
        """Group findings by severity level."""
        groups: dict[Severity, list[ReviewFinding]] = {}
        for f in findings:
            groups.setdefault(f.severity, []).append(f)
        return groups

    @staticmethod
    def _empty_review_message(result: ReviewResult) -> str:
        """Generate message for clean reviews."""
        return (
            "## Code Review Summary\n\n"
            f"Analyzed **{result.files_analyzed}** files "
            f"in {result.duration_seconds:.1f}s.\n\n"
            "No issues found. The code looks good."
        )
