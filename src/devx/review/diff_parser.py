"""Git unified diff parser.

Converts raw ``git diff`` output into structured ``FileDiff`` / ``DiffHunk``
models.  Handles renames, new files, deletions, and multi-hunk diffs.
"""

from __future__ import annotations

import re

from devx.core.models import DiffHunk, FileDiff

# Regex patterns for unified diff parsing
_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_OLD_FILE = re.compile(r"^--- (?:a/)?(.+)$")
_NEW_FILE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_NEW_FILE_MODE = re.compile(r"^new file mode")
_DELETED_FILE_MODE = re.compile(r"^deleted file mode")
_RENAME_FROM = re.compile(r"^rename from (.+)$")
_RENAME_TO = re.compile(r"^rename to (.+)$")


class DiffParser:
    """Parse unified diff text into structured models.

    Example::

        parser = DiffParser()
        file_diffs = parser.parse(raw_diff_text)
        for fd in file_diffs:
            print(fd.path, fd.total_additions, fd.total_deletions)
    """

    def parse(self, diff_text: str) -> list[FileDiff]:
        """Parse a full unified diff into a list of ``FileDiff`` objects.

        Args:
            diff_text: Raw output from ``git diff``.

        Returns:
            List of parsed file diffs.
        """
        if not diff_text.strip():
            return []

        file_diffs: list[FileDiff] = []
        lines = diff_text.splitlines()
        idx = 0

        while idx < len(lines):
            header_match = _DIFF_HEADER.match(lines[idx])
            if not header_match:
                idx += 1
                continue

            old_path = header_match.group(1)
            new_path = header_match.group(2)
            is_new = False
            is_deleted = False
            is_rename = False
            idx += 1

            # Parse metadata lines between diff header and hunks
            while (
                idx < len(lines)
                and not lines[idx].startswith("---")
                and not _DIFF_HEADER.match(lines[idx])
            ):
                if _NEW_FILE_MODE.match(lines[idx]):
                    is_new = True
                elif _DELETED_FILE_MODE.match(lines[idx]):
                    is_deleted = True
                elif _RENAME_FROM.match(lines[idx]):
                    is_rename = True
                    m = _RENAME_FROM.match(lines[idx])
                    if m:
                        old_path = m.group(1)
                elif _RENAME_TO.match(lines[idx]):
                    m = _RENAME_TO.match(lines[idx])
                    if m:
                        new_path = m.group(1)
                idx += 1

            # Parse --- and +++ lines
            if idx < len(lines) and lines[idx].startswith("---"):
                m = _OLD_FILE.match(lines[idx])
                if m and m.group(1) != "/dev/null":
                    old_path = m.group(1)
                elif m and m.group(1) == "/dev/null":
                    is_new = True
                idx += 1

            if idx < len(lines) and lines[idx].startswith("+++"):
                m = _NEW_FILE.match(lines[idx])
                if m and m.group(1) != "/dev/null":
                    new_path = m.group(1)
                elif m and m.group(1) == "/dev/null":
                    is_deleted = True
                idx += 1

            # Parse hunks
            hunks: list[DiffHunk] = []
            while idx < len(lines) and not _DIFF_HEADER.match(lines[idx]):
                hunk_match = _HUNK_HEADER.match(lines[idx])
                if hunk_match:
                    hunk, idx = self._parse_hunk(lines, idx, hunk_match)
                    hunks.append(hunk)
                else:
                    idx += 1

            file_diffs.append(
                FileDiff(
                    old_path=old_path if not is_new else None,
                    new_path=new_path if not is_deleted else None,
                    hunks=hunks,
                    is_new=is_new,
                    is_deleted=is_deleted,
                    is_rename=is_rename,
                )
            )

        return file_diffs

    def _parse_hunk(
        self,
        lines: list[str],
        idx: int,
        header_match: re.Match[str],
    ) -> tuple[DiffHunk, int]:
        """Parse a single hunk starting at the @@ line.

        Returns:
            Tuple of (DiffHunk, next line index).
        """
        old_start = int(header_match.group(1))
        old_count = int(header_match.group(2) or "1")
        new_start = int(header_match.group(3))
        new_count = int(header_match.group(4) or "1")
        header = lines[idx]
        idx += 1

        content_lines: list[str] = []
        while idx < len(lines):
            line = lines[idx]
            if line.startswith(("diff --git", "@@")):
                break
            # Hunk content: context, additions, deletions, and no-newline markers
            if line.startswith(("+", "-", " ", "\\")):
                content_lines.append(line)
                idx += 1
            elif line == "":
                # Empty lines in diffs are context lines (missing space prefix)
                content_lines.append(" ")
                idx += 1
            else:
                # Treat unexpected lines as context to be safe
                break

        return (
            DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header=header,
                content="\n".join(content_lines),
            ),
            idx,
        )
