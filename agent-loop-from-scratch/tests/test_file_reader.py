"""
Tests for the read_file tool: reading, listing, pagination, and the sandbox.

`tmp_path` is a built-in pytest fixture: a fresh, empty temporary folder for
each test, deleted afterwards. We build a tiny fake workspace inside it.
"""

from pathlib import Path

import pytest

from agent_loop.tools import ToolError
from agent_loop.tools.file_reader import make_read_file_tool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A small sandbox folder, plus a 'secret' file just OUTSIDE it."""
    root = tmp_path / "workspace"
    (root / "reports").mkdir(parents=True)
    (root / "notes.md").write_text("line one\nline two\nline three\n")
    (root / "reports" / "q3.txt").write_text("July: 100\n")
    (root / ".env").write_text("SECRET=inside-but-hidden\n")
    (root / "image.bin").write_bytes(b"\x89PNG\x00\x00\x00binary")
    (tmp_path / "secret.txt").write_text("TOP SECRET: outside the sandbox\n")
    return root


def read(workspace: Path, **tool_input) -> str:
    return make_read_file_tool(workspace).execute(tool_input)


def test_reads_a_file_with_line_numbers(workspace: Path) -> None:
    result = read(workspace, path="notes.md")
    assert result.startswith("File: notes.md (3 lines")
    assert "1\tline one" in result
    assert "3\tline three" in result


def test_lists_a_directory_without_hidden_files(workspace: Path) -> None:
    result = read(workspace, path=".")
    assert "reports/" in result
    assert "notes.md" in result
    assert ".env" not in result


def test_reads_files_in_subfolders(workspace: Path) -> None:
    assert "July: 100" in read(workspace, path="reports/q3.txt")


@pytest.mark.parametrize(
    "escape_attempt",
    [
        "../secret.txt",              # climb out with ..
        "reports/../../secret.txt",   # hide the .. in the middle
        "/etc/passwd",                # absolute path
    ],
)
def test_paths_outside_the_sandbox_are_refused(workspace: Path, escape_attempt: str) -> None:
    with pytest.raises(ToolError, match="outside the workspace"):
        read(workspace, path=escape_attempt)


def test_a_symlink_pointing_outside_is_refused(workspace: Path) -> None:
    # A link INSIDE the sandbox that points OUTSIDE. resolve() follows it,
    # so the check sees where it really goes.
    (workspace / "innocent_link.txt").symlink_to(workspace.parent / "secret.txt")
    with pytest.raises(ToolError, match="outside the workspace"):
        read(workspace, path="innocent_link.txt")


def test_hidden_files_are_refused_even_inside_the_sandbox(workspace: Path) -> None:
    with pytest.raises(ToolError, match="hidden"):
        read(workspace, path=".env")


def test_missing_files_get_a_helpful_hint(workspace: Path) -> None:
    with pytest.raises(ToolError, match="path '.' to list"):
        read(workspace, path="nope.txt")


def test_binary_files_are_refused(workspace: Path) -> None:
    with pytest.raises(ToolError, match="binary"):
        read(workspace, path="image.bin")


def test_large_files_are_refused(workspace: Path) -> None:
    tool = make_read_file_tool(workspace, max_file_bytes=10)
    with pytest.raises(ToolError, match="too large"):
        tool.execute({"path": "notes.md"})


def test_long_files_are_paginated_with_instructions_to_continue(workspace: Path) -> None:
    (workspace / "long.txt").write_text("\n".join(f"row {n}" for n in range(1, 26)))
    tool = make_read_file_tool(workspace, max_lines=10)

    first_page = tool.execute({"path": "long.txt"})
    assert "10\trow 10" in first_page
    assert "11\trow 11" not in first_page
    assert "call read_file again with start_line=11" in first_page

    last_page = tool.execute({"path": "long.txt", "start_line": 21})
    assert "25\trow 25" in last_page
    assert "Showing lines 21-25 of 25.]" in last_page  # no "read more" on the final page


@pytest.mark.parametrize(("start_line", "message"), [(0, "1 or more"), (99, "past the end")])
def test_bad_start_lines_are_explained(workspace: Path, start_line: int, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        read(workspace, path="notes.md", start_line=start_line)
