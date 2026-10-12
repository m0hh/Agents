"""
tools/file_reader.py: read text files, but ONLY inside a sandbox folder.

THE THREAT
    The model chooses the `path`. If we simply did `open(path)`, then
        "../.env"                  → your API key
        "/etc/passwd"              → system files
        "~/.ssh/id_rsa"            → your SSH private key
    would all be one tool call away, and the model might be TRICKED into
    asking for them by text it read earlier (prompt injection). This attack
    is called PATH TRAVERSAL.

THE DEFENSE: resolve, then check
    1. Join the path onto the sandbox folder and RESOLVE it into a clean
       absolute path: `..` steps are collapsed and symlinks are followed to
       where they really point.
    2. Check that the result is still INSIDE the sandbox folder.
    Checking the raw text for ".." is NOT enough. Absolute paths, symlinks and
    other tricks get around naive string checks. Always resolve first.

    As extra protection, hidden files and folders (names starting with ".",
    like `.env` or `.git`) are refused too. That's where secrets usually live.

KEEPING OUTPUT SMALL
    Everything a tool returns is added to the conversation and RESENT on every
    later request (Phase 2's lesson). So we return at most 200 lines per call,
    with line numbers, and tell the model how to read the next chunk
    ("call again with start_line=201"). This is PAGINATION, and every
    serious agent tool does it.
"""

from pathlib import Path

from agent_loop.tools.base import Tool, ToolError

# Defaults for the limits. They're parameters of make_read_file_tool, so
# tests can use tiny limits.
MAX_FILE_BYTES = 1_000_000    # refuse to open files bigger than 1 MB
MAX_LINES_PER_CALL = 200      # lines returned per call (pagination)
MAX_CHARS_PER_CALL = 12_000   # and never more than this many characters
MAX_LISTING_ENTRIES = 200     # entries shown when listing a directory


def _is_hidden(path: Path, root: Path) -> bool:
    """True if any part of `path` below `root` starts with a dot (.env, .git/...)."""
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _human_size(size: int) -> str:
    """1536 → '1.5 KB'."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def make_read_file_tool(
    root: Path,
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_lines: int = MAX_LINES_PER_CALL,
    max_chars: int = MAX_CHARS_PER_CALL,
) -> Tool:
    """Build a read_file Tool that can only see files inside `root`.

    This is a FACTORY: a function that builds and returns a configured object.
    The inner `read_file` function "remembers" root and the limits (a Python
    *closure*), so every call is confined to this sandbox.
    """
    # resolve() once, up front: the sandbox's true absolute path.
    root = root.resolve()

    def resolve_inside_root(user_path: str) -> Path:
        """Turn the model's path into a safe absolute path, or refuse."""
        # `root / user_path`: if user_path is ABSOLUTE ("/etc/passwd"), pathlib
        # throws root away and returns user_path itself. The check below
        # catches that, along with "../" tricks and symlinks.
        candidate = (root / user_path).resolve()
        if not candidate.is_relative_to(root):
            raise ToolError(
                f"Access denied: '{user_path}' is outside the workspace folder. "
                "Only files inside the workspace can be read; use paths like 'notes.md'."
            )
        if candidate != root and _is_hidden(candidate, root):
            raise ToolError(f"Access denied: '{user_path}' is a hidden file or folder.")
        return candidate

    def list_directory(directory: Path) -> str:
        """A readable listing: folders end with '/', files show their size."""
        entries = [entry for entry in directory.iterdir() if not entry.name.startswith(".")]
        # Folders first, then files, each alphabetically.
        entries.sort(key=lambda entry: (not entry.is_dir(), entry.name.lower()))
        relative = directory.relative_to(root).as_posix()
        label = "." if relative == "." else relative + "/"
        lines = [f"Directory: {label} ({len(entries)} entries)"]
        for entry in entries[:MAX_LISTING_ENTRIES]:
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
            else:
                lines.append(f"  {entry.name}  ({_human_size(entry.stat().st_size)})")
        if len(entries) > MAX_LISTING_ENTRIES:
            lines.append(f"  … and {len(entries) - MAX_LISTING_ENTRIES} more")
        return "\n".join(lines)

    def read_text(file: Path, start_line: int) -> str:
        """Return one page of a text file, with line numbers."""
        size = file.stat().st_size
        if size > max_file_bytes:
            raise ToolError(f"File is too large ({_human_size(size)}; limit {_human_size(max_file_bytes)}).")

        data = file.read_bytes()
        # A NUL byte almost never appears in text, but is everywhere in binary files.
        if b"\x00" in data[:8192]:
            raise ToolError("This looks like a binary file (image, archive, ...); only text files can be read.")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError("The file is not valid UTF-8 text.") from None

        relative = file.relative_to(root).as_posix()
        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return f"File: {relative} (empty)"
        if start_line < 1:
            # Range rule checked in CODE: strict mode doesn't allow "minimum" in the schema.
            raise ToolError("start_line must be 1 or more (lines are numbered from 1).")
        if start_line > total:
            raise ToolError(f"start_line {start_line} is past the end of the file ({total} lines).")

        # Collect lines from start_line until we hit the line or character limit.
        output: list[str] = []
        used_chars = 0
        last_line = start_line - 1
        for number in range(start_line, min(total, start_line + max_lines - 1) + 1):
            numbered = f"{number}\t{lines[number - 1]}"  # like `cat -n`
            if len(numbered) > max_chars:
                # One gigantic line (e.g. minified JSON) must not blow the limit on its own.
                numbered = numbered[:max_chars] + " …[line truncated]"
            if output and used_chars + len(numbered) > max_chars:
                break
            output.append(numbered)
            used_chars += len(numbered) + 1
            last_line = number

        header = f"File: {relative} ({total} lines, {_human_size(size)})"
        body = "\n".join(output)
        if start_line == 1 and last_line == total:
            return f"{header}\n{body}"
        # Tell the model EXACTLY how to get the rest. It follows instructions like this well.
        footer = f"[Showing lines {start_line}-{last_line} of {total}."
        footer += f" To read more, call read_file again with start_line={last_line + 1}.]" if last_line < total else "]"
        return f"{header}\n{body}\n{footer}"

    def read_file(path: str, start_line: int = 1) -> str:
        """The tool's function: read a file (one page) or list a directory."""
        target = resolve_inside_root(path)
        if not target.exists():
            raise ToolError(f"No such file or directory: '{path}'. Call read_file with path '.' to list the workspace.")
        if target.is_dir():
            return list_directory(target)
        return read_text(target, start_line)

    return Tool(
        name="read_file",
        description=(
            "Read a UTF-8 text file from the user's workspace folder, or list a folder's contents. "
            "Use this whenever the user refers to a file, its data, or 'my files'. Paths are relative "
            "to the workspace, e.g. 'prices.csv' or 'reports/q3_summary.txt'; use '.' to list everything. "
            f"Returns up to {max_lines} numbered lines per call; for longer files, call again with "
            "start_line to read the next part."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or folder path relative to the workspace, e.g. 'notes.md' or '.'.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional. 1-based line number to start reading from (default 1).",
                },
            },
            # start_line is NOT required: the model may leave it out, and then
            # Python's default value (1) applies.
            "required": ["path"],
            "additionalProperties": False,
        },
        function=read_file,
    )
