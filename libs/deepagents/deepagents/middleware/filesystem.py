"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.backends import StateBackend

# Re-export type here for backwards compatibility
from deepagents.backends.protocol import BACKEND_TYPES as BACKEND_TYPES
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
)

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 10000


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """Merge file updates with support for deletions.

    This reducer enables file deletion by treating `None` values in the right
    dictionary as deletion markers. It's designed to work with LangGraph's
    state management where annotated reducers control how state updates merge.

    Args:
        left: Existing files dictionary. May be `None` during initialization.
        right: New files dictionary to merge. Files with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # Result: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    This function is designed for virtual filesystem paths and rejects
    Windows absolute paths (e.g., C:/..., F:/...) to maintain consistency
    and prevent path format ambiguity.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes. If provided,
            the normalized path must start with one of these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`), is a
            Windows absolute path (e.g., C:/...), or does not start with an
            allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Reject Windows absolute paths (e.g., C:\..., D:/...)
    # This maintains consistency in virtual filesystem paths
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""


# LIST_FILES_TOOL_DESCRIPTION = """Lists all files in the filesystem, filtering by directory.

# Usage:
# - The path parameter must be an absolute path, not a relative path
# - The list_files tool will return a list of all files in the specified directory.
# - This is very useful for exploring the file system and finding the right file to read or edit.
# - You should almost ALWAYS use this tool before using the Read or Edit tools."""
LIST_FILES_TOOL_DESCRIPTION = """列出文件系统中的所有文件，按目录过滤。

用法：
- 路径参数必须是绝对路径，而不是相对路径
- list_files 工具将返回指定目录中的所有文件列表。
- 这对于探索文件系统和找到要读取或编辑的正确文件非常有用。
- 在使用读取或编辑工具之前，几乎总是应该使用此工具。"""

# READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem. You can access any file directly by using this tool.
# Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

# Usage:
# - The file_path parameter must be an absolute path, not a relative path
# - By default, it reads up to 500 lines starting from the beginning of the file
# - **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
#   - First scan: read_file(path, limit=100) to see file structure
#   - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
#   - Only omit limit (read full file) when necessary for editing
# - Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
# - Any lines longer than 2000 characters will be truncated
# - Results are returned using cat -n format, with line numbers starting at 1
# - You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
# - If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
# - You should ALWAYS make sure a file has been read before editing it."""
READ_FILE_TOOL_DESCRIPTION = """从文件系统中读取文件。您可以通过使用此工具直接访问任何文件。
假设此工具能够读取机器上的所有文件。如果用户提供文件路径，则假定该路径有效。读取不存在的文件是可以的；将返回错误。

用法：
- file_path 参数必须是绝对路径，而不是相对路径
- 默认情况下，它从文件开头开始读取最多 10000 行
- **对于大文件和代码库探索很重要**：使用 offset 和 limit 参数进行分页以避免上下文溢出
  - 首次扫描：read_file(path, limit=10000) 查看文件结构
  - 读取更多部分：read_file(path, offset=10000, limit=2000) 读取接下来的 2000 行
  - 只有在需要编辑时才省略 limit（读取完整文件）
- 指定 offset 和 limit：read_file(path, offset=0, limit=10000) 读取前 10000 行
- 任何超过 2000 个字符的行将被截断
- 结果使用 cat -n 格式返回，行号从 1 开始
- 您有能力在单个响应中调用多个工具。批量推测性地读取多个可能有用的文件总是更好的选择。
- 如果您读取了一个存在但内容为空的文件，您将在文件内容位置收到系统提醒警告。
- 在编辑文件之前，您应该始终确保文件已被读取。"""

# EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

# Usage:
# - You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
# - When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
# - ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
# - Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
# - The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
# - Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""
EDIT_FILE_TOOL_DESCRIPTION = """在文件中执行精确的字符串替换。

用法：
- 在编辑之前，您必须在对话中至少使用一次 `Read` 工具。如果您在未读取文件的情况下尝试编辑，此工具将出错。
- 当从 Read 工具输出中编辑文本时，请确保保留与行号前缀之后显示的完全相同的缩进（制表符/空格）。行号前缀格式为：空格 + 行号 + 制表符。制表符之后的所有内容都是要匹配的实际文件内容。切勿在 old_string 或 new_string 中包含行号前缀的任何部分。
- 始终优先编辑现有文件。除非明确要求，否则永远不要写入新文件。
- 仅在用户明确要求时才使用表情符号。除非被要求，否则避免向文件添加表情符号。
- 如果 `old_string` 在文件中不是唯一的，编辑将失败。请提供一个更大的字符串以及更多的周围上下文使其唯一，或使用 `replace_all` 来更改 `old_string` 的每个实例。
- 对于在整个文件中替换和重命名字符串，请使用 `replace_all`。如果您想重命名变量，此参数很有用。"""


# WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

# Usage:
# - The file_path parameter must be an absolute path, not a relative path
# - The content parameter must be a string
# - The write_file tool will create the a new file.
# - Prefer to edit existing files over creating new ones when possible."""
WRITE_FILE_TOOL_DESCRIPTION = """在文件系统中创建新文件并写入内容。

用法：
- file_path 参数必须是绝对路径，不能是相对路径
- content 参数必须是字符串
- write_file 工具将会创建一个新文件。
- 当可能时，优先选择编辑现有文件而不是创建新文件。"""


# GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

# Usage:
# - The glob tool finds files by matching patterns with wildcards
# - Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character)
# - Patterns can be absolute (starting with `/`) or relative
# - Returns a list of absolute file paths that match the pattern

# Examples:
# - `**/*.py` - Find all Python files
# - `*.txt` - Find all text files in root
# - `/subdir/**/*.md` - Find all markdown files under /subdir"""
GLOB_TOOL_DESCRIPTION = """使用通配符模式查找匹配的文件。

用法：
- glob工具通过匹配包含通配符的模式来查找文件
- 支持标准的glob模式：`*`（任意字符）、`**`（任意目录）、`?`（单个字符）
- 模式可以是绝对路径（以`/`开头）或相对路径
- 返回匹配该模式的绝对文件路径列表

示例：
- `**/*.py` - 查找所有Python文件
- `*.txt` - 查找根目录下所有文本文件
- `/subdir/**/*.md` - 查找/subdir目录下的所有markdown文件"""

# GREP_TOOL_DESCRIPTION = """Search for a pattern in files.

# Usage:
# - The grep tool searches for text patterns across files
# - The pattern parameter is the text to search for (literal string, not regex)
# - The path parameter filters which directory to search in (default is the current working directory)
# - The glob parameter accepts a glob pattern to filter which files to search (e.g., `*.py`)
# - The output_mode parameter controls the output format:
#   - `files_with_matches`: List only file paths containing matches (default)
#   - `content`: Show matching lines with file path and line numbers
#   - `count`: Show count of matches per file

# Examples:
# - Search all files: `grep(pattern="TODO")`
# - Search Python files only: `grep(pattern="import", glob="*.py")`
# - Show matching lines: `grep(pattern="error", output_mode="content")`"""
GREP_TOOL_DESCRIPTION = """在文件中搜索模式。

用法：
- grep工具在多个文件中搜索文本模式
- pattern参数是要搜索的文本（字面字符串，非正则表达式）
- path参数过滤要在哪个目录中搜索（默认为当前工作目录）
- glob参数接受一个glob模式来过滤要搜索的文件（例如，`*.py`）
- output_mode参数控制输出格式：
  - `files_with_matches`：仅列出包含匹配项的文件路径（默认）
  - `content`：显示匹配行及文件路径和行号
  - `count`：显示每个文件的匹配次数

示例：
- 搜索所有文件：`grep(pattern="TODO")`
- 仅搜索Python文件：`grep(pattern="import", glob="*.py")`
- 显示匹配行：`grep(pattern="error", output_mode="content")`"""

# EXECUTE_TOOL_DESCRIPTION = """Executes a given command in the sandbox environment with proper handling and security measures.

# Before executing the command, please follow these steps:

# 1. Directory Verification:
#    - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
#    - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory

# 2. Command Execution:
#    - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
#    - Examples of proper quoting:
#      - cd "/Users/name/My Documents" (correct)
#      - cd /Users/name/My Documents (incorrect - will fail)
#      - python "/path/with spaces/script.py" (correct)
#      - python /path/with spaces/script.py (incorrect - will fail)
#    - After ensuring proper quoting, execute the command
#    - Capture the output of the command

# Usage notes:
#   - The command parameter is required
#   - Commands run in an isolated sandbox environment
#   - Returns combined stdout/stderr output with exit code
#   - If the output is very large, it may be truncated
#   - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
#   - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
#     - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
#     - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
#   - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

# Examples:
#   Good examples:
#     - execute(command="pytest /foo/bar/tests")
#     - execute(command="python /path/to/script.py")
#     - execute(command="npm install && npm test")

#   Bad examples (avoid these):
#     - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
#     - execute(command="cat file.txt")  # Use read_file tool instead
#     - execute(command="find . -name '*.py'")  # Use glob tool instead
#     - execute(command="grep -r 'pattern' .")  # Use grep tool instead

# Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
# If execution is not supported, the tool will return an error message."""
EXECUTE_TOOL_DESCRIPTION = """在沙箱环境中执行给定命令，并进行适当的处理和安全措施。

执行命令前，请遵循以下步骤：

1. 目录验证：
   - 如果命令将创建新目录或文件，首先使用ls工具验证父目录是否存在且位置正确
   - 例如，在运行"mkdir foo/bar"之前，先使用ls检查"foo"是否存在且是预期的父目录

2. 命令执行：
   - 始终用双引号引用包含空格的文件路径（例如，cd "path with spaces/file.txt"）
   - 正确引用的示例：
     - cd "/Users/name/My Documents"（正确）
     - cd /Users/name/My Documents（错误 - 将失败）
     - python "/path/with spaces/script.py"（正确）
     - python /path/with spaces/script.py（错误 - 将失败）
   - 确保正确引用后，执行命令
   - 捕获命令的输出

用法说明：
  - command参数是必需的
  - 命令在隔离的沙箱环境中运行
  - 返回组合的stdout/stderr输出和退出码
  - 如果输出很大，可能会被截断
  - 非常重要：您必须避免使用搜索命令如find和grep。而应使用grep、glob工具进行搜索。您必须避免使用读取工具如cat、head、tail，而应使用read_file读取文件
  - 发出多个命令时，使用';'或'&&'运算符分隔。不要使用换行符（换行符在引用字符串中是可以的）
    - 当命令相互依赖时使用'&&'（例如，"mkdir dir && cd dir"）
    - 仅当需要按顺序运行命令但不关心早期命令是否失败时才使用';'
  - 通过使用绝对路径并避免使用cd，尽量在整个会话中保持当前工作目录

示例：
  好的例子：
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")

  坏的例子（避免这些）：
    - execute(command="cd /foo/bar && pytest tests")  # 使用绝对路径代替
    - execute(command="cat file.txt")  # 使用read_file工具代替
    - execute(command="find . -name '*.py'")  # 使用glob工具代替
    - execute(command="grep -r 'pattern' .")  # 使用grep工具代替

注意：此工具仅在后端支持执行时可用（SandboxBackendProtocol）。
如果执行不受支持，工具将返回错误消息。"""

# FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

# You have access to a filesystem which you can interact with using these tools.
# All file paths must start with a /.

# Important guidelines for file operations:
# - Always use virtual paths starting with / (e.g., /file.txt)
# - The root path (/) maps to the current working directory
# - Use the ls command to explore the filesystem structure
# - Do not attempt to find parent directories; only search within the current working directory or use absolute paths to avoid unexpected results

# - ls: list files in a directory (requires absolute path)
# - read_file: read a file from the filesystem
# - write_file: write to a file in the filesystem
# - edit_file: edit a file in the filesystem
# - glob: find files matching a pattern (e.g., "**/*.py")
# - grep: search for text within files"""
FILESYSTEM_SYSTEM_PROMPT = """## 文件系统工具 `list_directory_tree`、`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`

您可以使用这些工具与文件系统进行交互。
所有文件路径必须以/开头。

文件操作的重要指南：
- 始终使用以/开头的虚拟路径（例如，/file.txt）
- 根路径(/)映射到当前工作目录
- 当需要查找文件时，使用list_directory_tree工具探索文件系统结构
- 不要尝试查找父目录；仅在当前工作目录内搜索或使用绝对路径以避免意外结果

- list_directory_tree: 列出当前工作目录的目录及文件树结构
- ls：列出目录中的文件（需要绝对路径）
- read_file：从文件系统读取文件
- write_file：向文件系统中的文件写入
- edit_file：编辑文件系统中的文件
- glob：查找符合模式的文件（例如，"**/*.py"）
- grep：在文件中搜索文本"""

# EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

# You have access to an `execute` tool for running shell commands in a sandboxed environment.
# Use this tool to run commands, scripts, tests, builds, and other shell operations.

# - execute: run a shell command in the sandbox (returns output and exit code)"""
EXECUTION_SYSTEM_PROMPT = """## 执行工具 `execute`

您可以使用`execute`工具在沙箱环境中运行shell命令。
使用此工具运行命令、脚本、测试、构建和其他shell操作。

- execute：在沙箱中运行shell命令（返回输出和退出码）"""


def _get_backend(backend: BACKEND_TYPES, runtime: ToolRuntime) -> BackendProtocol:
    """Get the resolved backend instance from backend or factory.

    Args:
        backend: Backend instance or factory function.
        runtime: The tool runtime context.

    Returns:
        Resolved backend instance.
    """
    if callable(backend):
        return backend(runtime)
    return backend


def _ls_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the ls (list files) tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured ls tool that lists files using the backend.
    """
    tool_description = custom_description or LIST_FILES_TOOL_DESCRIPTION

    def sync_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """Synchronous wrapper for ls tool."""
        resolved_backend = _get_backend(backend, runtime)
        validated_path = _validate_path(path)
        
        if os.getenv("DEBUG_FILE_SYSTEM") == "true":
          print(f"ls: {validated_path}")
        infos = resolved_backend.ls_info(validated_path)
        for fi in infos:
            if os.getenv("DEBUG_FILE_SYSTEM") == "true":
              print(f"ls infos: {fi}")
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """Asynchronous wrapper for ls tool."""
        resolved_backend = _get_backend(backend, runtime)
        validated_path = _validate_path(path)
        infos = await resolved_backend.als_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="ls",
        description=tool_description,
        func=sync_ls,
        coroutine=async_ls,
    )


def _read_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the read_file tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured read_file tool that reads files using the backend.
    """
    tool_description = custom_description or READ_FILE_TOOL_DESCRIPTION

    def sync_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """Synchronous wrapper for read_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        return resolved_backend.read(file_path, offset=offset, limit=limit)

    async def async_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """Asynchronous wrapper for read_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        return await resolved_backend.aread(file_path, offset=offset, limit=limit)

    return StructuredTool.from_function(
        name="read_file",
        description=tool_description,
        func=sync_read_file,
        coroutine=async_read_file,
    )


def _write_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the write_file tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured write_file tool that creates new files using the backend.
    """
    tool_description = custom_description or WRITE_FILE_TOOL_DESCRIPTION

    def sync_write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        """Synchronous wrapper for write_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: WriteResult = resolved_backend.write(file_path, content)
        if res.error:
            return res.error
        # If backend returns state update, wrap into Command with ToolMessage
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Updated file {res.path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Updated file {res.path}"

    async def async_write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        """Asynchronous wrapper for write_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: WriteResult = await resolved_backend.awrite(file_path, content)
        if res.error:
            return res.error
        # If backend returns state update, wrap into Command with ToolMessage
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Updated file {res.path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Updated file {res.path}"

    return StructuredTool.from_function(
        name="write_file",
        description=tool_description,
        func=sync_write_file,
        coroutine=async_write_file,
    )


def _edit_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the edit_file tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured edit_file tool that performs string replacements in files using the backend.
    """
    tool_description = custom_description or EDIT_FILE_TOOL_DESCRIPTION

    def sync_edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        """Synchronous wrapper for edit_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: EditResult = resolved_backend.edit(file_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    async def async_edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        """Asynchronous wrapper for edit_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: EditResult = await resolved_backend.aedit(file_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    return StructuredTool.from_function(
        name="edit_file",
        description=tool_description,
        func=sync_edit_file,
        coroutine=async_edit_file,
    )


def _glob_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the glob tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured glob tool that finds files by pattern using the backend.
    """
    tool_description = custom_description or GLOB_TOOL_DESCRIPTION

    def sync_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """Synchronous wrapper for glob tool."""
        resolved_backend = _get_backend(backend, runtime)
        infos = resolved_backend.glob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """Asynchronous wrapper for glob tool."""
        resolved_backend = _get_backend(backend, runtime)
        infos = await resolved_backend.aglob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="glob",
        description=tool_description,
        func=sync_glob,
        coroutine=async_glob,
    )


def _grep_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the grep tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured grep tool that searches for patterns in files using the backend.
    """
    tool_description = custom_description or GREP_TOOL_DESCRIPTION

    def sync_grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """Synchronous wrapper for grep tool."""
        resolved_backend = _get_backend(backend, runtime)
        raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    async def async_grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """Asynchronous wrapper for grep tool."""
        resolved_backend = _get_backend(backend, runtime)
        raw = await resolved_backend.agrep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    return StructuredTool.from_function(
        name="grep",
        description=tool_description,
        func=sync_grep,
        coroutine=async_grep,
    )


def _supports_execution(backend: BackendProtocol) -> bool:
    """Check if a backend supports command execution.

    For CompositeBackend, checks if the default backend supports execution.
    For other backends, checks if they implement SandboxBackendProtocol.

    Args:
        backend: The backend to check.

    Returns:
        True if the backend supports execution, False otherwise.
    """
    # Import here to avoid circular dependency
    from deepagents.backends.composite import CompositeBackend

    # For CompositeBackend, check the default backend
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)

    # For other backends, use isinstance check
    return isinstance(backend, SandboxBackendProtocol)


def _execute_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the execute tool for sandbox command execution.

    Args:
        backend: Backend to use for execution, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured execute tool that runs commands if backend supports SandboxBackendProtocol.
    """
    tool_description = custom_description or EXECUTE_TOOL_DESCRIPTION

    def sync_execute(
        command: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Synchronous wrapper for execute tool."""
        resolved_backend = _get_backend(backend, runtime)

        # Runtime check - fail gracefully if not supported
        if not _supports_execution(resolved_backend):
            return (
                "Error: Execution not available. This agent's backend "
                "does not support command execution (SandboxBackendProtocol). "
                "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
            )

        try:
            result = resolved_backend.execute(command)
        except NotImplementedError as e:
            # Handle case where execute() exists but raises NotImplementedError
            return f"Error: Execution not available. {e}"

        # Format output for LLM consumption
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)

    async def async_execute(
        command: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Asynchronous wrapper for execute tool."""
        resolved_backend = _get_backend(backend, runtime)

        # Runtime check - fail gracefully if not supported
        if not _supports_execution(resolved_backend):
            return (
                "Error: Execution not available. This agent's backend "
                "does not support command execution (SandboxBackendProtocol). "
                "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
            )

        try:
            result = await resolved_backend.aexecute(command)
        except NotImplementedError as e:
            # Handle case where execute() exists but raises NotImplementedError
            return f"Error: Execution not available. {e}"

        # Format output for LLM consumption
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)

    return StructuredTool.from_function(
        name="execute",
        description=tool_description,
        func=sync_execute,
        coroutine=async_execute,
    )


TOOL_GENERATORS = {
    "ls": _ls_tool_generator,
    "read_file": _read_file_tool_generator,
    "write_file": _write_file_tool_generator,
    "edit_file": _edit_file_tool_generator,
    "glob": _glob_tool_generator,
    "grep": _grep_tool_generator,
    "execute": _execute_tool_generator,
}


def _get_filesystem_tools(
    backend: BackendProtocol,
    custom_tool_descriptions: dict[str, str] | None = None,
) -> list[BaseTool]:
    """Get filesystem and execution tools.

    Args:
        backend: Backend to use for file storage and optional execution, or a factory function that takes runtime and returns a backend.
        custom_tool_descriptions: Optional custom descriptions for tools.

    Returns:
        List of configured tools: ls, read_file, write_file, edit_file, glob, grep, execute.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []

    print(f"Generating filesystem tools {custom_tool_descriptions}")
    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
        tools.append(tool)
    return tools


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}
You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.
You can do this by specifying an offset and limit in the read_file tool call.
For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here are the first 10 lines of the result:
{content_sample}
"""


class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem and optional execution tools to an agent.

    This middleware adds filesystem tools to the agent: ls, read_file, write_file,
    edit_file, glob, and grep. Files can be stored using any backend that implements
    the BackendProtocol.

    If the backend implements SandboxBackendProtocol, an execute tool is also added
    for running shell commands.

    Args:
        backend: Backend for file storage and optional execution. If not provided, defaults to StateBackend
            (ephemeral storage in agent state). For persistent storage or hybrid setups,
            use CompositeBackend with custom routes. For execution support, use a backend
            that implements SandboxBackendProtocol.
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # Ephemeral storage only (default, no execution)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # With hybrid storage (ephemeral + persistent /memories/)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # With sandbox backend (supports execution)
        from my_sandbox import DockerSandboxBackend

        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            backend: Backend for file storage and optional execution, or a factory callable.
                Defaults to StateBackend if not provided.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
            tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.
        """
        self.tool_token_limit_before_evict = tool_token_limit_before_evict

        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # Set system prompt (allow full override or None to generate dynamically)
        self._custom_system_prompt = system_prompt

        self.tools = _get_filesystem_tools(self.backend, custom_tool_descriptions)

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)
            # print(f"FilesystemMiddleware wrap_model_call system_prompt: {system_prompt}")

        if system_prompt:
            request = request.override(system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)
            # print(f"FilesystemMiddleware wrap_model_call system_prompt: {system_prompt}")

        if system_prompt:
            request = request.override(system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt)

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        content = message.content
        if not isinstance(content, str) or len(content) <= 4 * self.tool_token_limit_before_evict:
            return message, None

        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content)
        if result.error:
            return message, None
        content_sample = format_content_with_line_numbers([line[:1000] for line in content.splitlines()[:10]], start_line=1)
        processed_message = ToolMessage(
            TOO_LARGE_TOOL_MSG.format(
                tool_call_id=message.tool_call_id,
                file_path=file_path,
                content_sample=content_sample,
            ),
            tool_call_id=message.tool_call_id,
        )
        return processed_message, result.files_update

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        if isinstance(tool_result, ToolMessage) and isinstance(tool_result.content, str):
            if not (self.tool_token_limit_before_evict and len(tool_result.content) > 4 * self.tool_token_limit_before_evict):
                return tool_result
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not (
                    self.tool_token_limit_before_evict
                    and isinstance(message, ToolMessage)
                    and isinstance(message.content, str)
                    and len(message.content) > 4 * self.tool_token_limit_before_evict
                ):
                    processed_messages.append(message)
                    continue
                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})

        return tool_result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        # 打印工具调用详情
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]
        if os.getenv("DEBUG_FILE_SYSTEM") == "true":
            print(f"即将调用异步工具: {tool_name}")
            print(f"工具参数: {tool_args}")

        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
          result = await handler(request)
          if os.getenv("DEBUG_FILE_SYSTEM") == "true":
            print(f"异步工具 {tool_name} 返回结果: {result}")
          return result

        tool_result = await handler(request)
        if os.getenv("DEBUG_FILE_SYSTEM") == "true":
            # 打印工具调用结果
            print(f"异步工具 {tool_name} 返回结果: {tool_result}")

        return self._intercept_large_tool_result(tool_result, request.runtime)
