"""Agent management and creation for the CLI."""

import os
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.sandbox import SandboxBackendProtocol
from langchain.agents.middleware import (
    InterruptOnConfig,
)
from langchain.agents.middleware.types import AgentState
from langchain.messages import ToolCall
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from deepagents_cli.agent_memory import AgentMemoryMiddleware
from deepagents_cli.config import COLORS, config, console, get_default_coding_instructions, settings
from deepagents_cli.integrations.sandbox_factory import get_default_working_dir
from deepagents_cli.shell import ShellMiddleware
from deepagents_cli.skills import SkillsMiddleware


def list_agents() -> None:
    """List all available agents."""
    agents_dir = settings.user_deepagents_dir

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"

            if agent_md.exists():
                console.print(f"  • [bold]{agent_name}[/bold]", style=COLORS["primary"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(
                    f"  • [bold]{agent_name}[/bold] [dim](incomplete)[/dim]", style=COLORS["tool"]
                )
                console.print(f"    {agent_path}", style=COLORS["dim"])

    console.print()


def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """Reset an agent to default or copy from another agent."""
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "agent.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found "
                "or has no agent.md"
            )
            return

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"])

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)

    console.print(f"✓ Agent '{agent_name}' reset to {action_desc}", style=COLORS["primary"])
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def get_system_prompt(assistant_id: str, sandbox_type: str | None = None) -> str:
    """获取代理的基本系统提示。

    参数:
        assistant_id: 代理标识符，用于路径引用
        sandbox_type: 沙箱提供者类型 ("modal", "runloop", "daytona")。
                     如果为None，则代理在本地模式下运行。

    返回:
        系统提示字符串（不包含agent.md内容）
    """
    agent_dir_path = f"~/.deepagents/{assistant_id}"

    if sandbox_type:
        # 获取提供者特定的工作目录

        working_dir = get_default_working_dir(sandbox_type)

        working_dir_section = f"""### 当前工作目录

您正在 `{working_dir}` 的**远程Linux沙箱**中操作。

所有代码执行和文件操作都在此沙箱环境中进行。

**重要:**
- CLI在用户本地机器上运行，但您在远程执行代码
- 对所有操作使用 `{working_dir}` 作为您的工作目录

"""
    else:
        cwd = Path.cwd()
        working_dir_section = f"""<env>
工作目录: {cwd}
</env>

### 当前工作目录

文件系统后端当前在: `{cwd}` 中运行

### 文件系统和路径

**重要 - 路径处理:**
- 对于文件系统工具（ls、read_file等），请使用以 / 开头的虚拟路径
- 根路径 (/) 映射到当前工作目录 (`{cwd}`)
- 示例: 要访问工作目录中的文件，使用 `/file.txt`
- 永远不要使用相对路径
- 可以使用 ls 命令探索文件系统结构
- 在 Windows 系统上，本地文件通过虚拟路径访问，例如：`/test.txt` 对应 `{cwd}\\test.txt`

"""

    return (
        working_dir_section
        + f"""### 技能目录

您的技能存储在: `{agent_dir_path}/skills/`
技能可能包含脚本或支持文件。执行技能脚本时使用真实文件系统路径:
示例: `bash python {agent_dir_path}/skills/web-research/script.py`

### 人类参与工具审批

某些工具调用需要用户批准才能执行。当用户拒绝工具调用时:
1. 立即接受他们的决定 - 不要重试相同命令
2. 解释您理解他们拒绝了该操作
3. 提供替代方法或询问澄清
4. 永远不要再尝试完全相同的被拒绝命令

尊重用户的决定并与他们协作。

### 网络搜索工具使用

当您使用web_search工具时:
1. 工具将返回带有标题、URL和内容摘录的搜索结果
2. 您必须阅读并处理这些结果，然后自然地回应用户
3. 永远不要直接向用户显示原始JSON或工具结果
4. 将来自多个来源的信息综合成连贯的答案
5. 必要时通过提及页面标题或URL来引用来源
6. 如果搜索没有找到所需内容，解释您找到了什么并询问澄清问题

用户只能看到您的文本回复 - 不是工具结果。使用web_search后始终提供完整、自然语言的答案。

### 待办事项列表管理

当使用write_todos工具时:
1. 保持待办事项列表最小化 - 最多3-6个项目
2. 仅为真正需要跟踪的复杂、多步骤任务创建待办事项
3. 将工作分解为清晰、可操作的项目，避免过度细分
4. 对于简单任务（1-2步），直接执行而不创建待办事项
5. 首次为任务创建待办事项列表时，务必询问用户计划是否良好后再开始工作
   - 创建待办事项，让它们渲染，然后询问："这个计划看起来好吗？"或类似问题
   - 等待用户回应后再将第一个待办事项标记为in_progress
   - 如果他们想要更改，请相应调整计划
6. 在完成每个项目后及时更新待办事项状态

待办事项列表是一个规划工具 - 明智地使用它以避免用过多的任务跟踪压倒用户。"""
    )


def _format_write_file_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format write_file tool call for approval prompt."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    content = args.get("content", "")

    action = "Overwrite" if Path(file_path).exists() else "Create"
    line_count = len(content.splitlines())

    return f"File: {file_path}\nAction: {action} file\nLines: {line_count}"


def _format_edit_file_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format edit_file tool call for approval prompt."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    replace_all = bool(args.get("replace_all", False))

    return (
        f"File: {file_path}\n"
        f"Action: Replace text ({'all occurrences' if replace_all else 'single occurrence'})"
    )


def _format_web_search_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format web_search tool call for approval prompt."""
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return f"Query: {query}\nMax results: {max_results}\n\n⚠️  This will use Tavily API credits"


def _format_fetch_url_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format fetch_url tool call for approval prompt."""
    args = tool_call["args"]
    url = args.get("url", "unknown")
    timeout = args.get("timeout", 30)

    return f"URL: {url}\nTimeout: {timeout}s\n\n⚠️  Will fetch and convert web content to markdown"


def _format_task_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format task (subagent) tool call for approval prompt.

    The task tool signature is: task(description: str, subagent_type: str)
    The description contains all instructions that will be sent to the subagent.
    """
    args = tool_call["args"]
    description = args.get("description", "unknown")
    subagent_type = args.get("subagent_type", "unknown")

    # Truncate description if too long for display
    description_preview = description
    if len(description) > 500:
        description_preview = description[:500] + "..."

    return (
        f"Subagent Type: {subagent_type}\n\n"
        f"Task Instructions:\n"
        f"{'─' * 40}\n"
        f"{description_preview}\n"
        f"{'─' * 40}\n\n"
        f"⚠️  Subagent will have access to file operations and shell commands"
    )


def _format_shell_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format shell tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "N/A")
    return f"Shell Command: {command}\nWorking Directory: {Path.cwd()}"


def _format_execute_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format execute tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "N/A")
    return f"Execute Command: {command}\nLocation: Remote Sandbox"


def _add_interrupt_on() -> dict[str, InterruptOnConfig]:
    """Configure human-in-the-loop interrupt_on settings for destructive tools."""
    shell_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_shell_description,
    }

    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,
    }
    return {
        "shell": shell_interrupt_config,
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
    }


def create_agent_with_config(
    model: str | BaseChatModel,
    assistant_id: str,
    tools: list[BaseTool],
    *,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
) -> tuple[Pregel, CompositeBackend]:
    """Create and configure an agent with the specified model and tools.

    Args:
        model: LLM model to use
        assistant_id: Agent identifier for memory storage
        tools: Additional tools to provide to agent
        sandbox: Optional sandbox backend for remote execution (e.g., ModalBackend).
                 If None, uses local filesystem + shell.
        sandbox_type: Type of sandbox provider ("modal", "runloop", "daytona")

    Returns:
        2-tuple of graph and backend
    """
    # Setup agent directory for persistent memory (same for both local and remote modes)
    agent_dir = settings.ensure_agent_dir(assistant_id)
    agent_md = agent_dir / "agent.md"
    if not agent_md.exists():
        source_content = get_default_coding_instructions()
        agent_md.write_text(source_content)

    # Skills directory - per-agent (user-level)
    skills_dir = settings.ensure_user_skills_dir(assistant_id)

    # Project-level skills directory (if in a project)
    project_skills_dir = settings.get_project_skills_dir()

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        # Backend: Local filesystem for code (no virtual routes)
        composite_backend = CompositeBackend(
            default=FilesystemBackend(root_dir=".", virtual_mode=True),  # Current working directory
            routes={},  # No virtualization - use real paths
        )

        # Middleware: AgentMemoryMiddleware, SkillsMiddleware, ShellToolMiddleware
        agent_middleware = [
            AgentMemoryMiddleware(settings=settings, assistant_id=assistant_id),
            SkillsMiddleware(
                skills_dir=skills_dir,
                assistant_id=assistant_id,
                project_skills_dir=project_skills_dir,
            ),
            ShellMiddleware(
                workspace_root=str(Path.cwd()),
                env=os.environ,
            ),
        ]
    else:
        # ========== REMOTE SANDBOX MODE ==========
        # Backend: Remote sandbox for code (no /memories/ route needed with filesystem-based memory)
        composite_backend = CompositeBackend(
            default=sandbox,  # Remote sandbox (ModalBackend, etc.)
            routes={},  # No virtualization
        )

        # Middleware: AgentMemoryMiddleware and SkillsMiddleware
        # NOTE: File operations (ls, read, write, edit, glob, grep) and execute tool
        # are automatically provided by create_deep_agent when backend is a SandboxBackend.
        agent_middleware = [
            AgentMemoryMiddleware(settings=settings, assistant_id=assistant_id),
            SkillsMiddleware(
                skills_dir=skills_dir,
                assistant_id=assistant_id,
                project_skills_dir=project_skills_dir,
            ),
        ]

    # Get the system prompt (sandbox-aware and with skills)
    system_prompt = get_system_prompt(assistant_id=assistant_id, sandbox_type=sandbox_type)

    interrupt_on = _add_interrupt_on()

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
    ).with_config(config)

    agent.checkpointer = InMemorySaver()

    return agent, composite_backend
