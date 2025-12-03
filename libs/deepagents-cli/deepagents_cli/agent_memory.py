"""Middleware for loading agent-specific long-term memory into the system prompt."""

import contextlib
from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from deepagents_cli.config import Settings


class AgentMemoryState(AgentState):
    """State for the agent memory middleware."""

    user_memory: NotRequired[str]
    """Personal preferences from ~/.deepagents/{agent}/ (applies everywhere)."""

    project_memory: NotRequired[str]
    """Project-specific context (loaded from project root)."""


class AgentMemoryStateUpdate(TypedDict):
    """A state update for the agent memory middleware."""

    user_memory: NotRequired[str]
    """Personal preferences from ~/.deepagents/{agent}/ (applies everywhere)."""

    project_memory: NotRequired[str]
    """Project-specific context (loaded from project root)."""


# Long-term Memory Documentation
# Note: Claude Code loads CLAUDE.md files hierarchically and combines them (not precedence-based):
# - Loads recursively from cwd up to (but not including) root directory
# - Multiple files are combined hierarchically: enterprise → project → user
# - Both [project-root]/CLAUDE.md and [project-root]/.claude/CLAUDE.md are loaded if both exist
# - Files higher in hierarchy load first, providing foundation for more specific memories
# We will follow that pattern for deepagents-cli
# LONGTERM_MEMORY_SYSTEM_PROMPT = """

# ## Long-term Memory

# Your long-term memory is stored in files on the filesystem and persists across sessions.

# **User Memory Location**: `{agent_dir_absolute}` (displays as `{agent_dir_display}`)
# **Project Memory Location**: {project_memory_info}

# Your system prompt is loaded from TWO sources at startup:
# 1. **User agent.md**: `{agent_dir_absolute}/agent.md` - Your personal preferences across all projects
# 2. **Project agent.md**: Loaded from project root if available - Project-specific instructions

# Project-specific agent.md is loaded from these locations (both combined if both exist):
# - `[project-root]/.deepagents/agent.md` (preferred)
# - `[project-root]/agent.md` (fallback, but also included if both exist)

# **When to CHECK/READ memories (CRITICAL - do this FIRST):**
# - **At the start of ANY new session**: Check both user and project memories
#   - User: `ls {agent_dir_absolute}`
#   - Project: `ls {project_deepagents_dir}` (if in a project)
# - **BEFORE answering questions**: If asked "what do you know about X?" or "how do I do Y?", check project memories FIRST, then user
# - **When user asks you to do something**: Check if you have project-specific guides or examples
# - **When user references past work**: Search project memory files for related context

# **Memory-first response pattern:**
# 1. User asks a question → Check project directory first: `ls {project_deepagents_dir}`
# 2. If relevant files exist → Read them with `read_file '{project_deepagents_dir}/[filename]'`
# 3. Check user memory if needed → `ls {agent_dir_absolute}`
# 4. Base your answer on saved knowledge supplemented by general knowledge

# **When to update memories:**
# - **IMMEDIATELY when the user describes your role or how you should behave**
# - **IMMEDIATELY when the user gives feedback on your work** - Update memories to capture what was wrong and how to do it better
# - When the user explicitly asks you to remember something
# - When patterns or preferences emerge (coding styles, conventions, workflows)
# - After significant work where context would help in future sessions

# **Learning from feedback:**
# - When user says something is better/worse, capture WHY and encode it as a pattern
# - Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions
# - When user says "you should remember X" or "be careful about Y", treat this as HIGH PRIORITY - update memories IMMEDIATELY
# - Look for the underlying principle behind corrections, not just the specific mistake

# ## Deciding Where to Store Memory

# When writing or updating agent memory, decide whether each fact, configuration, or behavior belongs in:

# ### User Agent File: `{agent_dir_absolute}/agent.md`
# → Describes the agent's **personality, style, and universal behavior** across all projects.

# **Store here:**
# - Your general tone and communication style
# - Universal coding preferences (formatting, comment style, etc.)
# - General workflows and methodologies you follow
# - Tool usage patterns that apply everywhere
# - Personal preferences that don't change per-project

# **Examples:**
# - "Be concise and direct in responses"
# - "Always use type hints in Python"
# - "Prefer functional programming patterns"

# ### Project Agent File: `{project_deepagents_dir}/agent.md`
# → Describes **how this specific project works** and **how the agent should behave here only.**

# **Store here:**
# - Project-specific architecture and design patterns
# - Coding conventions specific to this codebase
# - Project structure and organization
# - Testing strategies for this project
# - Deployment processes and workflows
# - Team conventions and guidelines

# **Examples:**
# - "This project uses FastAPI with SQLAlchemy"
# - "Tests go in tests/ directory mirroring src/ structure"
# - "All API changes require updating OpenAPI spec"

# ### Project Memory Files: `{project_deepagents_dir}/*.md`
# → Use for **project-specific reference information** and structured notes.

# **Store here:**
# - API design documentation
# - Architecture decisions and rationale
# - Deployment procedures
# - Common debugging patterns
# - Onboarding information

# **Examples:**
# - `{project_deepagents_dir}/api-design.md` - REST API patterns used
# - `{project_deepagents_dir}/architecture.md` - System architecture overview
# - `{project_deepagents_dir}/deployment.md` - How to deploy this project

# ### File Operations:

# **User memory:**
# ```
# ls {agent_dir_absolute}                              # List user memory files
# read_file '{agent_dir_absolute}/agent.md'            # Read user preferences
# edit_file '{agent_dir_absolute}/agent.md' ...        # Update user preferences
# ```

# **Project memory (preferred for project-specific information):**
# ```
# ls {project_deepagents_dir}                          # List project memory files
# read_file '{project_deepagents_dir}/agent.md'        # Read project instructions
# edit_file '{project_deepagents_dir}/agent.md' ...    # Update project instructions
# write_file '{project_deepagents_dir}/agent.md' ...  # Create project memory file
# ```

# **Important**:
# - Project memory files are stored in `.deepagents/` inside the project root
# - Always use absolute paths for file operations
# - Check project memories BEFORE user when answering project-specific questions"""
LONGTERM_MEMORY_SYSTEM_PROMPT = """
## 长期记忆

你的长期记忆存储在文件系统中的文件里，并且在会话之间保持不变。
记忆分为两类：
1. 系统提示词（又分为用户系统提示词和项目系统提示词）
2. 除了系统提示词以外的各种主题知识，简称记忆（又分为用户记忆和项目记忆）

### 系统提示词的位置

**用户系统提示词位置**: `{agent_dir_absolute}` (显示为 `{agent_dir_display}`)
**项目系统提示词位置**: {project_memory_info}

你的系统提示在启动时从两个来源加载：
1. **用户 agent.md**: `{agent_dir_absolute}/agent.md` - 你在所有项目中的个人偏好
2. **项目 agent.md**: 如果可用，则从项目根目录加载 - 项目特定的系统提示词

项目特定的 agent.md 从以下位置加载（如果两者都存在则合并）：
- `[project-root]/.deepagents/agent.md` (首选)
- `[project-root]/agent.md` (后备，但如果两者都存在也会包含)

### 除了系统提示词以外的各种主题知识的位置，简称记忆

**用户记忆位置**: `{agent_dir_absolute}/memories/`
**项目记忆位置**: `{project_deepagents_dir}/memories/`

**何时检查/读取记忆（关键 - 首先执行此操作）：**
- **任何新会话开始时**：首先检查用户和项目记忆
  - 用户：`ls {agent_dir_absolute}/memories/`
  - 项目：`ls {project_deepagents_dir}/memories/` (如果在项目中)
- **回答问题之前**：如果被问到"你对X了解什么？"或"我如何做Y？"，首先检查项目记忆，然后是用户记忆
- **当用户要求你做某事时**：检查是否有项目特定的指南或示例
- **当用户引用过去的工作时**：搜索项目记忆文件以获取相关上下文

**基于记忆的响应模式：**
1. 用户提出问题 → 首先检查项目目录：`ls {project_deepagents_dir}/memories/`
2. 如果存在相关文件 → 使用 `read_file '{project_deepagents_dir}/memories/[filename]'` 读取它们
3. 如有需要检查用户记忆 → `ls {agent_dir_absolute}/memories/`
4. 基于已保存的知识并结合通用知识来回答

**何时更新系统提示词或记忆：**
- **当用户描述你的角色或行为方式时立即更新系统提示词**
- **当用户对你的工作提供反馈时立即更新系统提示词** - 更新系统提示词以捕捉错误之处以及如何做得更好
- 当用户明确要求你记住某些事情时，更新用户记忆或项目记忆
- 当出现模式或偏好时（编码风格、约定、工作流程）更新用户系统提示词或项目系统提示词
- 在重要工作之后，这些上下文将在未来的会话中有帮助时

**从反馈中学习：**
- 当用户说某件事更好/更差时，捕捉原因并将其编码为模式
- 每次纠正都是永久改进的机会 - 不要只修复眼前的问题，还要更新你的指导原则
- 当用户说"你应该记住X"或"注意Y"时，将其视为高优先级 - 立即更新记忆
- 寻找纠正背后的底层原理，而不仅仅是具体错误

## 决定将提示词或记忆存储在哪里

在编写或更新系统提示词或记忆时，决定每个事实、配置或行为属于何处：

### 用户系统提示词: `{agent_dir_absolute}/agent.md`
→ 描述agent在所有项目中的**个性、风格和普遍行为**。

**存储在这里：**
- 你的通用语调和沟通风格
- 普遍的编码偏好（格式化、注释风格等）
- 通用的工作流程和方法论
- 到处适用的工具使用模式
- 不因项目而改变的个人偏好

**示例：**
- "在回应中保持简洁直接"
- "始终在Python中使用类型提示"
- "更喜欢函数式编程模式"

### 项目系统提示词: `{project_deepagents_dir}/agent.md`
→ 描述**这个特定项目如何工作**以及**agent在此项目中应如何表现**。

**存储在这里：**
- 项目特定的架构和设计模式
- 此代码库特有的编码约定
- 项目结构和组织
- 此项目的测试策略
- 部署过程和工作流程
- 团队约定和指南

**示例：**
- "此项目使用FastAPI与SQLAlchemy"
- "测试文件放在tests/目录中，与src/结构对应"
- "所有API更改都需要更新OpenAPI规范"

### 除了系统提示词以外的项目级主题知识，也就是项目记忆: `{project_deepagents_dir}/memories/*.md`
→ 用于**项目特定的参考信息**和结构化笔记。

**存储在这里：**
- API设计文档
- 架构决策和理由
- 部署程序
- 常见调试模式

**示例：**
- `{project_deepagents_dir}/memories/api-design.md` - 使用的REST API模式
- `{project_deepagents_dir}/memories/architecture.md` - 系统架构概述
- `{project_deepagents_dir}/memories/deployment.md` - 如何部署此项目

### 除了系统提示词以外的用户级主题知识，也就是用户记忆: `{agent_dir_absolute}/memories/*.md`
→ 用于**用户特定的参考信息**和结构化笔记。

**存储在这里：**
- 用户的个人总结
- 用户行为偏好
- 用户的个人信息

**示例：**
- `{agent_dir_absolute}/memories/summary_about_python_dev.md` - 关于python开发的总结
- `{agent_dir_absolute}/memories/my_preferences.md` - 我的偏好
- `{agent_dir_absolute}/memories/personal_info.md` - 我的个人信息

### 文件操作：

**用户记忆：**
```
ls {agent_dir_absolute}/memories/ # 列出用户记忆文件
read_file '{agent_dir_absolute}/memories/*.md' # 读取用户记忆文件
edit_file '{agent_dir_absolute}/memories/*.md' ... # 更新记忆文件
```


**项目记忆（项目特定信息的首选）：**
```
ls {project_deepagents_dir}/memories/                     # 列出项目记忆文件
read_file '{project_deepagents_dir}/memories/*.md'        # 读取项目记忆文件
edit_file '{project_deepagents_dir}/memories/*.md' ...    # 更新项目记忆文件
write_file '{project_deepagents_dir}/memories/*.md' ...   # 创建项目记忆文件
```

**重要事项**：
- 项目记忆文件存储在项目根目录内的 `.deepagents/memories/` 中
- 文件操作始终使用绝对路径
- 回答项目特定问题时首先检查项目记忆"""


DEFAULT_MEMORY_SNIPPET = """<user_prompt>
{user_memory}
</user_prompt>

<project_prompt>
{project_memory}
</project_prompt>"""


class AgentMemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent-specific long-term memory.

    This middleware loads the agent's long-term memory from a file (agent.md)
    and injects it into the system prompt. The memory is loaded once at the
    start of the conversation and stored in state.
    """

    state_schema = AgentMemoryState

    def __init__(
        self,
        *,
        settings: Settings,
        assistant_id: str,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the agent memory middleware.

        Args:
            settings: Global settings instance with project detection and paths.
            assistant_id: The agent identifier.
            system_prompt_template: Optional custom template for injecting
                agent memory into system prompt.
        """
        self.settings = settings
        self.assistant_id = assistant_id
        print(f"AgentMemoryMiddleware assistant_id: {assistant_id}")

        # User paths
        self.agent_dir = settings.get_agent_dir(assistant_id)
        print(f"AgentMemoryMiddleware agent_dir: {self.agent_dir}")

        # Store both display path (with ~) and absolute path for file operations
        self.agent_dir_display = f"~/.deepagents/{assistant_id}"
        self.agent_dir_absolute = str(self.agent_dir)
        print(f"AgentMemoryMiddleware agent_dir_display: {self.agent_dir_display}")
        print(f"AgentMemoryMiddleware agent_dir_absolute: {self.agent_dir_absolute}")
        
        # Project paths (from settings)
        self.project_root = settings.project_root
        print(f"AgentMemoryMiddleware project_root: {self.project_root}")

        print(f"AgentMemoryMiddleware system_prompt_template: {len(system_prompt_template) if system_prompt_template else 'None'}")
        self.system_prompt_template = system_prompt_template or DEFAULT_MEMORY_SNIPPET

    def before_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """Load agent memory from file before agent execution.

        Loads both user agent.md and project-specific agent.md if available.
        Only loads if not already present in state.

        Dynamically checks for file existence on every call to catch user updates.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with user_memory and project_memory populated.
        """
        result: AgentMemoryStateUpdate = {}

        # Load user memory if not already in state
        if "user_memory" not in state:
            user_path = self.settings.get_user_agent_md_path(self.assistant_id)
            # print(f"Loading user memory from {user_path}")
            if user_path.exists():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    result["user_memory"] = user_path.read_text(encoding="utf-8")

        # Load project memory if not already in state
        if "project_memory" not in state:
            project_path = self.settings.get_project_agent_md_path()
            # print(f"Loading project memory from {project_path}")
            if project_path and project_path.exists():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    result["project_memory"] = project_path.read_text(encoding="utf-8")

        return result

    def _build_system_prompt(self, request: ModelRequest) -> str:
        """Build the complete system prompt with memory sections.

        Args:
            request: The model request containing state and base system prompt.

        Returns:
            Complete system prompt with memory sections injected.
        """
        # Extract memory from state
        state = cast("AgentMemoryState", request.state)
        user_memory = state.get("user_memory")
        project_memory = state.get("project_memory")
        base_system_prompt = request.system_prompt

        # Build project memory info for documentation
        if self.project_root and project_memory:
            project_memory_info = f"`{self.project_root}` (detected)"
        elif self.project_root:
            project_memory_info = f"`{self.project_root}` (no agent.md found)"
        else:
            project_memory_info = "None (not in a git project)"
        # print(f"AgentMemoryMiddleware Project memory info: {project_memory_info}")
        

        # Build project deepagents directory path
        if self.project_root:
            project_deepagents_dir = str(self.project_root / ".deepagents")
        else:
            project_deepagents_dir = "[project-root]/.deepagents (not in a project)"
        # print(f"AgentMemoryMiddleware Project deepagents dir: {project_deepagents_dir}")

        # Format memory section with both memories
        memory_section = self.system_prompt_template.format(
            user_memory=user_memory if user_memory else "(No user agent.md)",
            project_memory=project_memory if project_memory else "(No project agent.md)",
        )

        system_prompt = "(这一行是调试信息，忽略)来自：AgentMemoryMiddleware system_prompt_template 或 DEFAULT_MEMORY_SNIPPET\n\n" + memory_section

        if base_system_prompt:
            system_prompt += "\n\n(这一行是调试信息，忽略)来自：AgentMemoryMiddleware request.system_prompt\n\n" + base_system_prompt

        # # 处理\.写入文件后变成.
        # project_deepagents_dir_resovled = project_deepagents_dir.replace("\\.", "\\\\.")
        # print(f"AgentMemoryMiddleware project_deepagents_dir_resovled: {project_deepagents_dir_resovled}")

        system_prompt += "\n\n(这一行是调试信息，忽略)来自：AgentMemoryMiddleware LONGTERM_MEMORY_SYSTEM_PROMPT\n\n" + LONGTERM_MEMORY_SYSTEM_PROMPT.format(
            agent_dir_absolute=self.agent_dir_absolute,
            agent_dir_display=self.agent_dir_display,
            project_memory_info=project_memory_info,
            # project_deepagents_dir=project_deepagents_dir_resovled,
            project_deepagents_dir=project_deepagents_dir
        )

        return system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject agent memory into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        system_prompt = self._build_system_prompt(request)
        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject agent memory into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        system_prompt = self._build_system_prompt(request)
        return await handler(request.override(system_prompt=system_prompt))
