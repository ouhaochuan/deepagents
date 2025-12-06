"""Middleware for loading and exposing agent skills to the system prompt.

This middleware implements Anthropic's "Agent Skills" pattern with progressive disclosure:
1. Parse YAML frontmatter from SKILL.md files at session start
2. Inject skills metadata (name + description) into system prompt
3. Agent reads full SKILL.md content when relevant to a task

Skills directory structure (per-agent + project):
User-level: ~/.deepagents/{AGENT_NAME}/skills/
Project-level: {PROJECT_ROOT}/.deepagents/skills/

Example structure:
~/.deepagents/{AGENT_NAME}/skills/
├── web-research/
│   ├── SKILL.md        # Required: YAML frontmatter + instructions
│   └── helper.py       # Optional: supporting files
├── code-review/
│   ├── SKILL.md
│   └── checklist.md

.deepagents/skills/
├── project-specific/
│   └── SKILL.md        # Project-specific skills
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from deepagents_cli.skills.load import SkillMetadata, list_skills


class SkillsState(AgentState):
    """State for the skills middleware."""

    skills_metadata: NotRequired[list[SkillMetadata]]
    """List of loaded skill metadata (name, description, path)."""


class SkillsStateUpdate(TypedDict):
    """State update for the skills middleware."""

    skills_metadata: list[SkillMetadata]
    """List of loaded skill metadata (name, description, path)."""


# Skills System Documentation
# SKILLS_SYSTEM_PROMPT = """

# ## Skills System

# You have access to a skills library that provides specialized capabilities and domain knowledge.

# {skills_locations}

# **Available Skills:**

# {skills_list}

# **How to Use Skills (Progressive Disclosure):**

# Skills follow a **progressive disclosure** pattern - you know they exist (name + description above), but you only read the full instructions when needed:

# 1. **Recognize when a skill applies**: Check if the user's task matches any skill's description
# 2. **Read the skill's full instructions**: The skill list above shows the exact path to use with read_file
# 3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
# 4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs - use absolute paths

# **When to Use Skills:**
# - When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
# - When you need specialized knowledge or structured workflows
# - When a skill provides proven patterns for complex tasks

# **Skills are Self-Documenting:**
# - Each SKILL.md tells you exactly what the skill does and how to use it
# - The skill list above shows the full path for each skill's SKILL.md file

# **Executing Skill Scripts:**
# Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

# **Example Workflow:**

# User: "Can you research the latest developments in quantum computing?"

# 1. Check available skills above → See "web-research" skill with its full path
# 2. Read the skill using the path shown in the list
# 3. Follow the skill's research workflow (search → organize → synthesize)
# 4. Use any helper scripts with absolute paths

# Remember: Skills are tools to make you more capable and consistent. When in doubt, check if a skill exists for the task!
# """
SKILLS_SYSTEM_PROMPT = """

## 技能系统

你可以访问一个技能库，它提供了专业能力和领域知识。

{skills_locations}

**可用技能：**

{skills_list}

**如何使用技能（渐进式披露）：**

技能遵循**渐进式披露**模式 - 你知道它们的存在（上面的名称+描述），但只有在需要时才阅读完整说明：

1. **识别何时应用技能**：检查用户的任务是否与任何技能的描述匹配
2. **阅读技能的完整说明**：上面的技能列表显示了与read_file一起使用的确切路径
3. **遵循技能的说明**：SKILL.md包含逐步的工作流程、最佳实践和示例
4. **访问支持文件**：技能可能包括Python脚本、配置或参考文档 - 使用绝对路径

**何时使用技能：**
- 当用户的请求与技能领域匹配时（例如，"研究X" → web-research技能）
- 当你需要专业知识或结构化工作流程时
- 当技能为复杂任务提供经过验证的模式时

**技能是自我记录的：**
- 每个SKILL.md都会准确告诉你该技能做什么以及如何使用它
- 上面的技能列表显示了每个技能的SKILL.md文件的完整路径

**执行技能脚本：**
技能可能包含Python脚本或其他可执行文件。始终使用技能列表中的绝对路径。

**示例工作流程：**

用户："你能研究一下量子计算的最新发展吗？"

1. 查看上面的可用技能 → 看到"web-research"技能及其完整路径
2. 使用列表中显示的路径读取技能
3. 遵循技能的研究工作流程（搜索 → 组织 → 合成）
4. 使用带有绝对路径的任何辅助脚本

记住：技能是让你变得更强大和一致的工具。如有疑问，请检查是否存在适用于该任务的技能！
"""


class SkillsMiddleware(AgentMiddleware):
    """Middleware for loading and exposing agent skills.

    This middleware implements Anthropic's agent skills pattern:
    - Loads skills metadata (name, description) from YAML frontmatter at session start
    - Injects skills list into system prompt for discoverability
    - Agent reads full SKILL.md content when a skill is relevant (progressive disclosure)

    Supports both user-level and project-level skills:
    - User skills: ~/.deepagents/{AGENT_NAME}/skills/
    - Project skills: {PROJECT_ROOT}/.deepagents/skills/
    - Project skills override user skills with the same name

    Args:
        skills_dir: Path to the user-level skills directory (per-agent).
        assistant_id: The agent identifier for path references in prompts.
        project_skills_dir: Optional path to project-level skills directory.
    """

    state_schema = SkillsState

    def __init__(
        self,
        *,
        skills_dir: str | Path,
        assistant_id: str,
        project_skills_dir: str | Path | None = None,
    ) -> None:
        """Initialize the skills middleware.

        Args:
            skills_dir: Path to the user-level skills directory.
            assistant_id: The agent identifier.
            project_skills_dir: Optional path to the project-level skills directory.
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self.assistant_id = assistant_id
        self.project_skills_dir = (
            Path(project_skills_dir).expanduser() if project_skills_dir else None
        )
        # Store display paths for prompts
        # self.user_skills_display = f"~/.deepagents/{assistant_id}/skills"
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _format_skills_locations(self) -> str:
        """Format skills locations for display in system prompt."""
        locations = [f"**User Skills**: `{self.skills_dir}`"]
        if self.project_skills_dir:
            locations.append(
                f"**Project Skills**: `{self.project_skills_dir}` (overrides user skills)"
            )
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata for display in system prompt."""
        if not skills:
            locations = [f"{self.skills_dir}/"]
            if self.project_skills_dir:
                locations.append(f"{self.project_skills_dir}/")
            return f"(No skills available yet. You can create skills in {' or '.join(locations)})"

        # Group skills by source
        user_skills = [s for s in skills if s["source"] == "user"]
        project_skills = [s for s in skills if s["source"] == "project"]

        lines = []

        # Show user skills
        if user_skills:
            lines.append("**User Skills:**")
            for skill in user_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → Read `{skill['path']}` for full instructions")
            lines.append("")

        # Show project skills
        if project_skills:
            lines.append("**Project Skills:**")
            for skill in project_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → Read `{skill['path']}` for full instructions")

        return "\n".join(lines)

    def before_agent(self, state: SkillsState, runtime: Runtime) -> SkillsStateUpdate | None:
        """Load skills metadata before agent execution.

        This runs once at session start to discover available skills from both
        user-level and project-level directories.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with skills_metadata populated.
        """
        # We re-load skills on every new interaction with the agent to capture
        # any changes in the skills directories.
        skills = list_skills(
            user_skills_dir=self.skills_dir,
            project_skills_dir=self.project_skills_dir,
        )
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject skills documentation into the system prompt.

        This runs on every model call to ensure skills info is always available.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Get skills metadata from state
        skills_metadata = request.state.get("skills_metadata", [])

        # Format skills locations and list
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject skills documentation into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # The state is guaranteed to be SkillsState due to state_schema
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        # Format skills locations and list
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        # Inject into system prompt
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return await handler(request.override(system_prompt=system_prompt))
