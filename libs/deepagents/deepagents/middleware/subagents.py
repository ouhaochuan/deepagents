"""Middleware for providing subagents to an agent via a `task` tool."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command


class SubAgent(TypedDict):
    """Specification for an agent.

    When specifying custom agents, the `default_middleware` from `SubAgentMiddleware`
    will be applied first, followed by any `middleware` specified in this spec.
    To use only custom middleware without the defaults, pass `default_middleware=[]`
    to `SubAgentMiddleware`.
    """

    name: str
    """The name of the agent."""

    description: str
    """The description of the agent."""

    system_prompt: str
    """The system prompt to use for the agent."""

    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    """The tools to use for the agent."""

    model: NotRequired[str | BaseChatModel]
    """The model for the agent. Defaults to `default_model`."""

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware to append after `default_middleware`."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """The tool configs to use for the agent."""


class CompiledSubAgent(TypedDict):
    """A pre-compiled agent spec."""

    name: str
    """The name of the agent."""

    description: str
    """The description of the agent."""

    runnable: Runnable
    """The Runnable to use for the agent."""


DEFAULT_SUBAGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# State keys that should be excluded when passing state to subagents
_EXCLUDED_STATE_KEYS = ("messages", "todos")

# TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

# Available agent types and the tools they have access to:
# {available_agents}

# When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

# ## Usage notes:
# 1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
# 2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
# 3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
# 4. The agent's outputs should generally be trusted
# 5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
# 6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
# 7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

# ### Example usage of the general-purpose agent:

# <example_agent_descriptions>
# "general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
# </example_agent_descriptions>

# <example>
# User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
# Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
# Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
# <commentary>
# Research is a complex, multi-step task in it of itself.
# The research of each individual player is not dependent on the research of the other players.
# The assistant uses the task tool to break down the complex objective into three isolated tasks.
# Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
# This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
# </commentary>
# </example>

# <example>
# User: "Analyze a single large code repository for security vulnerabilities and generate a report."
# Assistant: *Launches a single `task` subagent for the repository analysis*
# Assistant: *Receives report and integrates results into final summary*
# <commentary>
# Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
# If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
# </commentary>
# </example>

# <example>
# User: "Schedule two meetings for me and prepare agendas for each."
# Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
# Assistant: *Returns final schedules and agendas*
# <commentary>
# Tasks are simple individually, but subagents help silo agenda preparation.
# Each subagent only needs to worry about the agenda for one meeting.
# </commentary>
# </example>

# <example>
# User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
# Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
# <commentary>
# The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
# It is better to just complete the task directly and NOT use the `task`tool.
# </commentary>
# </example>

# ### Example usage with custom agents:

# <example_agent_descriptions>
# "content-reviewer": use this agent after you are done creating significant content or documents
# "greeting-responder": use this agent when to respond to user greetings with a friendly joke
# "research-analyst": use this agent to conduct thorough research on complex topics
# </example_agent_description>

# <example>
# user: "Please write a function that checks if a number is prime"
# assistant: Sure let me write a function that checks if a number is prime
# assistant: First let me use the Write tool to write a function that checks if a number is prime
# assistant: I'm going to use the Write tool to write the following code:
# <code>
# function isPrime(n) {{
#   if (n <= 1) return false
#   for (let i = 2; i * i <= n; i++) {{
#     if (n % i === 0) return false
#   }}
#   return true
# }}
# </code>
# <commentary>
# Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
# </commentary>
# assistant: Now let me use the content-reviewer agent to review the code
# assistant: Uses the Task tool to launch with the content-reviewer agent
# </example>

# <example>
# user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
# <commentary>
# This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
# </commentary>
# assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
# assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
# </example>

# <example>
# user: "Hello"
# <commentary>
# Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
# </commentary>
# assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
# </example>"""  # noqa: E501
TASK_TOOL_DESCRIPTION = """启动一个临时subagent来处理复杂的、多步骤的独立任务，并具有隔离的上下文窗口。

可用的subagent类型及其可访问的工具：
{available_agents}

使用Task工具时，必须指定subagent_type参数来选择要使用的subagent类型。

## 使用说明：
1. 尽可能并发启动多个agent以最大化性能；为此，请在单个消息中使用多个工具调用
2. 当agent完成时，它会向您返回一条消息。agent返回的结果对用户不可见。要向用户显示结果，您应该发送一条文本消息给用户，其中包含结果的简明摘要。
3. 每次agent调用都是无状态的。您将无法向agent发送额外的消息，agent也无法在其最终报告之外与您通信。因此，您的提示应包含详细的任务描述，以便agent能够自主执行，并且您应明确指定agent应在其最终且唯一的返回消息中向您提供哪些信息。
4. 通常应信任agent的输出
5. 清楚地告诉agent您期望它是创建内容、执行分析还是仅仅进行研究（搜索、文件读取、网络获取等），因为它不知道用户的意图
6. 如果agent描述提到应主动使用它，那么您应尽最大努力在用户未首先要求的情况下使用它。请运用您的判断力。
7. 当仅提供通用agent时，您应将其用于所有任务。它非常适合隔离上下文和token使用，并完成特定的复杂任务，因为它拥有与主agent相同的所有功能。

### 通用agent的使用示例：

<example_agent_descriptions>
"general-purpose": 使用此agent处理一般性任务，它可以访问所有与主agent相同的工具。
</example_agent_descriptions>

<example>
用户："我想研究勒布朗·詹姆斯、迈克尔·乔丹和科比·布莱恩特的成就，然后进行比较。"
助理：*并行使用task工具对三位球员分别进行独立研究*
助理：*综合三个独立研究任务的结果并回应用户*
<commentary>
研究本身就是一个复杂的、多步骤的任务。
每个球员的单独研究并不依赖于其他球员的研究。
助手使用task工具将复杂的总体目标分解为三个独立的任务。
每项研究任务只需要关注一个球员的上下文和tokens，然后将关于该球员的综合信息作为工具结果返回。
这意味着每项研究任务都可以深入挖掘并花费tokens和上下文深度研究每位球员，但最终结果是综合信息，在长远来看比较球员时为我们节省了tokens。
</commentary>
</example>

<example>
用户："分析单个大型代码库的安全漏洞并生成报告。"
助理：*启动单个`task` subagent进行仓库分析*
助理：*接收报告并将结果整合到最终摘要中*
<commentary>
Subagent用于隔离大型、上下文繁重的任务，即使只有一个任务。这可以防止主线程过载细节。
如果用户随后提出后续问题，我们有一份简洁的报告可供参考，而不是整个分析和工具调用的历史记录，这样既省时又省钱。
</commentary>
</example>

<example>
用户："为我安排两次会议并准备每次会议的议程。"
助理：*并行调用task工具启动两个`task` subagent（每次会议一个）来准备议程*
助理：*返回最终时间安排和议程*
<commentary>
每项任务单独来看都很简单，但subagent有助于隔离议程准备工作。
每个subagent只需要关心一次会议的议程。
</commentary>
</example>

<example>
用户："我想从达美乐订购披萨，从麦当劳订购汉堡，从赛百味订购沙拉。"
助理：*并行直接调用工具从达美乐订购披萨、从麦当劳订购汉堡、从赛百味订购沙拉*
<commentary>
任务非常简单明了，只需要几次简单的工具调用。
直接完成任务比使用`task`工具更好。
</commentary>
</example>

### 自定义agent的使用示例：

<example_agent_descriptions>
"content-reviewer": 在您完成创建重要内容或文档后使用此agent
"greeting-responder": 使用此agent以友好的笑话回应用户问候
"research-analyst": 使用此agent对复杂主题进行彻底研究
</example_agent_descriptions>

<example>
用户："请写一个检查数字是否为质数的函数"
助理：好的让我写一个检查数字是否为质数的函数
助理：首先让我使用Write工具写一个检查数字是否为质数的函数
助理：我将使用Write工具编写以下代码：
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
由于创建了重要内容并且任务已完成，现在使用content-reviewer agent审查工作
</commentary>
助理：现在让我使用content-reviewer agent审查代码
助理：使用Task工具启动content-reviewer agent
</example>

<example>
用户："你能帮我研究不同可再生能源的环境影响并制作一份全面的报告吗？"
<commentary>
这是一项复杂的研究任务，使用research-analyst agent进行深入分析将有所帮助
</commentary>
助理：我将帮您研究可再生能源的环境影响。让我使用research-analyst agent对此主题进行全面研究。
助理：使用Task工具启动research-analyst agent，提供有关要进行的研究和报告格式的详细说明
</example>
"""  # noqa: E501
# 去掉这个例子，否则输入你好会触发greeting-responder agent，没有必要
# <example>
# 用户："你好"
# <commentary>
# 由于用户在打招呼，使用greeting-responder agent以友好的笑话回应
# </commentary>
# 助理："我将使用Task工具启动greeting-responder agent"
# </example>

# TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

# You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

# When to use the task tool:
# - When a task is complex and multi-step, and can be fully delegated in isolation
# - When a task is independent of other tasks and can run in parallel
# - When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
# - When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
# - When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

# Subagent lifecycle:
# 1. **Spawn** → Provide clear role, instructions, and expected output
# 2. **Run** → The subagent completes the task autonomously
# 3. **Return** → The subagent provides a single structured result
# 4. **Reconcile** → Incorporate or synthesize the result into the main thread

# When NOT to use the task tool:
# - If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
# - If the task is trivial (a few tool calls or simple lookup)
# - If delegating does not reduce token usage, complexity, or context switching
# - If splitting would add latency without benefit

# ## Important Task Tool Usage Notes to Remember
# - Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
# - Remember to use the `task` tool to silo independent tasks within a multi-part objective.
# - You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task` (subagent生成器)

您可以使用 `task` 工具启动短期存在的subagent来处理独立任务。这些agent是短暂的——它们只在任务期间存在并返回单个结果。

何时使用 task 工具：
- 当任务复杂且多步骤，并且可以完全独立委派时
- 当任务独立于其他任务并且可以并行运行时
- 当任务需要专注推理或大量令牌/上下文使用量会膨胀协调器线程时
- 当沙箱提高可靠性时（例如代码执行、结构化搜索、数据格式化）
- 当您只关心subagent的输出而不是中间步骤时（例如进行大量研究然后返回综合报告，执行一系列计算或查找以获得简洁、相关的答案）

subagent生命周期：
1. **生成** → 提供清晰的角色、指令和预期输出
2. **运行** → subagent自主完成任务
3. **返回** → subagent提供单个结构化结果
4. **协调** → 将结果合并或综合到主线程中

何时不使用 task 工具：
- 如果您需要在subagent完成后查看中间推理或步骤（task 工具隐藏了它们）
- 如果任务微不足道（几次工具调用或简单查找）
- 如果委派不会减少令牌使用量、复杂性或上下文切换
- 如果分割会增加延迟而没有好处

## 重要 Task 工具使用注意事项
- 只要可能，请并行化您的工作。这对 tool_calls 和 tasks 都适用。当您有独立步骤需要完成时——并行进行 tool_calls 或启动 tasks（subagent）以更快地完成它们。这为用户节省了时间，这是非常重要的。
- 记住使用 `task` 工具来隔离多部分目标内的独立任务。
- 当您有一个复杂的任务需要多个步骤，并且与agent需要完成的其他任务无关时，应该使用 `task` 工具。这些agent非常能干且高效。"""  # noqa: E501

# DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501
DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "用于研究复杂问题、搜索文件和内容以及执行多步骤任务的通用agent。当您在搜索关键词或文件时，如果您不确定能在前几次尝试中找到正确匹配项，请使用此agent为您执行搜索。此agent可以访问与主agent相同的所有工具。"  # noqa: E501


def _get_subagents(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Create subagent instances from specifications.

    Args:
        default_model: Default model for subagents that don't specify one.
        default_tools: Default tools for subagents that don't specify tools.
        default_middleware: Middleware to apply to all subagents. If `None`,
            no default middleware is applied.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent. These
            are also the fallback for any subagents that don't specify their own tool configs.
        subagents: List of agent specifications or pre-compiled agents.
        general_purpose_agent: Whether to include a general-purpose subagent.

    Returns:
        Tuple of (agent_dict, description_list) where agent_dict maps agent names
        to runnable instances and description_list contains formatted descriptions.
    """
    # Use empty list if None (no default middleware)
    default_subagent_middleware = default_middleware or []

    agents: dict[str, Any] = {}
    subagent_descriptions = []

    # Create general-purpose agent if enabled
    if general_purpose_agent:
        general_purpose_middleware = [*default_subagent_middleware]
        if default_interrupt_on:
            general_purpose_middleware.append(HumanInTheLoopMiddleware(interrupt_on=default_interrupt_on))
        general_purpose_subagent = create_agent(
            default_model,
            system_prompt=DEFAULT_SUBAGENT_PROMPT,
            tools=default_tools,
            middleware=general_purpose_middleware,
        )
        agents["general-purpose"] = general_purpose_subagent
        subagent_descriptions.append(f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")

    # Process custom subagents
    for agent_ in subagents:
        subagent_descriptions.append(f"- {agent_['name']}: {agent_['description']}")
        if "runnable" in agent_:
            custom_agent = cast("CompiledSubAgent", agent_)
            agents[custom_agent["name"]] = custom_agent["runnable"]
            continue
        _tools = agent_.get("tools", list(default_tools))

        subagent_model = agent_.get("model", default_model)

        _middleware = [*default_subagent_middleware, *agent_["middleware"]] if "middleware" in agent_ else [*default_subagent_middleware]

        interrupt_on = agent_.get("interrupt_on", default_interrupt_on)
        if interrupt_on:
            _middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        agents[agent_["name"]] = create_agent(
            subagent_model,
            system_prompt=agent_["system_prompt"],
            tools=_tools,
            middleware=_middleware,
        )
    return agents, subagent_descriptions


def _create_task_tool(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
    task_description: str | None = None,
) -> BaseTool:
    """Create a task tool for invoking subagents.

    Args:
        default_model: Default model for subagents.
        default_tools: Default tools for subagents.
        default_middleware: Middleware to apply to all subagents.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent. These
            are also the fallback for any subagents that don't specify their own tool configs.
        subagents: List of subagent specifications.
        general_purpose_agent: Whether to include general-purpose agent.
        task_description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """
    subagent_graphs, subagent_descriptions = _get_subagents(
        default_model=default_model,
        default_tools=default_tools,
        default_middleware=default_middleware,
        default_interrupt_on=default_interrupt_on,
        subagents=subagents,
        general_purpose_agent=general_purpose_agent,
    )
    subagent_description_str = "\n".join(subagent_descriptions)

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(result["messages"][-1].text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(subagent_type: str, description: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
        """Prepare state for invocation."""
        subagent = subagent_graphs[subagent_type]
        # Create a new state dict to avoid mutating the original
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    # Use custom description if provided, otherwise use default template
    if task_description is None:
        task_description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        # If custom description has placeholder, format with agent descriptions
        task_description = task_description.format(available_agents=subagent_description_str)

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = subagent.invoke(subagent_state)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = await subagent.ainvoke(subagent_state)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=task_description,
    )


class SubAgentMiddleware(AgentMiddleware):
    """Middleware for providing subagents to an agent via a `task` tool.

    This  middleware adds a `task` tool to the agent that can be used to invoke subagents.
    Subagents are useful for handling complex tasks that require multiple steps, or tasks
    that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks, and then return
    a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require a narrower
    subset of tools and focus.

    This middleware comes with a default general-purpose subagent that can be used to
    handle the same tasks as the main agent, but with isolated context.

    Args:
        default_model: The model to use for subagents.
            Can be a LanguageModelLike or a dict for init_chat_model.
        default_tools: The tools to use for the default general-purpose subagent.
        default_middleware: Default middleware to apply to all subagents. If `None` (default),
            no default middleware is applied. Pass a list to specify custom middleware.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent. These
            are also the fallback for any subagents that don't specify their own tool configs.
        subagents: A list of additional subagents to provide to the agent.
        system_prompt: Full system prompt override. When provided, completely replaces
            the agent's system prompt.
        general_purpose_agent: Whether to include the general-purpose agent. Defaults to `True`.
        task_description: Custom description for the task tool. If `None`, uses the
            default description template.

    Example:
        ```python
        from langchain.agents.middleware.subagents import SubAgentMiddleware
        from langchain.agents import create_agent

        # Basic usage with defaults (no default middleware)
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    subagents=[],
                )
            ],
        )

        # Add custom middleware to subagents
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    default_middleware=[TodoListMiddleware()],
                    subagents=[],
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        default_middleware: list[AgentMiddleware] | None = None,
        default_interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        general_purpose_agent: bool = True,
        task_description: str | None = None,
    ) -> None:
        """Initialize the SubAgentMiddleware."""
        super().__init__()
        self.system_prompt = system_prompt
        task_tool = _create_task_tool(
            default_model=default_model,
            default_tools=default_tools or [],
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
            task_description=task_description,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt to include instructions on using subagents."""
        if self.system_prompt is not None:
            system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
            return handler(request.override(system_prompt=system_prompt))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt to include instructions on using subagents."""
        if self.system_prompt is not None:
            system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
            return await handler(request.override(system_prompt=system_prompt))
        return await handler(request)
