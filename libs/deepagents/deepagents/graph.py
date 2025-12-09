"""Deepagents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.middleware.prompt_logger import PromptLoggerNodeMiddleware
from deepagents.middleware.directory_tree import DirectoryTreeMiddleware

#BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."
BASE_AGENT_PROMPT = "为了完成用户交给你的目标，你可以使用许多标准工具。"


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        ChatAnthropic instance configured with Claude Sonnet 4.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )

# totolist工具的系统提示词翻译
TODO_LIST_SYSTEM_PROMPT = """## `write_todos`
您可以使用 `write_todos`工具来帮助您管理和规划复杂的目标。
对于复杂的目标，请使用此工具以确保您正在跟踪每个必要的步骤，并让用户能够看到您的进度。
这个工具对规划复杂目标非常有帮助，可以将这些较大的复杂目标分解成更小的步骤。

一旦完成某个步骤，您必须立即将待办事项标记为已完成。不要累积多个步骤后再一起标记为完成。
对于只需要几个步骤的简单目标，最好直接完成目标，而不要使用此工具。
编写待办事项需要时间和token，只在管理复杂的多步骤问题时使用它！而不是用于简单的几步就能完成的请求。

## 需要记住的重要待办事项列表使用说明
- 切勿并行多次调用 `write_todos`工具。
- 随着工作的进展，不要害怕修改待办事项列表。新的信息可能会揭示出需要完成的新任务，或者发现旧任务已经不相关。
"""

def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    seven file and execution tools: ls, read_file, write_file, edit_file, glob, grep, execute,
    and a tool to call subagents.

    The execute tool allows running shell commands if the backend implements SandboxBackendProtocol.
    For non-sandbox backends, the execute tool will return an error message.

    Args:
        model: The model to use. Defaults to Claude Sonnet 4.
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the
                  sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict
                  settings)
                - (optional) `middleware` (list of AgentMiddleware)
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses StoreBackend).
        backend: Optional backend for file storage and execution. Pass either a Backend instance
            or a callable factory like `lambda rt: StateBackend(rt)`. For execution support,
            use a backend that implements SandboxBackendProtocol.
        interrupt_on: Optional Dict[str, bool | InterruptOnConfig] mapping tool names to
            interrupt configs.
        debug: Whether to enable debug mode. Passed through to create_agent.
        name: The name of the agent. Passed through to create_agent.
        cache: The cache to use for the agent. Passed through to create_agent.

    Returns:
        A configured deep agent.
    """
    # 打印传入的middleware
    # print(f"middleware: {middleware}")

    if model is None:
        model = get_default_model()

    if (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    ):
        trigger = ("fraction", 0.85)
        keep = ("fraction", 0.10)
    else:
        trigger = ("tokens", 170000)
        keep = ("messages", 6)

    deepagent_middleware = [
        PromptLoggerNodeMiddleware(),
        TodoListMiddleware(system_prompt=TODO_LIST_SYSTEM_PROMPT),# 提示词注入点-todolist工具系统提示词
        FilesystemMiddleware(backend=backend, ignore_output_truncate_tools=["list_directory_tree"]),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(system_prompt=TODO_LIST_SYSTEM_PROMPT),
                FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(
                    model=model,
                    trigger=trigger,
                    keep=keep,
                    trim_tokens_to_summarize=None,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            trim_tokens_to_summarize=None,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
        DirectoryTreeMiddleware(),
    ]
    if middleware:
        deepagent_middleware.extend(middleware)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # 打印合并后的middleware
    # print(f"deepagent_middleware: {deepagent_middleware}")

    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
