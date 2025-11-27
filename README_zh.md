# 🚀🧠 深度智能体 (Deep Agents)

智能体可以越来越有效地处理长期任务，[智能体任务长度每7个月翻一番](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)！但是，长期任务通常涉及数十次工具调用，这带来了成本和可靠性挑战。像[Claude Code](https://code.claude.com/docs)和[Manus](https://www.youtube.com/watch?v=6_BcCthVvb8)这样的流行智能体使用一些共同原则来应对这些挑战，包括**规划**（在任务执行前）、**计算机访问**（给予访问shell和文件系统的权限）和**子智能体委派**（隔离的任务执行）。`deepagents`是一个实现了这些工具的简单智能体框架，它是开源且易于扩展以适应您的自定义工具和指令。

<img src=".github/images/deepagents_banner.png" alt="深度智能体" width="100%"/>

## 📚 资源

- **[文档](https://docs.langchain.com/oss/python/deepagents/overview)** - 完整概述和API参考
- **[快速入门仓库](https://github.com/langchain-ai/deepagents-quickstarts)** - 示例和用例
- **[命令行界面](libs/deepagents-cli/)** - 带有技能、记忆和人机协作工作流的交互式命令行界面

## 🚀 快速开始

您可以为`deepagents`提供自定义工具。下面，我们将可选地提供`tavily`工具来进行网络搜索。此工具将添加到`deepagents`内置工具中（见下文）。

```bash
pip install deepagents tavily-python
```

在环境中设置`TAVILY_API_KEY`（[在此获取](https://www.tavily.com/)）：

```python
import os
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(query: str, max_results: int = 5):
    """运行网络搜索"""
    return tavily_client.search(query, max_results=max_results)

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt="进行研究并撰写一份精炼的报告。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "什么是LangGraph?"}]})
```

通过`create_deep_agent`创建的智能体是已编译的[LangGraph StateGraph](https://docs.langchain.com/oss/python/langgraph/overview)，因此它可以与流式传输、人机协作、记忆或Studio一起使用，就像任何LangGraph智能体一样。更多示例请参见我们的[快速入门仓库](https://github.com/langchain-ai/deepagents-quickstarts)。

## 自定义深度智能体

有几个参数可以传递给`create_deep_agent`。

### `model`

默认情况下，`deepagents`使用`"claude-sonnet-4-5-20250929"`。您可以通过传递任何[LangChain模型对象](https://python.langchain.com/docs/integrations/chat/)来自定义它。

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model("openai:gpt-4o")
agent = create_deep_agent(
    model=model,
)
```

### `system_prompt`

您可以向`create_deep_agent()`提供`system_prompt`参数。这个自定义提示会**追加到**中间件自动注入的默认指令之后。

编写自定义系统提示时，您应该：

- ✅ 定义领域特定的工作流程（例如研究方法论、数据分析步骤）
- ✅ 为您的用例提供具体示例
- ✅ 添加专业指导（例如"将类似的研究任务批处理成单个待办事项"）
- ✅ 定义停止标准和资源限制
- ✅ 解释工具如何在您的工作流程中协同工作

**不要：**

- ❌ 重复解释标准工具的功能（中间件已经涵盖）
- ❌ 复制关于工具使用的中间件指令
- ❌ 与默认指令相矛盾（应与其协作而非对抗）

```python
from deepagents import create_deep_agent
research_instructions = """您的自定义系统提示"""
agent = create_deep_agent(
    system_prompt=research_instructions,
)
```

更多示例请参见我们的[快速入门仓库](https://github.com/langchain-ai/deepagents-quickstarts)。

### `tools`

为您的智能体提供自定义工具（除了[内置工具](#内置工具)）：

```python
from deepagents import create_deep_agent

def internet_search(query: str) -> str:
    """运行网络搜索"""
    return tavily_client.search(query)

agent = create_deep_agent(tools=[internet_search])
```

您也可以通过[langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)连接MCP工具：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()
    agent = create_deep_agent(tools=mcp_tools)

    async for chunk in agent.astream({"messages": [{"role": "user", "content": "..."}]}):
        chunk["messages"][-1].pretty_print()
```

### `middleware`

深度智能体使用[中间件](https://docs.langchain.com/oss/python/langchain/middleware)实现可扩展性（有关默认值，请参见[内置工具](#内置工具)）。添加自定义中间件以注入工具、修改提示或挂钩到智能体生命周期：

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware

@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city}的天气晴朗。"

class WeatherMiddleware(AgentMiddleware):
    tools = [get_weather]

agent = create_deep_agent(middleware=[WeatherMiddleware()])
```

### `subagents`

主智能体可以通过`task`工具（见[内置工具](#内置工具)）将工作委派给子智能体。您可以为上下文隔离和自定义指令提供自定义子智能体：

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "research-agent",
    "description": "用于深入研究问题",
    "prompt": "您是一位专业的研究员",
    "tools": [internet_search],
    "model": "openai:gpt-4o",  # 可选，默认为主智能体模型
}

agent = create_deep_agent(subagents=[research_subagent])
```

对于复杂情况，传递预构建的LangGraph图：

```python
from deepagents import CompiledSubAgent, create_deep_agent

custom_graph = create_agent(model=..., tools=..., prompt=...)

agent = create_deep_agent(
    subagents=[CompiledSubAgent(
        name="data-analyzer",
        description="专门用于数据分析的智能体",
        runnable=custom_graph
    )]
)
```

更多详情请参见[子智能体文档](https://docs.langchain.com/oss/python/deepagents/subagents)。

### `interrupt_on`

某些工具可能很敏感，在执行前需要人工批准。Deepagents通过LangGraph的中断功能支持人机协作工作流。您可以使用检查点配置哪些工具需要批准。

这些工具配置被传递给我们预建的[HITL中间件](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop)，使智能体暂停执行并在执行配置的工具之前等待用户反馈。

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city}的天气晴朗。"

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },
    }
)
```

更多详情请参见[人机协作文档](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)。

### `backend`

深度智能体使用可插拔后端来控制文件系统操作的工作方式。默认情况下，文件存储在智能体的临时状态中。您可以配置不同的后端以实现本地磁盘访问、跨对话持久化存储或混合路由。

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/project"),
)
```

可用的后端包括：

- **StateBackend**（默认）：存储在智能体状态中的临时文件
- **FilesystemBackend**：在根目录下的真实磁盘操作
- **StoreBackend**：使用LangGraph Store的持久化存储
- **CompositeBackend**：将不同路径路由到不同后端

更多详情请参见[后端文档](https://docs.langchain.com/oss/python/deepagents/backends)。

### 长期记忆

深度智能体可以使用`CompositeBackend`将特定路径路由到持久化存储，从而在对话间保持持久记忆。

这使得混合记忆成为可能，其中工作文件保持临时状态，而重要数据（如用户偏好或知识库）在所有线程中持续存在。

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(store=InMemoryStore())},
    ),
)
```

`/memories/`下的文件将在所有对话中持久化，而其他路径保持临时状态。使用案例包括：

- 在会话间保留用户偏好
- 从多个对话中构建知识库
- 基于反馈自我改进的指令
- 在会话间保持研究进度

更多详情请参见[长期记忆文档](https://docs.langchain.com/oss/python/deepagents/long-term-memory)。

## 内置工具

<img src=".github/images/deepagents_tools.png" alt="深度智能体" width="600"/>

每个通过`create_deep_agent`创建的深度智能体都带有一套标准工具：

| 工具名称 | 描述 | 提供者 |
|----------|------|--------|
| `write_todos` | 创建和管理结构化任务列表，用于跟踪复杂工作流程的进度 | TodoListMiddleware |
| `read_todos` | 读取当前待办事项列表状态 | TodoListMiddleware |
| `ls` | 列出目录中的所有文件（需要绝对路径） | FilesystemMiddleware |
| `read_file` | 从文件中读取内容，带有可选的分页（偏移量/限制参数） | FilesystemMiddleware |
| `write_file` | 创建新文件或完全覆盖现有文件 | FilesystemMiddleware |
| `edit_file` | 在文件中执行精确字符串替换 | FilesystemMiddleware |
| `glob` | 查找匹配模式的文件（例如，`**/*.py`） | FilesystemMiddleware |
| `grep` | 在文件中搜索文本模式 | FilesystemMiddleware |
| `execute`* | 在沙盒环境中运行shell命令 | FilesystemMiddleware |
| `task` | 将任务委派给具有隔离上下文窗口的专业子智能体 | SubAgentMiddleware |

只有当后端实现`SandboxBackendProtocol`时，`execute`工具才可用。默认情况下，它使用内存状态后端，该后端不支持命令执行。如图所示，这些工具（以及其他功能）由默认中间件提供：

更多关于内置工具和功能的详情请参见[智能体框架文档](https://docs.langchain.com/oss/python/deepagents/harness)。

## 内置中间件

`deepagents`在底层使用中间件。以下是所使用的中间件列表。

| 中间件 | 目的 |
|--------|------|
| **TodoListMiddleware** | 任务规划和进度跟踪 |
| **FilesystemMiddleware** | 文件操作和上下文卸载（自动保存大型结果） |
| **SubAgentMiddleware** | 将任务委派给隔离的子智能体 |
| **SummarizationMiddleware** | 当上下文超过170k令牌时自动摘要 |
| **AnthropicPromptCachingMiddleware** | 缓存系统提示以降低成本（仅限Anthropic） |
| **PatchToolCallsMiddleware** | 修复因中断而悬空的工具调用 |
| **HumanInTheLoopMiddleware** | 暂停执行以等待人工批准（需要`interrupt_on`配置） |

## 内置提示

中间件会自动添加关于标准工具的指令。您的自定义指令应该**补充而不是重复**这些默认值：

#### 来自[TodoListMiddleware](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/middleware/todo.py)

- 解释何时使用`write_todos`和`read_todos`
- 关于标记任务完成的指导
- 待办事项列表管理的最佳实践
- 何时不使用待办事项（简单任务）

#### 来自[FilesystemMiddleware](libs/deepagents/deepagents/middleware/filesystem.py)

- 列出所有文件系统工具（`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`execute`*）
- 解释文件路径必须以`/`开头
- 描述每个工具的目的和参数
- 关于为大型工具结果进行上下文卸载的说明

#### 来自[SubAgentMiddleware](libs/deepagents/deepagents/middleware/subagents.py)

- 解释用于委派给子智能体的`task()`工具
- 何时使用子智能体以及何时不使用它们
- 关于并行执行的指导
- 子智能体生命周期（生成→运行→返回→协调）