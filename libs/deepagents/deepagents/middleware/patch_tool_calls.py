"""Middleware to patch dangling tool calls in the messages history."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
import os


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, handle dangling tool calls from any AIMessage."""
        messages = state["messages"]
        if not messages or len(messages) == 0:
            return None

        patched_messages = []
        # Iterate over the messages and add any dangling tool calls
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # We have a dangling tool call which needs a ToolMessage
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        if os.environ.get("DEBUG_PATCH_TOOL_CALLS") == "true":
          # 打印即将发送给LLM的消息
          print("即将发送给LLM的消息:")
          for i, msg in enumerate(patched_messages):
            print(f"  消息 {i}: {type(msg).__name__}")
            if hasattr(msg, 'content'):
                print(f"    内容: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"    工具调用: {msg.tool_calls}")
            if msg.type == "tool":
                print(f"    工具名称: {getattr(msg, 'name', 'N/A')}")
                print(f"    工具调用ID: {msg.tool_call_id}")
        return {"messages": Overwrite(patched_messages)}
