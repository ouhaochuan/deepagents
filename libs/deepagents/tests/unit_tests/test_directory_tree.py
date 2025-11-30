"""Tests for DirectoryTreeMiddleware."""

from unittest.mock import Mock
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.types import Command

from deepagents.middleware.directory_tree import DirectoryTreeMiddleware


class TestDirectoryTreeMiddleware:
    """Test cases for DirectoryTreeMiddleware."""
    
    def test_after_model_no_messages(self):
        """Test after_model when no messages in state."""
        middleware = DirectoryTreeMiddleware()
        state = {}
        result = middleware.after_model(state, Mock())
        assert result is None
    
    def test_after_model_empty_messages(self):
        """Test after_model when messages list is empty."""
        middleware = DirectoryTreeMiddleware()
        state = {"messages": []}
        result = middleware.after_model(state, Mock())
        assert result is None
    
    def test_after_model_less_than_three_messages(self):
        """Test after_model when less than 3 messages."""
        middleware = DirectoryTreeMiddleware()
        state = {"messages": [HumanMessage(content="hello")]}
        result = middleware.after_model(state, Mock())
        assert result is None
        
        state = {"messages": [
            HumanMessage(content="hello"),
            AIMessage(content="hi")
        ]}
        result = middleware.after_model(state, Mock())
        assert result is None
    
    def test_after_model_with_list_directory_tree_call_and_result(self):
        """Test after_model when there's list_directory_tree tool call and result."""
        middleware = DirectoryTreeMiddleware()
        
        tool_call_id = "tool_call_123"
        state = {
            "messages": [
                HumanMessage(content="Show me the directory tree"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_directory_tree",
                            "args": {"max_depth": 3},
                            "id": tool_call_id
                        }
                    ]
                ),
                ToolMessage(
                    content='{"current_directory": "/home/user", "tree": {"name": "user"}}',
                    tool_call_id=tool_call_id
                ),
                AIMessage(content="Here is the directory tree you requested.")
            ]
        }
        
        result = middleware.after_model(state, Mock())
        assert isinstance(result, Command)
        assert "messages" in result.update
        # Should remove the tool call and tool result messages, keeping first and last
        assert len(result.update["messages"]) == 2
        assert isinstance(result.update["messages"][0], HumanMessage)
        assert isinstance(result.update["messages"][1], AIMessage)
        assert result.update["messages"][1].content == "Here is the directory tree you requested."
    
    def test_after_model_with_non_list_directory_tree_call(self):
        """Test after_model when there's a different tool call."""
        middleware = DirectoryTreeMiddleware()
        
        tool_call_id = "tool_call_123"
        state = {
            "messages": [
                HumanMessage(content="Do something else"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "some_other_tool",
                            "args": {},
                            "id": tool_call_id
                        }
                    ]
                ),
                ToolMessage(
                    content='{"result": "success"}',
                    tool_call_id=tool_call_id
                ),
                AIMessage(content="Task completed.")
            ]
        }
        
        result = middleware.after_model(state, Mock())
        # Should not modify messages for non-list_directory_tree tools
        assert result is None
    
    def test_after_model_without_matching_tool_result(self):
        """Test after_model when tool call and result don't match."""
        middleware = DirectoryTreeMiddleware()
        
        state = {
            "messages": [
                HumanMessage(content="Show me the directory tree"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_directory_tree",
                            "args": {"max_depth": 3},
                            "id": "tool_call_123"
                        }
                    ]
                ),
                ToolMessage(
                    content='{"result": "success"}',
                    tool_call_id="different_tool_call_id"
                ),
                AIMessage(content="Some message.")
            ]
        }
        
        result = middleware.after_model(state, Mock())
        # Should not modify messages when tool_call_id doesn't match
        assert result is None