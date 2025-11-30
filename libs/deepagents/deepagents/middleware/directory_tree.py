"""Middleware for providing directory tree functionality to an agent."""
from pathlib import Path
from typing import Any, Dict, List, Union, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.types import Overwrite


def get_directory_tree(path: Path, max_depth: int = 3, current_depth: int = 0) -> dict[str, Any]:
    """递归获取目录树结构
    
    Args:
        path: 要遍历的目录路径
        max_depth: 最大遍历深度
        current_depth: 当前深度（内部使用）
        
    Returns:
        表示目录树的字典
    """
    if current_depth > max_depth:
        return {}
    
    tree = {
        "name": path.name or path.absolute().name,
        "path": str(path.relative_to(Path.cwd())) if not path.is_absolute() else str(path),
        "is_dir": path.is_dir(),
    }
    
    if path.is_dir():
        try:
            children = []
            for child in path.iterdir():
                # 隐藏文件和目录通常以 . 开头，可以选择性地跳过
                if not child.name.startswith('.'):
                    children.append(get_directory_tree(child, max_depth, current_depth + 1))
            tree["children"] = children
        except PermissionError:
            tree["error"] = "Permission denied"
    else:
        try:
            tree["size"] = path.stat().st_size
        except (OSError, PermissionError):
            tree["size"] = None
            
    return tree


@tool
def list_directory_tree(max_depth: int = 3) -> dict[str, Any]:
    """以JSON结构输出当前工作目录的文件树
    
    Args:
        max_depth: 最大遍历深度，默认为3层
        
    Returns:
        包含目录树结构的字典
    """
    current_dir = Path.cwd()
    tree = get_directory_tree(current_dir, max_depth)
    return {
        "current_directory": str(current_dir),
        "tree": tree
    }


class DirectoryTreeMiddleware(AgentMiddleware):
    """Middleware that provides directory tree functionality and removes related messages after processing."""
    
    def __init__(self) -> None:
        """Initialize the middleware with the list_directory_tree tool."""
        self.tools = [list_directory_tree]
    
    # def after_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
    #     """Remove list_directory_tree tool calls and results from messages after model execution.
    #     
    #     Args:
    #         state: Current agent state containing messages
    #         runtime: Runtime context
    #         
    #     Returns:
    #         Updated state with filtered messages, or None if no changes needed
    #     """
    #     print("DirectoryTreeMiddleware: After model call")
    #     if "messages" not in state or not state["messages"]:
    #         return None
    #         
    #     messages = state["messages"]
    #     print(f'len(messages): {len(messages)}')
    #
    #     if len(messages) < 3:  # 确保消息列表至少有三个元素
    #         return None
    #
    #    # 检查倒数第三条和倒数第二条消息是否为list_directory_tree工具调用及其结果
    #     third_last_msg = messages[-3]
    #     second_last_msg = messages[-2]
    #     
    #     # 检查倒数第三条消息是否为list_directory_tree工具调用
    #     if (isinstance(third_last_msg, AIMessage) and third_last_msg.tool_calls and 
    #         any(tc["name"] == "list_directory_tree" for tc in third_last_msg.tool_calls)):
    #         
    #         # 获取工具调用ID
    #         tool_call_ids = [tc["id"] for tc in third_last_msg.tool_calls if tc["name"] == "list_directory_tree"]
    #         
    #         # 检查倒数第二条消息是否为对应的工具结果
    #         if (isinstance(second_last_msg, ToolMessage) and 
    #             second_last_msg.tool_call_id in tool_call_ids):
    #             # 移除倒数第三条和倒数第二条消息
    #             new_messages = messages[:-3] + messages[-1:]
    #             print(f'len(new_messages): {len(new_messages)}')
    #             # return Command(update={"messages": new_messages})
    #             return {"messages": Overwrite(new_messages)}
    #     
    #     return None
    
    def after_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """Modify list_directory_tree tool message content after model execution.
        
        Args:
            state: Current agent state containing messages
            runtime: Runtime context
            
        Returns:
            Updated state with modified tool message content, or None if no changes needed
        """
        print("DirectoryTreeMiddleware: After model call")
        if "messages" not in state or not state["messages"]:
            return None
            
        messages = state["messages"]
        print(f'len(messages): {len(messages)}')

        if len(messages) < 3:  # 确保消息列表至少有三个元素
            return None

        # 检查倒数第三条和倒数第二条消息是否为list_directory_tree工具调用及其结果
        third_last_msg = messages[-3]
        second_last_msg = messages[-2]
        
        # 检查倒数第三条消息是否为list_directory_tree工具调用
        if (isinstance(third_last_msg, AIMessage) and third_last_msg.tool_calls and 
            any(tc["name"] == "list_directory_tree" for tc in third_last_msg.tool_calls)):
            
            # 获取工具调用ID
            tool_call_ids = [tc["id"] for tc in third_last_msg.tool_calls if tc["name"] == "list_directory_tree"]
            
            # 检查倒数第二条消息是否为对应的工具结果
            if (isinstance(second_last_msg, ToolMessage) and 
                second_last_msg.tool_call_id in tool_call_ids and
                second_last_msg.name == "list_directory_tree"):
                # 修改工具消息内容
                new_content = "list_directory_tree工具已正确返回并且你已经正确处理了该工具返回的内容，但因为内容过长，已被清理掉，如果还需要该结果请重新执行list_directory_tree工具来获取。"
                
                # 创建新的消息列表，只替换倒数第二条消息的内容
                new_messages = list(messages)
                new_messages[-2] = ToolMessage(
                    content=new_content,
                    tool_call_id=second_last_msg.tool_call_id,
                    name=second_last_msg.name
                )
                
                print(f'Modified tool message content')
                return {"messages": Overwrite(new_messages)}
        
        return None
