"""Middleware for providing directory tree functionality to an agent."""
from pathlib import Path
from typing import Any, Dict, List, Union, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage


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
    
    def _extract_tool_calls_to_remove(self, messages: List[Any]) -> List[str]:
        """从消息中提取需要移除的工具调用ID列表
        
        Args:
            messages: 消息列表
            
        Returns:
            需要移除的工具调用ID列表
        """
        if not messages:
            return []
            
        # 获取最新的AI消息
        last_message = messages[-1]
        tool_calls_to_remove = []
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "list_directory_tree":
                    tool_calls_to_remove.append(tool_call["id"])
                    
        return tool_calls_to_remove
    
    def _create_post_processor(self, tool_calls_to_remove: List[str]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """创建用于移除工具调用消息的后处理器
        
        Args:
            tool_calls_to_remove: 需要移除的工具调用ID列表
            
        Returns:
            后处理器函数
        """
        def post_process_messages(state: Dict[str, Any]) -> Dict[str, Any]:
            if "messages" not in state:
                return state
                
            new_messages = []
            i = 0
            while i < len(state["messages"]):
                msg = state["messages"][i]
                
                # 检查是否是需要移除的工具调用消息
                if (isinstance(msg, AIMessage) and msg.tool_calls and 
                    any(tc["id"] in tool_calls_to_remove for tc in msg.tool_calls)):
                    # 跳过这个AIMessage和后续相关的ToolMessage
                    i += 1  # 跳过AIMessage
                    
                    # 继续跳过所有相关的ToolMessage
                    while i < len(state["messages"]):
                        next_msg = state["messages"][i]
                        if (isinstance(next_msg, ToolMessage) and 
                            next_msg.tool_call_id in tool_calls_to_remove):
                            i += 1  # 跳过ToolMessage
                        else:
                            new_messages.append(next_msg)
                            i += 1
                else:
                    new_messages.append(msg)
                    i += 1
            
            return {"messages": new_messages}
            
        return post_process_messages
    
    def _process_response(self, request: ModelRequest, response: ModelResponse) -> ModelResponse:
        """处理响应，添加后处理器以移除相关工具调用消息
        
        Args:
            request: 模型请求
            response: 模型响应
            
        Returns:
            处理后的模型响应
        """
        # 获取所有消息
        messages = list(request.messages)
        # 检查响应是否包含消息并相应处理
        if hasattr(response, 'message') and response.message:
            messages.append(response.message)
        elif hasattr(response, 'messages') and response.messages:
            messages.extend(response.messages)
            
        # 检查是否有使用list_directory_tree工具的调用
        tool_calls_to_remove = self._extract_tool_calls_to_remove(messages)
        
        # 如果有相关的工具调用，添加后处理器
        if tool_calls_to_remove:
            post_processor = self._create_post_processor(tool_calls_to_remove)
            
            # 将处理函数附加到响应中
            if not hasattr(response, 'post_processors'):
                response.post_processors = []
            response.post_processors.append(post_processor)
        
        return response
    
    def wrap_model_call(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Process model response to remove directory tree tool calls and results."""
        # 先调用原始处理器获取响应
        response = handler(request)
        
        # 处理响应
        return self._process_response(request, response)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Async process model response to remove directory tree tool calls and results."""
        # 先调用原始处理器获取响应
        response = await handler(request)
        
        # 处理响应
        return self._process_response(request, response)