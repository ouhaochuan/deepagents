from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
import json
from datetime import datetime
from typing import Callable, Awaitable, Dict, Any
import os

from deepagents.utils import load_env_with_fallback_verbose
from langchain.agents import AgentState

# 加载环境变量（仅当PROMPT_LOGGER_ENABLED未设置时）
if os.getenv('PROMPT_LOGGER_ENABLED') is None:
    load_env_with_fallback_verbose()

class LogState(AgentState):
  """AgentState with log_request and log_response methods."""
  call_count: int = 1

  def __init__(self):
    self.call_count = 1

class PromptLoggerBaseMiddleware(AgentMiddleware):

  state_schema = LogState
  
  def __init__(self):
         # 检查是否启用日志记录
        self.enabled = os.getenv('PROMPT_LOGGER_ENABLED', 'true').lower() == 'true'
        
        if self.enabled:
            # 在用户主目录下创建.deepagents-cli/logs目录
            home_dir = os.path.expanduser("~")
            self.log_dir = os.path.join(home_dir, ".deepagents-cli", "logs")
            os.makedirs(self.log_dir, exist_ok=True)
        
        self.call_count = 0
    
  def _log_request(self, request: ModelRequest):
      """记录请求信息的通用方法"""
      # 如果未启用，则直接返回
      if not self.enabled:
          return
          
      self.call_count = request.state.get('call_count', 1)
      # print(f"read call_count: {self.call_count}")
      
      # 获取state的实际类型名称
      state_type_name = type(request.state).__name__
      state_module = type(request.state).__module__
      full_state_type = f"{state_module}.{state_type_name}"
      
      # 从ModelRequest中正确获取系统消息和消息列表
      system_message = request.system_message
      messages = request.state['messages']
      
      # 创建Markdown格式的日志内容
      timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      md_content = []
      md_content.append(f"# Agent Call #{self.call_count}")
      md_content.append(f"")
      md_content.append(f"- **Timestamp**: {timestamp_str}")
      md_content.append(f"- **State Type**: {full_state_type}")
      md_content.append(f"- **Messages Count**: {len(messages)}")
      md_content.append(f"")
      
      # 添加系统提示词
      if system_message and system_message.content:
          md_content.append(f"## System Prompt")
          md_content.append(f"")
          md_content.append(system_message.content)
          md_content.append(f"")
      
      # 添加消息历史
      md_content.append(f"## Message History")
      md_content.append(f"")
      for i, msg in enumerate(messages):
          md_content.append(f"### Message {i+1} ({type(msg).__name__})")
          md_content.append(f"")
          
          # 消息内容
          content = msg.content if hasattr(msg, 'content') else str(msg)
          if content:
              md_content.append(f"**Content**:")
              md_content.append(f"")
              md_content.append(content)
              md_content.append(f"")
          
          # 工具调用
          if hasattr(msg, 'tool_calls') and msg.tool_calls:
              md_content.append(f"**Tool Calls**:")
              md_content.append(f"")
              for j, tool_call in enumerate(msg.tool_calls):
                  md_content.append(f"#### Tool Call {j+1}")
                  md_content.append(f"")
                  md_content.append(f"- **ID**: {tool_call.get('id', 'N/A')}")
                  md_content.append(f"- **Name**: {tool_call.get('name', 'N/A')}")
                  md_content.append(f"- **Arguments**: {json.dumps(tool_call.get('args', {}), indent=2, ensure_ascii=False)}")
                  md_content.append(f"")
          
          # 工具调用ID
          if hasattr(msg, 'tool_call_id') and getattr(msg, 'tool_call_id', None):
              md_content.append(f"**Tool Call ID**: {getattr(msg, 'tool_call_id')}")
              md_content.append(f"")
      
      # 保存到文件，文件名加上时间戳，将call放到最后
      timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
      date_dir = datetime.now().strftime("%Y-%m-%d")
      dated_log_dir = os.path.join(self.log_dir, date_dir)
      os.makedirs(dated_log_dir, exist_ok=True)
      log_file = os.path.join(dated_log_dir, f"{timestamp_filename}_{self.call_count:03d}_call.md")
      with open(log_file, 'w', encoding='utf-8') as f:
          f.write('\n'.join(md_content))
      
      print(f"Log saved to: {log_file}\n")
  
  def _log_response(self, state: Dict[str, Any]):
      """在after_model中记录响应信息的方法"""
      # 如果未启用，则直接返回
      if not self.enabled:
          return
          
      self.call_count = state.get('call_count', 1)
      # print(f"read call_count: {self.call_count}")
          
      # 获取最新的消息作为响应
      messages = state.get('messages', [])
      if not messages:
          return
          
      # 创建Markdown格式的日志内容
      timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      md_content = []
      md_content.append(f"# Agent Response #{self.call_count}")
      md_content.append(f"")
      md_content.append(f"- **Timestamp**: {timestamp_str}")
      md_content.append(f"- **Messages Count**: {len(messages)}")
      md_content.append(f"")
      
      # 添加完整的消息历史
      md_content.append(f"## Complete Message History")
      md_content.append(f"")
      for i, msg in enumerate(messages):
          md_content.append(f"### Message {i+1} ({type(msg).__name__})")
          md_content.append(f"")
          
          # 消息内容
          content = msg.content if hasattr(msg, 'content') else str(msg)
          if content:
              md_content.append(f"**Content**:")
              md_content.append(f"")
              md_content.append(content)
              md_content.append(f"")
          
          # 工具调用
          if hasattr(msg, 'tool_calls') and msg.tool_calls:
              md_content.append(f"**Tool Calls**:")
              md_content.append(f"")
              for j, tool_call in enumerate(msg.tool_calls):
                  md_content.append(f"#### Tool Call {j+1}")
                  md_content.append(f"")
                  md_content.append(f"- **ID**: {tool_call.get('id', 'N/A')}")
                  md_content.append(f"- **Name**: {tool_call.get('name', 'N/A')}")
                  md_content.append(f"- **Arguments**: {json.dumps(tool_call.get('args', {}), indent=2, ensure_ascii=False)}")
                  md_content.append(f"")
          
          # 工具调用ID
          if hasattr(msg, 'tool_call_id') and getattr(msg, 'tool_call_id', None):
              md_content.append(f"**Tool Call ID**: {getattr(msg, 'tool_call_id')}")
              md_content.append(f"")
      
      # 保存到文件，文件名加上时间戳，将response放到最后
      timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
      date_dir = datetime.now().strftime("%Y-%m-%d")
      dated_log_dir = os.path.join(self.log_dir, date_dir)
      os.makedirs(dated_log_dir, exist_ok=True)
      log_file = os.path.join(dated_log_dir, f"{timestamp_filename}_{self.call_count:03d}_response.md")
      with open(log_file, 'w', encoding='utf-8') as f:
          f.write('\n'.join(md_content))
      
      self.call_count += 1
      print(f"call_count+1 = {self.call_count}")
      print(f"Response log saved to: {log_file}\n")

class PromptLoggerNodeMiddleware(PromptLoggerBaseMiddleware):
    def after_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
          """在模型调用后记录响应信息"""
          # print(f"PromptLoggerNodeMiddleware: state: {state}")
          # print(f"PromptLoggerNodeMiddleware: runtime: {runtime}")
          if self.enabled:
              # print("PromptLoggerNodeMiddleware: After model call...")
              self._log_response(state)
          # print(f"PromptLoggerNodeMiddleware: update call_count: {self.call_count}")
          return {
            "call_count": self.call_count
          }

class PromptLoggerWrapperMiddleware(PromptLoggerBaseMiddleware):
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """拦截模型调用以记录提示信息"""
        # print(f"PromptLoggerWrapperMiddleware: request: {request}")
        if self.enabled:
            # print("PromptLoggerWrapperMiddleware: Wrap model call...")
            self._log_request(request)
        # 调用原始处理函数并返回结果
        return handler(request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步拦截模型调用以记录提示信息"""
        # print(f"PromptLoggerWrapperMiddleware: request: {request}")
        if self.enabled:
            # print("PromptLoggerWrapperMiddleware: Async wrap model call...")
            self._log_request(request)
        # 异步调用原始处理函数并返回结果
        return await handler(request)