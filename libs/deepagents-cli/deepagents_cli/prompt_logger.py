from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
import json
from datetime import datetime
from typing import Callable, Awaitable, Dict, Any
import os
from deepagents.middleware.prompt_logger import PromptLoggerBaseMiddleware

from deepagents.utils import load_env_with_fallback_verbose

# 加载环境变量（仅当PROMPT_LOGGER_ENABLED未设置时）
if os.getenv('PROMPT_LOGGER_ENABLED') is None:
    load_env_with_fallback_verbose()

class PromptLoggerWrapperMiddleware(PromptLoggerBaseMiddleware):
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """拦截模型调用以记录提示信息"""
        if self.enabled:
            print("PromptLoggerWrapperMiddleware: Wrap model call...")
            self._log_request(request)
        # 调用原始处理函数并返回结果
        return handler(request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步拦截模型调用以记录提示信息"""
        if self.enabled:
            print("PromptLoggerWrapperMiddleware: Async wrap model call...")
            self._log_request(request)
        # 异步调用原始处理函数并返回结果
        return await handler(request)