"""Shared utilities for deepagents package."""

import os
from pathlib import Path
from typing import Optional

import dotenv


def load_env_with_fallback_verbose(required_vars: Optional[list] = None, agent_name: Optional[str] = None) -> Optional[str]:
    """
    Enhanced environment variable loading with detailed logging and required variable validation
    
    Args:
        required_vars: List of required environment variables
        agent_name: Optional agent name for resolve loading path
    Returns:
        Path to loaded .env file, or None if not found
    """
    if required_vars is None:
        required_vars = []
    
    search_paths = [
        ("当前工作目录", Path.cwd() / '.env')
    ]
    if agent_name:
        search_paths.append(("agent目录", Path.home() / '.deepagents' / agent_name / '.env'))
    search_paths.append(("用户配置目录", Path.home() / '.deepagents-cli' / '.env'))
    
    print("🔍 开始查找 .env 文件...")
    
    for location_name, env_path in search_paths:
        print(f"  检查 {location_name}: {env_path}")
        
        if env_path.exists() and env_path.is_file():
            # Load environment variables
            dotenv.load_dotenv(env_path)
            print(f"✅ 从 {location_name} 加载环境变量: {env_path}")
            
            # Validate required variables
            if required_vars:
                missing_vars = []
                for var in required_vars:
                    if not os.getenv(var):
                        missing_vars.append(var)
                
                if missing_vars:
                    print(f"⚠️  警告: 以下必需变量未设置: {missing_vars}")
                else:
                    print("✅ 所有必需环境变量都已设置")
            
            return str(env_path)
        else:
            print(f"   ❌ 文件不存在")
    
    print("❌ 在所有搜索路径中均未找到 .env 文件")
    return None