根据对代码和示例文件的分析，我可以为您总结skills目录下的处理逻辑：

## Skills 功能处理逻辑总结

### 1. 整体架构

Skills系统采用Anthropic提出的"Agent Skills"模式，通过YAML前端元数据解析实现技能管理。每个技能是一个包含SKILL.md文件的目录。

### 2. 核心组件

#### load.py - 技能加载器
- 负责从SKILL.md文件中解析YAML前端元数据
- 实现安全路径检查，防止目录遍历攻击
- 支持用户级和项目级技能加载
- 项目技能会覆盖同名的用户技能

技能文件结构：
```
skills/
├── skill-name/
│   ├── SKILL.md        # 必需：带YAML前端的指令
│   ├── script.py       # 可选：支持文件
│   └── config.json     # 可选：支持文件
```

#### commands.py - 命令行接口
提供三个主要命令：
- `skills list`: 列出所有可用技能
- `skills create`: 创建新技能模板
- `skills info`: 显示特定技能详细信息

实现了安全验证机制，防止路径遍历攻击。

#### middleware.py - 中间件集成
- 在代理执行前加载技能元数据
- 将技能信息注入系统提示词中
- 实现渐进式披露机制：代理只知道技能的存在（名称+描述），只在需要时才读取完整内容

### 3. 技能格式规范

SKILL.md文件包含：
1. YAML前端（必须包含name和description）
2. Markdown格式的代理使用说明
3. 可选的支持文件（脚本、配置等）

示例格式：
```markdown
---
name: web-research
description: Structured approach to conducting thorough web research
---

# Web Research Skill
...
```

### 4. 多层级技能管理

支持两种技能存储位置：
- 用户级技能：`~/.deepagents/{AGENT_NAME}/skills/`
- 项目级技能：`{PROJECT_ROOT}/.deepagents/skills/`

项目级技能优先于同名用户级技能。

### 5. 安全机制

- 文件大小限制（最大10MB）
- 路径安全检查，防止目录遍历
- 名称验证，只允许字母数字字符、连字符和下划线

### 6. 渐进式披露模式

这是skills系统的核心理念：
1. 代理在系统提示中只看到技能名称和简要描述
2. 当任务需要时，代理才会读取完整的SKILL.md内容
3. 这种方式既让代理知道可用技能，又不会过度占用上下文空间

这种设计使得代理可以灵活地使用各种专业技能，同时保持高效和专注。