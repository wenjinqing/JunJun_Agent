"""能力查询 skill：get_capabilities（对齐原 capabilities 插件语义）。"""

from langchain_core.tools import tool

from junjun_skills.builtin.memory_skills import current_chat_id


@tool
def get_capabilities(query_type: str = "all") -> str:
    """查询你当前已启用的所有功能（插件、skill、MCP 工具、命令）。

    当用户问"你有什么功能"、"你能做什么"、"你会什么"、"你的 skill 有哪些"、
    "你的 MCP 有哪些"时使用。

    Args:
        query_type: 查询类型：all(所有)、skills(内置 skill)、mcp(MCP 工具)、commands(命令)、plugins(插件)
    """
    from junjun_skills.registry import list_skills
    from junjun_agent.commands import list_commands
    from junjun_agent.interceptors import list_interceptors

    skills = list_skills()
    commands = list_commands()
    interceptors = list_interceptors()

    # 按插件分组
    by_plugin = {}
    for s in skills:
        if not s["enabled"]:
            continue
        by_plugin.setdefault(s["plugin"], []).append(s)

    parts = []

    if query_type in ("all", "plugins"):
        parts.append("## 已启用的插件")
        for plugin, items in sorted(by_plugin.items()):
            if plugin != "builtin":
                parts.append(f"- {plugin}: {len(items)} 个工具")

    if query_type in ("all", "skills"):
        builtin = by_plugin.get("builtin", [])
        if builtin:
            parts.append("## 内置 Skill")
            for s in builtin:
                parts.append(f"- {s['name']}: {s['description'][:50]}")

    if query_type in ("all", "mcp"):
        # MCP 工具在 registry 里 plugin="mcp"，name 带 mcp_ 前缀
        mcp_tools = [s for s in skills if s["enabled"] and s["plugin"] == "mcp"]
        if mcp_tools:
            parts.append("## MCP 工具")
            for s in mcp_tools[:20]:  # 最多列 20 个
                parts.append(f"- {s['name']}: {s['description'][:50]}")
            if len(mcp_tools) > 20:
                parts.append(f"  ... 共 {len(mcp_tools)} 个")
        else:
            parts.append("## MCP 工具：当前无已连接的 MCP server")

    if query_type in ("all", "commands"):
        if commands:
            parts.append("## 可用命令")
            for c in commands[:15]:
                raw_mark = "（关键词触发）" if c.get("raw") else ""
                parts.append(f"- /{c['name']}{raw_mark}: {c.get('description', '')[:40]}")

    if not parts:
        return "当前没有启用的功能模块。"

    return "\n".join(parts)
