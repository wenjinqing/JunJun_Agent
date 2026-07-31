"""能力查询 skill：get_capabilities（对齐原 capabilities 插件语义）。
用户身份解析 skill：find_user_id（昵称 -> QQ 号，关系/画像类工具的前置）。"""

from langchain_core.tools import tool

from junjun_skills.builtin.memory_skills import current_chat_id


@tool
def find_user_id(nickname: str) -> str:
    """按昵称查一个人的 QQ 号。要惩罚/查画像/记称呼的对象只知道昵称时，先调这个拿到 QQ 号。
    当前会话的发言人优先，其次全库历史消息；支持模糊匹配，多人同名会列出候选。

    Args:
        nickname: 对方在群里的昵称（或名片），如「白菜兔」
    """
    from junjun_core.database import Messages
    nick = (nickname or "").strip()
    if not nick:
        return "昵称是空的，查不了。"
    if nick.isdigit():
        return f"{nick} 本身就是 QQ 号，直接用。"
    try:
        rows = list(Messages.select()
                    .where((Messages.user_nickname.contains(nick))
                           & (Messages.is_bot == False)  # noqa: E712
                           & (Messages.user_id != ""))
                    .order_by(Messages.time.desc()).limit(50))
    except Exception as e:
        return f"查询失败（{type(e).__name__}），稍后再试。"
    if not rows:
        return (f"没找到昵称含「{nick}」的人（ta 最近可能没说过话）。"
                f"可以让对方说句话，或直接问管理员要 QQ 号。")
    chat_id = current_chat_id.get()
    # 去重保序：精确匹配优先，当前会话优先，时间近优先
    seen, candidates = set(), []
    for r in sorted(rows, key=lambda r: (r.user_nickname != nick,
                                         bool(chat_id) and r.chat_id != chat_id)):
        if r.user_id in seen:
            continue
        seen.add(r.user_id)
        candidates.append(r)
        if len(candidates) >= 3:
            break
    if len(candidates) == 1:
        r = candidates[0]
        where = "当前会话" if r.chat_id == chat_id else "历史消息"
        return f"「{r.user_nickname}」的 QQ 号是 {r.user_id}（{where}）。"
    lines = [f"昵称含「{nick}」的有 {len(candidates)} 个人，确认是哪一个："]
    for r in candidates:
        lines.append(f"- {r.user_nickname}：QQ {r.user_id}")
    return "\n".join(lines)


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

    if query_type in ("all", "skills"):
        builtin = by_plugin.get("builtin", [])
        if builtin:
            parts.append("## 内置 Skill")
            for s in builtin:
                parts.append(f"- {s['name']}: {s['description'][:50]}")

    if query_type in ("all", "plugins"):
        # 插件工具（ai_draw/music/junzone 等）按插件分组显示
        plugin_tools = {p: items for p, items in by_plugin.items() if p not in ("builtin", "mcp")}
        if plugin_tools:
            parts.append("## 插件功能")
            for plugin, items in sorted(plugin_tools.items()):
                parts.append(f"- {plugin}: {len(items)} 个工具")

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


from junjun_agent.commands import register_command  # noqa: E402


@register_command("forget", aliases=["忘掉"], plugin="builtin",
                  admin_only=True, description="删除含关键词的长期记忆")
async def forget_cmd(ctx) -> str:
    """/forget <关键词>：删除所有含该关键词的长期记忆（含向量索引重建）。"""
    kw = (ctx.args or "").strip()
    if not kw:
        return "用法：/forget <关键词>"
    from junjun_memory.long_term import get_long_term_memory
    removed = get_long_term_memory().remove_where(lambda it: kw in it.text)
    if removed:
        return f"已删除 {removed} 条含「{kw}」的记忆。"
    return f"没找到含「{kw}」的记忆。"


@register_command("diary", aliases=["日记"], plugin="builtin",
                  admin_only=True, description="看日记（/diary now = 立即写今天的）")
async def diary_cmd(ctx) -> str:
    """/diary [日期|now]：查看最新/指定日期的日记；now 强制重写今天的。"""
    from junjun_express import diary as diary_mod

    arg = (ctx.args or "").strip()
    if arg == "now":
        content = await diary_mod.write_diary(force=True)
        return f"今天的日记写好了：\n{content}" if content else "写日记失败了，看看日志。"
    day = arg or None
    if day is None:
        from junjun_core.database.models import DiaryEntry
        row = DiaryEntry.select().order_by(DiaryEntry.date.desc()).first()
    else:
        row = diary_mod._get_entry(day)
    if row is None:
        return f"还没有{'这一天' if day else ''}的日记（每天 {diary_mod._cfg().get('time', '23:30')} 自动写）。"
    mood_txt = f"\n心情：{row.mood}" if row.mood else ""
    return f"「{row.date}」的日记：\n{row.content}{mood_txt}"
