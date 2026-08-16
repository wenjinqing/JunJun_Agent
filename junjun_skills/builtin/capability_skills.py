"""能力查询 skill：get_capabilities（对齐原 capabilities 插件语义）。
用户身份解析 skill：find_user_id（昵称 -> QQ 号，关系/画像类工具的前置）。
自我介绍 skill：introduce_self（身份+能力分类+技术栈的策展简介，2026-08-15）。"""

from langchain_core.tools import tool

from junjun_skills.builtin.memory_skills import current_chat_id


# introduce_self 的能力分类映射：只列对用户可说的插件（键=插件目录名）。
# 不在表里的插件不出现——内部 loop 件（topic_finder）、敏感件（netdisk）、
# 未启用件自动隐身；新插件要进简介就在这加一行。
_INTRO_CATEGORIES = {
    "google_search": "联网搜索（网页/图片）",
    "async_task": "深度调研与后台长任务",
    "bilibili": "B 站视频（总结/看片讲内容）",
    "douyin": "抖音视频解析",
    "ai_draw": "AI 画图",
    "tts": "语音合成",
    "ja_tts": "日语语音",
    "music": "点歌放歌",
    "pixiv": "P 站搜图与画师订阅",
    "subscription": "订阅盯梢（作者/UP 主更新提醒）",
    "daily_report": "热点日报",
    "weekly_report": "每周周报",
    "news": "新闻速览",
    "workspace": "工作区（收发文件/读写/跑代码/处理表格图表）",
    "junzone": "QQ 空间（发说说/刷好友动态）",
    "fun_texts": "群娱乐（美文/整活文案）",
    "jrys": "今日运势",
    "wife": "群老婆抽签",
    "chat_screenshot": "聊天记录截图",
    "cross_scene": "跨群围观",
    "intimacy": "亲密度",
    "emoji_manage": "表情包收藏管理",
    "image_viewer": "识图（看图说话）",
}

# 技术栈策展文案：组件级，不含版本号/端点/接入平台名/路径（说了也没用，
# 还白送攻击面）。2026-08-15 用户拍板：具体模型型号不对外说——介绍重点是
# agent 架构与功能；文案里不留型号也根治「被问型号就现场编」（此前模型曾
# 臆造「混元大模型」），末尾一句给「不公开」的标准话术。要改就改这里，
# 别让模型现场编。
_INTRO_TECH = (
    "技术栈与架构：Python 写的；决策内核跑在 LangChain + LangGraph 上"
    "（工具调用循环，复杂委托自动规划成步骤图执行，做不顺会自己重规划换路子）；"
    "三层记忆（短期对话/长期事实/语义召回）记得住事；几十件工具按场景动态挂载；"
    "QQ 接入走 NapCat（OneBot 协议）；数据落本地 SQLite；"
    "跑代码有独立的 Docker 沙箱（隔离、无网络）。"
    "具体模型型号不公开——被追问就笑着带过，把话题引到能力上。"
)


@tool
def introduce_self() -> str:
    """自我介绍。对方问「你是谁」「你是什么做的/什么模型」「你会什么（要简介不要
    逐条清单）」「介绍下你自己」「你的技术栈」时使用。返回身份速写+能力分类概览+
    技术栈，用你自己的口气自然转述，别照念。
    对方明确问「什么模型/什么框架/技术栈」时，技术栈一段必须如实转述给对方
    （可以换成你的口气说）：组件名一个不许省（LangChain/LangGraph、三层记忆、
    NapCat、SQLite、Docker 沙箱都要点到），不许略过、不许说「不清楚」、
    不许拿「跑在服务器上的程序」一句概括掉——那不是秘密，是你的底牌。
    但具体模型型号不对外说：被追问型号就笑着带过，把话题引到能力上，绝不编型号。
    区别于 get_capabilities（那是逐条列工具的完整清单）——本工具是给陌生人
    的第一印象简介。内容已过安全筛：不含密钥/QQ 号/内部路径/供应商细节，
    直接说出去不泄密。"""
    from junjun_core.config import get_global_config
    from junjun_skills.registry import list_skills

    cfg = get_global_config()
    nickname = cfg.bot.nickname
    from junjun_agent.persona import persona_brief

    # 启用的插件 -> 分类标签（按 _INTRO_CATEGORIES 表序输出，稳定可读）
    enabled = {}
    for s in list_skills():
        if s["enabled"] and s["plugin"] != "builtin":
            enabled[s["plugin"]] = enabled.get(s["plugin"], 0) + 1
    caps = [label for name, label in _INTRO_CATEGORIES.items() if enabled.get(name)]
    n_tools = sum(1 for s in list_skills() if s["enabled"])

    lines = [
        f"我是{nickname}——{persona_brief()}",
        f"本体是个跑在服务器上的 AI 程序（被问起就大方承认），目前挂着 {n_tools} 件工具。",
        "平时能干的：" + "、".join(caps) + "。"
        if caps else "",
        "内置基本功：设提醒、记事回忆、查天气、翻聊天记录、发表情。",
        _INTRO_TECH,
        "想看逐条的完整能力清单就调 get_capabilities。",
    ]
    return "\n".join(l for l in lines if l)


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

    # 工具健康度（P5-4）：被问「你有什么功能」时如实上报故障中的功能
    try:
        from junjun_skills.health import degraded_tools
        degraded = degraded_tools()
        if degraded:
            parts.append("## 故障中（最近持续失败，在修）")
            for d in degraded:
                parts.append(f"- {d['tool']}: {d['kind']}类故障，"
                             f"连续失败 {d['fails']} 次")
    except Exception:
        pass

    return "\n".join(parts)


from junjun_agent.commands import register_command  # noqa: E402


@register_command("remember", aliases=["记住"], plugin="builtin",
                  description="钉住一条必须记住的事（每轮都会看到）")
async def remember_cmd(ctx) -> str:
    """/记住 <内容>：钉进最高优先级记忆（kind=pinned，每轮注入，不占召回额度）。"""
    content = (ctx.args or "").strip()
    if not content:
        return "用法：/记住 <内容>"
    from junjun_memory.long_term import get_long_term_memory
    from junjun_core.config import get_global_config
    chat_id = ctx.session.chat_id
    ltm = get_long_term_memory()
    try:
        cap = int(get_global_config().raw.get("memory", {}).get("pinned_max_per_chat", 20))
    except Exception:
        cap = 20
    if len(ltm.pinned(chat_id)) >= cap:
        return f"钉住的记忆已经 {cap} 条到上限了，先 /忘掉 一些再钉。"
    await ltm.add(content, chat_id, weight=1.5, kind="pinned")
    return f"钉好了：{content}（之后每轮我都会直接看到）"


@register_command("forget", aliases=["忘掉"], plugin="builtin",
                  description="删除含关键词的记忆（管理员删全局，其他人限本会话+本人画像）")
async def forget_cmd(ctx) -> str:
    """/forget <关键词>：删除含该关键词的长期记忆（含向量索引重建）。

    权限边界：管理员全局删除；其他人只删本会话的记忆（知识库/日记不动）
    + 本人画像里的记忆点。可控记忆只碰事实性记忆，人设/安全规则不在
    长期记忆库里，天然不可触及。
    """
    kw = (ctx.args or "").strip()
    if not kw:
        return "用法：/forget <关键词>"
    from junjun_memory.long_term import get_long_term_memory
    from junjun_core.security import is_admin
    ltm = get_long_term_memory()
    if is_admin(ctx.meta.user_id):
        removed = ltm.remove_where(lambda it: kw in it.text)
        scene_removed = 0
        try:
            from junjun_core.database.models import UserSceneProfile, _bot_id
            scene_removed = (UserSceneProfile.delete()
                             .where((UserSceneProfile.bot_id == _bot_id())
                                    & (UserSceneProfile.content.contains(kw)))
                             .execute())
        except Exception:
            pass
        if removed or scene_removed:
            extra = f"（另删跨场景档案 {scene_removed} 条）" if scene_removed else ""
            return f"已删除 {removed} 条含「{kw}」的记忆（全局）。{extra}"
        return f"没找到含「{kw}」的记忆。"
    chat_id = ctx.session.chat_id
    removed = ltm.remove_where(
        lambda it: kw in it.text and it.chat_id == chat_id
        and it.chat_id not in ("knowledge", "self:diary", "self:diary:private"))
    profile_removed = 0
    scene_removed = 0
    try:
        from junjun_memory.user_profile import get_profile_store
        profile_removed = get_profile_store().remove_points_where(
            ctx.session.platform, ctx.meta.user_id, kw)
    except Exception:
        pass
    try:
        from junjun_memory.scene_profile import forget_user_facts
        scene_removed = forget_user_facts(
            ctx.session.platform, ctx.meta.user_id, kw,
            admin=False, current_chat_id=chat_id)
    except Exception:
        pass
    if removed or profile_removed or scene_removed:
        parts = []
        if removed:
            parts.append(f"本会话记忆 {removed} 条")
        if profile_removed:
            parts.append(f"你的画像 {profile_removed} 条")
        if scene_removed:
            parts.append(f"你的跨场景档案 {scene_removed} 条")
        return f"已删除含「{kw}」的：{'、'.join(parts)}。"
    return f"没找到含「{kw}」的记忆（只能删本会话的记忆和你自己的画像）。"


@register_command("what_do_you_remember", aliases=["你记得我什么"], plugin="builtin",
                  description="导出她记住的关于你的画像")
async def what_do_you_remember_cmd(ctx) -> str:
    """/你记得我什么：导出她记住的关于你的画像（只看自己的，按用户隔离）。"""
    from junjun_memory.user_profile import get_profile_store
    store = get_profile_store()
    person = store.get_or_create(ctx.session.platform, ctx.meta.user_id)
    points = store.get_points(ctx.session.platform, ctx.meta.user_id, top_k=20)
    if not points and not person.person_name:
        return "我还不太了解你呢——多跟我聊聊你自己，或者直接用 /记住 告诉我。"
    lines = [f"我记得的关于你（{person.person_name or ctx.meta.nickname or '你'}）："]
    for p in points:
        lines.append(f"- {p['category']}: {p['content']}")
    lines.append("想删掉某条：/忘掉 <关键词>")
    return "\n".join(lines)


@register_command("quiet", aliases=["安静"], plugin="builtin",
                  description="本会话安静模式：不再主动发消息（/热闹 解除）")
async def quiet_cmd(ctx) -> str:
    """/安静：本会话持久静音主动消息（意向+主动搭话统一生效；提醒类必达不受影响）。"""
    from junjun_agent.loop.intention import mute_chat
    mute_chat(ctx.session.chat_id, hours=0)  # 持久，/热闹 解除
    return "好，我在这安静啦——不主动找你们了，想我热闹回来就说 /热闹。"


@register_command("lively", aliases=["热闹"], plugin="builtin",
                  description="解除安静模式，恢复主动消息")
async def lively_cmd(ctx) -> str:
    """/热闹：解除本会话安静模式。"""
    from junjun_agent.loop.intention import unmute_chat
    unmute_chat(ctx.session.chat_id)
    return "好嘞，我又会主动来找你们啦～"


@register_command("identity", aliases=["自我"], plugin="builtin",
                  description="看她从经历里长出来的自我认知")
async def identity_cmd(ctx) -> str:
    """/自我：查看她蒸馏出的自我认知条目（Identity Core）。"""
    from junjun_express import identity as id_mod
    rows = id_mod.get_entries(limit=20)
    if not rows:
        return "她还没攒够日记蒸馏自我认知（每周一次，至少 3 篇新日记）。"
    lines = ["她的自我认知（从日记里长出来的）："]
    for r in rows:
        lines.append(f"- {r.category}：{r.content}")
    return "\n".join(lines)


@register_command("reset_identity", aliases=["重置自我"], plugin="builtin",
                  admin_only=True, description="归档全部自我认知条目（重新长）")
async def reset_identity_cmd(ctx) -> str:
    """/重置自我（管理员）：自我认知长歪时的兜底——全部归档重来。"""
    from junjun_express import identity as id_mod
    n = id_mod.reset_identity()
    return f"已归档 {n} 条自我认知，之后随日记蒸馏重新长。"


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


@register_command("patches", aliases=["补丁"], plugin="builtin", admin_only=True,
                  description="技能补丁管理（/补丁 list|启用 N|回滚 N）")
async def patches_cmd(ctx) -> str:
    """/补丁（管理员，P8-2）：经验回放补丁的人工门控。
    list 看候选与状态；启用 N 注入工具说明；回归就回滚 N。"""
    from junjun_skills import patches as p_mod

    arg = (ctx.args or "").strip()
    if not arg or arg == "list":
        rows = p_mod.list_patches()
        if not rows:
            return ("还没有技能补丁。工具失败攒够（默认 3 次/7 天）且 "
                    "[evolution] enable=true 后，每周复盘会自动产出候选。")
        icon = {"candidate": "🕐", "active": "✅", "rolled_back": "↩️", "merged": "🗜"}
        lines = ["技能补丁（候选需 /补丁 启用 N 人工启用）："]
        for r in rows:
            lines.append(f"{icon.get(r['status'], '?')} #{r['id']} [{r['tool']}] v{r['version']} "
                         f"{r['patch'][:36]}（{r['status']}）")
        return "\n".join(lines)
    for verb, fn in (("启用", p_mod.activate), ("回滚", p_mod.rollback)):
        if arg.startswith(verb):
            num = arg[len(verb):].strip()
            if not num.isdigit():
                return f"用法：/补丁 {verb} 编号"
            return fn(int(num))
    return "用法：/补丁 list | /补丁 启用 编号 | /补丁 回滚 编号"
