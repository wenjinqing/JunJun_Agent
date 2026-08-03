"""Skill 注册表：LangChain @tool 的统一注册与按会话过滤。"""

from typing import Callable, Dict, List, Optional

from langchain_core.tools import BaseTool

from junjun_core.observability import get_logger

logger = get_logger("skills.registry")

_registry: Dict[str, BaseTool] = {}
# skill 名 -> 可用性判断（None = 全会话可用）；session 由 gateway 传入
_availability: Dict[str, Optional[Callable]] = {}
# WebUI 插件管理：被禁用的 skill 不进 tools（默认全启用）
_disabled: set = set()
# 插件级管理：skill 名 -> 插件名；被禁用的插件其工具/命令/拦截器全部失效
_skill_plugin: Dict[str, str] = {}
_plugin_disabled: set = set()


def register(skill: BaseTool, available_for: Optional[Callable] = None,
             plugin: str = "builtin", admin_only: bool = False) -> None:
    """注册 skill。重名直接报错（拒绝静默覆盖）。

    available_for: (session) -> bool，None 表示全会话可用。
    plugin: 所属插件名（WebUI 插件级禁用用）。
    admin_only: True 时包一层权限门——非管理员调用直接拒绝并上报
                （security.report_violation），不进入工具本体。
                与工具内部的自定义校验可共存（框架门先触发，不会双重上报）。
    """
    if skill.name in _registry:
        raise ValueError(f"skill 重名: {skill.name}")
    _relax_str_args(skill)
    if admin_only:
        skill = _wrap_admin_gate(skill)
    skill = _wrap_error_feedback(skill)  # 最外层：逃逸异常 -> [TOOL_ERROR] 结构化文本
    _registry[skill.name] = skill
    _availability[skill.name] = available_for
    _skill_plugin[skill.name] = plugin
    try:
        from junjun_skills import patches
        patches.apply_to_registry(skill.name)  # P8-2：已有活跃补丁注入新注册工具
    except Exception:
        pass
    logger.debug(f"注册 skill: {skill.name} [{plugin}]{' (admin)' if admin_only else ''}")


# ---------------------------------------------------------------- str 参数宽松化
# 弱模型常把数字 ID 传成 JSON number（Qwen 尤甚：target=16689973 不带引号），
# pydantic v2 的 str 字段 lax 模式也不吃 int -> 校验报错，模型还多半原样重试
# 烧穿迭代上限（2026-08-01 实战 trace：5 次同样报错 -> recursion limit -> 沉默）。
# 服务端统一宽松：str 字段包 BeforeValidator 把数字转字符串。
# 模型侧 JSON schema 不变（仍声明 string），只影响服务端校验。

def _num_to_str(v):
    if isinstance(v, bool):
        return v  # bool 是 int 子类，str(True)="True" 更糟，留给校验报错
    if isinstance(v, (int, float)):
        return str(v)
    return v


def _relax_str_args(skill: BaseTool) -> None:
    """str 参数服务端宽松化（in-place 重建 args_schema）。非 pydantic schema 跳过。"""
    from typing import Annotated
    from pydantic import BeforeValidator, Field, create_model
    schema = getattr(skill, "args_schema", None)
    if schema is None or not hasattr(schema, "model_fields"):
        return
    defs = {}
    relaxed = []
    for name, f in schema.model_fields.items():
        if f.annotation is str:
            # FieldInfo 直接当 default 传会丢 description（模型理解参数的命根子），显式重建
            new_field = Field(default=f.default, description=f.description,
                              json_schema_extra=f.json_schema_extra)
            defs[name] = (Annotated[str, BeforeValidator(_num_to_str)], new_field)
            relaxed.append(name)
        else:
            defs[name] = (f.annotation, f)
    if relaxed:
        try:
            skill.args_schema = create_model(f"{schema.__name__}Lax", **defs)
            logger.debug(f"{skill.name} str 参数宽松化: {relaxed}")
        except Exception as e:
            logger.debug(f"{skill.name} args_schema 重建失败（保持原样）: {e}")


def _admin_refusal(tool_name: str, args: tuple, kwargs: dict) -> str:
    from junjun_core.security import current_nickname, current_user_id, report_violation
    from junjun_skills.builtin.memory_skills import current_chat_id
    detail = " ".join(str(a) for a in (*args, *kwargs.values()))[:80]
    report_violation(f"管理员工具 {tool_name}", current_user_id.get(),
                     current_nickname.get(),
                     current_chat_id.get(), detail)
    return "（权限不足：这个操作只有管理员能做，已通知管理员）"


def _wrap_admin_gate(skill: BaseTool) -> BaseTool:
    """给工具包管理员权限门（运行时按真实发送者 QQ 判定，LLM 不可伪造）。"""
    from junjun_core.security import is_admin_privileged
    name = skill.name
    if getattr(skill, "coroutine", None) is not None:
        original = skill.coroutine

        async def gated(*args, _orig=original, **kwargs):
            if not is_admin_privileged():
                return _admin_refusal(name, args, kwargs)
            return await _orig(*args, **kwargs)
        skill.coroutine = gated
    elif getattr(skill, "func", None) is not None:
        original_sync = skill.func

        def gated_sync(*args, _orig=original_sync, **kwargs):
            if not is_admin_privileged():
                return _admin_refusal(name, args, kwargs)
            return _orig(*args, **kwargs)
        skill.func = gated_sync
    else:
        logger.warning(f"admin 门包装失败（无可包装入口）: {name}")
    return skill


# ---------------------------------------------------------------- 错误结构化（P0-13）
# 工具抛出的裸异常统一分类成 [TOOL_ERROR kind=...] 文本喂回模型，
# 配合 system prompt 里的「换乘地图」，模型能按类别决定重试/换乘/放弃，
# 而不是看到一句英文堆栈就沉默或胡说。

def _classify_error(e: BaseException) -> str:
    """异常 -> 错误类别：网络 / 参数 / 权限 / 限流 / 未知。"""
    import asyncio

    import httpx
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return "限流"
        if code in (401, 403):
            return "权限"
        return "网络"  # 5xx 等服务端错误按网络类处理（可换乘重试）
    if isinstance(e, PermissionError):
        return "权限"
    if isinstance(e, (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError,
                      asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return "网络"
    if isinstance(e, (ValueError, TypeError, KeyError)):
        return "参数"
    return "未知"


def _tool_error_text(tool_name: str, e: BaseException) -> str:
    kind = _classify_error(e)
    logger.warning(f"工具 {tool_name} 异常[{kind}]: {type(e).__name__}: {e}")
    detail = str(e).replace("\n", " ")[:150]
    return f"[TOOL_ERROR kind={kind}] 工具 {tool_name} 执行失败：{type(e).__name__}: {detail}"


def _wrap_error_feedback(skill: BaseTool) -> BaseTool:
    """统一错误包装（最外层）：工具逃逸的异常 -> 结构化错误文本，不再抛给框架。
    同时上报工具健康度（P5-4）：异常记失败、正常返回记成功（自动恢复）。"""
    name = skill.name
    if getattr(skill, "coroutine", None) is not None:
        original = skill.coroutine

        async def wrapped(*args, _orig=original, **kwargs):
            from junjun_skills import health, patches
            try:
                result = await _orig(*args, **kwargs)
            except Exception as e:
                health.record_fail(name, _classify_error(e), str(e))
                patches.log_failure(name, _classify_error(e), str(e))
                return _tool_error_text(name, e)
            health.record_ok(name)
            return result
        skill.coroutine = wrapped
    elif getattr(skill, "func", None) is not None:
        original_sync = skill.func

        def wrapped_sync(*args, _orig=original_sync, **kwargs):
            from junjun_skills import health, patches
            try:
                result = _orig(*args, **kwargs)
            except Exception as e:
                health.record_fail(name, _classify_error(e), str(e))
                patches.log_failure(name, _classify_error(e), str(e))
                return _tool_error_text(name, e)
            health.record_ok(name)
            return result
        skill.func = wrapped_sync
    return skill


def set_enabled(name: str, enabled: bool) -> bool:
    """启用/禁用 skill 或插件（WebUI 插件管理）。不存在返回 False。"""
    if name in _registry:
        if enabled:
            _disabled.discard(name)
        else:
            _disabled.add(name)
        logger.info(f"skill {name} 已{'启用' if enabled else '禁用'}")
        return True
    return set_plugin_enabled(name, enabled)


def set_plugin_enabled(name: str, enabled: bool) -> bool:
    """启用/禁用整个插件（工具 + 命令 + 拦截器）。插件不存在返回 False。"""
    if name not in _skill_plugin.values():
        return False
    if enabled:
        _plugin_disabled.discard(name)
    else:
        _plugin_disabled.add(name)
    logger.info(f"插件 {name} 已{'启用' if enabled else '禁用'}")
    return True


def is_plugin_enabled(name: str) -> bool:
    """插件是否启用（命令/拦截器总线查询用）。"""
    return name not in _plugin_disabled


def list_skills() -> List[dict]:
    """插件管理用：全部 skill 及启用状态。"""
    return [{"name": n, "description": (s.description or "")[:80],
             "plugin": _skill_plugin.get(n, "builtin"),
             "enabled": n not in _disabled and is_plugin_enabled(_skill_plugin.get(n, "builtin"))}
            for n, s in _registry.items()]


# ---------------------------------------------------------------- 三层工具子集（P5-2）

# CORE 层：任何对话都可能用的刚需工具，永不掩码，≤8 个（Berkeley
# Function-Calling Leaderboard：工具越多模型越难选对）。顺序即绑定顺序。
_CORE_TOOLS = (
    "do_not_reply",   # 沉默决策
    "send_message",   # 回复
    "send_emoji",     # 表情
    "get_time",       # 时间感知
    "recall_memory",  # 记忆读
    "save_memory",    # 记忆写
    "manage_mood",    # 心情
    "web_search",     # 搜索（答疑兜底）
)
_CORE_SET = frozenset(_CORE_TOOLS)

# INTENT 层：当前消息命中强意图 -> 整组挂载。元数据同时驱动 agent 意图自检
# （primary = 「必须真调」的工具，None = 只挂载不追问）。
# 顺序敏感：先长后短（「取消订阅」含「订阅」，必须先匹配）。
_INTENT_GROUPS = [
    (("取消订阅", "别盯", "退订", "不用盯"),
     ("unsubscribe", "list_subscriptions"), "unsubscribe"),
    (("盯", "订阅", "更新了告诉", "出新了告诉", "出新叫我"),
     ("subscribe_updates", "list_subscriptions"), "subscribe_updates"),
    (("提醒我", "记得提醒", "到点叫", "到时候叫"),
     ("set_reminder", "list_reminders", "cancel_reminder_task"), "set_reminder"),
    # 调研/报告类必须派后台深度研究（当场埋头做会堵会话队列几十秒，对方干等）
    (("调研", "深研", "深度研究", "研究报告", "整理一份报告"),
     ("deep_research", "run_background_task", "list_background_tasks"), "deep_research"),
    (("发个语音", "发语音", "语音说", "唱首", "唱一", "念给", "说日语", "日语说"),
     ("unified_tts", "ja_tts"), None),
    (("帮我画", "画个", "画一张", "画张", "生成图", "画幅"),
     ("ai_draw",), None),
]


def intent_groups() -> list:
    """INTENT 层元数据（agent 意图自检共用，单一数据源）。"""
    return _INTENT_GROUPS


def _intent_mounted(text: str) -> List[str]:
    """当前消息命中的意图组 -> 挂载工具名（组定义序，去重）。"""
    out: List[str] = []
    if not text:
        return out
    for keywords, group, _primary in _INTENT_GROUPS:
        if any(kw in text for kw in keywords):
            for name in group:
                if name not in out:
                    out.append(name)
    return out


def _canonical_order(tools: List[BaseTool]) -> List[BaseTool]:
    """稳定序列化：CORE 固定序 + 其余按名字典序。

    同一子集每次绑定的字节完全一致——tools 是请求前缀的一部分，
    顺序抖动会白白打穿 prompt 前缀缓存（P5-2 缓存护栏）。
    """
    rank = {name: i for i, name in enumerate(_CORE_TOOLS)}
    return sorted(tools, key=lambda t: (rank.get(t.name, len(rank)), t.name))


def get_tools(session=None) -> List[BaseTool]:
    """按会话取可用工具集。session=None 返回全量（不含已禁用）。

    Berkeley Function-Calling Leaderboard：超过 20 工具性能显著下降，
    动态选择/掩码是必需。按会话最近话题做 embedding 检索相关工具：
    核心工具（决策/记忆/时间/提醒）永远保留，其余按语义相关性取前 8 个。
    """
    tools = []
    for name, skill in _registry.items():
        if name in _disabled or not is_plugin_enabled(_skill_plugin.get(name, "builtin")):
            continue
        gate = _availability.get(name)
        if session is None or gate is None or gate(session):
            tools.append(skill)

    # 动态掩码：超过 15 个时按语义相关性裁剪（保留核心 + embedding 检索相关）
    # MCP 工具特殊处理：按 server 分组，每组保留前 3 个（防 70+ 工具全被裁）
    if len(tools) > 15 and session is not None:
        mcp_tools = [t for t in tools if t.name.startswith("mcp_")]
        other_tools = [t for t in tools if not t.name.startswith("mcp_")]
        # MCP 按 server 分组保留（每组前 3 个），不参与话题相关性裁剪
        mcp_by_server = {}
        for t in mcp_tools:
            # mcp_<server>_<tool> 格式，取 server 名分组
            parts = t.name.split("_", 2)
            server = parts[1] if len(parts) > 1 else "unknown"
            mcp_by_server.setdefault(server, []).append(t)
        mcp_kept = []
        for server, ts in mcp_by_server.items():
            mcp_kept.extend(ts[:3])  # 每组最多 3 个
        # 其他工具按话题相关性裁剪
        other_kept = _mask_by_relevance(other_tools, session) if len(other_tools) > 15 else other_tools
        tools = _canonical_order(other_kept + mcp_kept)
    return tools


def _mask_by_relevance(tools: List[BaseTool], session) -> List[BaseTool]:
    """三层工具子集（P5-2）：CORE 常驻 + INTENT 整组挂载 + TOPIC 关键词钉住，
    余量按语义相关性补满。

    - CORE（≤8，_CORE_TOOLS）：回复/表情/记忆/时间/心情/搜索，永不掩码
    - INTENT：当前消息命中强意图（订阅/提醒/调研/语音…）整组挂载，防漏绑
      （订阅要 subscribe+list 一起给，只给一个会空口答应）
    - TOPIC：最近 3 条消息关键词命中钉住（上限 6，防关键词风暴）
    - 补满：embedding 余弦排序（LangGraph 工作线程无事件循环时降级关键词）
    """
    core_tools = [t for t in tools if t.name in _CORE_SET]
    other_tools = [t for t in tools if t.name not in _CORE_SET]

    entries = session.memory.entries if session.memory else []
    recent_text = " ".join(e.text for e in entries[-3:])
    current_text = entries[-1].text if entries else ""

    # INTENT 层：强意图整组挂载（不占 TOPIC 名额）
    intent_names = set(_intent_mounted(current_text))
    intent_tools = [t for t in other_tools if t.name in intent_names]
    # TOPIC 层：关键词钉住（上限 6）
    topic_tools = [t for t in _pinned_by_keywords(other_tools, recent_text)
                   if t not in intent_tools][:6]
    pinned = intent_tools + topic_tools
    fill_budget = max(0, 8 - len(pinned))

    def _fill(scored):
        return [t for _, t in scored if t not in pinned][:fill_budget]

    # embedding 检索（不可用降级关键词）
    # 注意：不在非主线程调 get_event_loop()——LangGraph 的 asyncio_N 线程
    # 没有事件循环，会炸 RuntimeError。embedding 检索改为可选（有则用，无则跳过）。
    try:
        from junjun_memory.embedding import get_embedding_client
        client = get_embedding_client()
        if client.available and recent_text:
            import asyncio
            # 检查当前线程是否有事件循环（没有则跳过 embedding 检索，降级关键词）
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    raise RuntimeError("no running loop")
            except RuntimeError:
                loop = None
            if loop:
                query_vec = loop.run_until_complete(client.embed_one(recent_text))
                if query_vec:
                    import numpy as np
                    q = np.array(query_vec)
                    q /= (np.linalg.norm(q) + 1e-9)
                    scored = []
                    for t in other_tools:
                        desc_vec = loop.run_until_complete(client.embed_one(t.description or t.name))
                        if desc_vec:
                            d = np.array(desc_vec)
                            d /= (np.linalg.norm(d) + 1e-9)
                            score = float(np.dot(q, d))
                        else:
                            score = 0.0
                        scored.append((score, t))
                    scored.sort(key=lambda x: -x[0])
                    return core_tools + pinned + _fill(scored)
    except Exception:
        pass

    # 降级：关键词匹配
    recent_lower = recent_text.lower()
    scored = []
    for t in other_tools:
        score = sum(1 for kw in _TOPIC_KEYWORDS.get(t.name, []) if kw in recent_lower)
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    kept = core_tools + pinned + _fill(scored)
    logger.debug(f"工具掩码: {len(tools)} -> {len(kept)}"
                 f"（钉住 {[t.name for t in pinned]}）")
    return kept


def _pinned_by_keywords(tools: List[BaseTool], recent_text: str) -> List[BaseTool]:
    """话题关键词命中的工具（强词触发直接钉住，不参与相关性排序）。"""
    recent_lower = recent_text.lower()
    return [t for t in tools
            if any(kw in recent_lower for kw in _TOPIC_KEYWORDS.get(t.name, []))]


# 工具名 -> 话题关键词（embedding 降级时的兜底）
_TOPIC_KEYWORDS = {
    "ai_draw": ["画", "图", "生成", "照片", "图片"],
    "get_weather": ["天气", "下雨", "温度", "热", "冷"],
    "web_search": ["搜", "查", "找", "什么是", "是谁", "哪里"],
    "search_knowledge": ["知识", "资料", "设定", "文档"],
    "send_emoji": ["表情", "emoji", "图"],
    "query_jargon": ["黑话", "梗", "什么意思", "缩写"],
    "query_chat_history": ["上次", "之前说过", "那天说", "聊天记录", "翻一下", "搜一下记录"],
    "manage_user_profile": ["记住", "我叫", "我喜欢", "我的"],
    "pin_memory": ["记住", "别忘了", "给我记住", "一定要记住"],
    "set_reminder": ["提醒", "到点", "到时候叫"],
    "list_reminders": ["提醒"],
    "cancel_reminder_task": ["取消提醒", "别提醒", "不用提醒"],
    "find_user_id": ["qq号", "谁的qq"],
    "get_capabilities": ["你会什么", "你能干什么", "你会干啥", "有什么功能", "能做什么"],
    "vrchat_play_pose": ["动作", "跳舞", "挥手", "vrchat"],
    "unified_tts": ["语音", "说话", "念", "听", "声音", "唱"],
    "ja_tts": ["日语", "日文", "语音", "声音"],
    "send_poke": ["戳", "poke"],
    "bilibili": ["b站", "bilibili", "视频", "bv"],
    "bilibili_summary": ["b站", "bilibili", "视频讲", "讲了啥", "讲了什么", "bv", "视频内容", "什么视频"],
    "watch_video": ["认真看", "看完", "看看这个视频", "讲一下这个视频", "b站", "bilibili", "bv", "抖音", "douyin"],
    "douyin": ["抖音", "douyin"],
    "douyin_summary": ["抖音", "douyin", "讲了啥", "讲了什么", "什么内容", "怎么样"],
    "music": ["音乐", "歌", "点歌"],
    "send_feed": ["说说", "空间", "qzone", "动态"],
    "read_feed": ["说说", "空间", "qzone", "动态"],
    "delete_feed": ["删说说", "删除说说", "删掉说说", "空间"],
    "answer_book": ["答案之书", "该不该", "要不要", "会不会"],
    "fun_quote": ["毒鸡汤", "鸡汤", "丧"],
    "draw_lot": ["抽签", "灵签", "观音", "文昌", "求签"],
    "make_qrcode": ["二维码"],
    "decode_qrcode": ["二维码", "扫码"],
    "today_in_history": ["历史上的今天", "今天是什么日子"],
    "subscribe_updates": ["订阅", "盯", "关注", "更新了告诉", "出新", "up主", "p站", "pixiv"],
    "list_subscriptions": ["订阅", "盯"],
    "unsubscribe": ["取消订阅", "别盯", "退订", "不用盯"],
    "deep_research": ["调研", "深研", "深度研究", "研究报告", "整理一份", "报告", "系统查", "查资料"],
    "run_background_task": ["后台", "慢慢做", "不着急", "研究一下", "做好了叫", "总结一下这个", "读一下这个"],
    "list_background_tasks": ["后台任务", "任务进度", "做得怎么样", "做好了吗"],
    "cancel_background_task": ["取消任务", "别做了", "不用做了", "停掉"],
}


def clear() -> None:
    """仅供测试。"""
    _registry.clear()
    _availability.clear()
    _disabled.clear()
    _skill_plugin.clear()
    _plugin_disabled.clear()


def load_builtin() -> None:
    """加载内置 skill（幂等）。"""
    if _registry:
        return
    from junjun_skills.builtin.get_time import get_time
    from junjun_skills.builtin.do_not_reply import do_not_reply
    from junjun_skills.builtin.memory_skills import (
        recall_memory, save_memory, pin_memory, manage_user_profile, query_jargon, learn_jargon,
    )
    from junjun_skills.builtin.reminder_skills import (
        set_reminder, list_reminders, cancel_reminder_task, manage_mood,
    )
    from junjun_skills.builtin.express_skills import send_emoji
    from junjun_skills.builtin.knowledge_skills import search_knowledge, import_knowledge
    from junjun_skills.builtin.action_skills import (
        send_message, send_poke, get_weather, query_chat_history,
    )
    from junjun_skills.builtin.capability_skills import get_capabilities, find_user_id

    register(get_time)
    register(do_not_reply)
    register(recall_memory)
    register(save_memory)
    register(pin_memory)
    register(manage_user_profile)
    register(query_jargon)
    register(learn_jargon)
    register(set_reminder)
    register(list_reminders)
    register(cancel_reminder_task)
    register(manage_mood)
    register(send_emoji)
    register(search_knowledge)
    register(import_knowledge)
    register(send_message)
    register(send_poke)
    register(get_weather)
    register(query_chat_history)
    register(get_capabilities)
    register(find_user_id)
    logger.info(f"内置 skill 已加载: {[t.name for t in get_tools()]}")
