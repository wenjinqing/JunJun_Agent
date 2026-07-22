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
    if admin_only:
        skill = _wrap_admin_gate(skill)
    _registry[skill.name] = skill
    _availability[skill.name] = available_for
    _skill_plugin[skill.name] = plugin
    logger.debug(f"注册 skill: {skill.name} [{plugin}]{' (admin)' if admin_only else ''}")


def _admin_refusal(tool_name: str, args: tuple, kwargs: dict) -> str:
    from junjun_core.security import current_user_id, report_violation
    from junjun_skills.builtin.memory_skills import current_chat_id
    detail = " ".join(str(a) for a in (*args, *kwargs.values()))[:80]
    report_violation(f"管理员工具 {tool_name}", current_user_id.get(), "",
                     current_chat_id.get(), detail)
    return "（权限不足：这个操作只有管理员能做，已通知管理员）"


def _wrap_admin_gate(skill: BaseTool) -> BaseTool:
    """给工具包管理员权限门（运行时按真实发送者 QQ 判定，LLM 不可伪造）。"""
    from junjun_core.security import current_user_id, is_admin
    name = skill.name
    if getattr(skill, "coroutine", None) is not None:
        original = skill.coroutine

        async def gated(*args, _orig=original, **kwargs):
            if not is_admin(current_user_id.get()):
                return _admin_refusal(name, args, kwargs)
            return await _orig(*args, **kwargs)
        skill.coroutine = gated
    elif getattr(skill, "func", None) is not None:
        original_sync = skill.func

        def gated_sync(*args, _orig=original_sync, **kwargs):
            if not is_admin(current_user_id.get()):
                return _admin_refusal(name, args, kwargs)
            return _orig(*args, **kwargs)
        skill.func = gated_sync
    else:
        logger.warning(f"admin 门包装失败（无可包装入口）: {name}")
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


def get_tools(session=None) -> List[BaseTool]:
    """按会话取可用工具集。session=None 返回全量（不含已禁用）。"""
    tools = []
    for name, skill in _registry.items():
        if name in _disabled or not is_plugin_enabled(_skill_plugin.get(name, "builtin")):
            continue
        gate = _availability.get(name)
        if session is None or gate is None or gate(session):
            tools.append(skill)
    return tools


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
        recall_memory, save_memory, manage_user_profile, query_jargon, learn_jargon,
    )
    from junjun_skills.builtin.reminder_skills import (
        set_reminder, list_reminders, cancel_reminder_task, manage_mood,
    )
    from junjun_skills.builtin.express_skills import send_emoji
    from junjun_skills.builtin.knowledge_skills import search_knowledge, import_knowledge
    from junjun_skills.builtin.action_skills import (
        send_message, send_poke, get_weather, query_chat_history,
    )

    register(get_time)
    register(do_not_reply)
    register(recall_memory)
    register(save_memory)
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
    logger.info(f"内置 skill 已加载: {[t.name for t in get_tools()]}")
