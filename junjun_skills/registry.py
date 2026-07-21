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


def register(skill: BaseTool, available_for: Optional[Callable] = None) -> None:
    """注册 skill。重名直接报错（拒绝静默覆盖）。

    available_for: (session) -> bool，None 表示全会话可用。
    """
    if skill.name in _registry:
        raise ValueError(f"skill 重名: {skill.name}")
    _registry[skill.name] = skill
    _availability[skill.name] = available_for
    logger.debug(f"注册 skill: {skill.name}")


def set_enabled(name: str, enabled: bool) -> bool:
    """启用/禁用 skill（WebUI 插件管理）。skill 不存在返回 False。"""
    if name not in _registry:
        return False
    if enabled:
        _disabled.discard(name)
    else:
        _disabled.add(name)
    logger.info(f"skill {name} 已{'启用' if enabled else '禁用'}")
    return True


def list_skills() -> List[dict]:
    """插件管理用：全部 skill 及启用状态。"""
    return [{"name": n, "description": (s.description or "")[:80], "enabled": n not in _disabled}
            for n, s in _registry.items()]


def get_tools(session=None) -> List[BaseTool]:
    """按会话取可用工具集。session=None 返回全量（不含已禁用）。"""
    tools = []
    for name, skill in _registry.items():
        if name in _disabled:
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
