"""Skill 注册表：LangChain @tool 的统一注册与按会话过滤。"""

from typing import Callable, Dict, List, Optional

from langchain_core.tools import BaseTool

from junjun_core.observability import get_logger

logger = get_logger("skills.registry")

_registry: Dict[str, BaseTool] = {}
# skill 名 -> 可用性判断（None = 全会话可用）；session 由 gateway 传入
_availability: Dict[str, Optional[Callable]] = {}


def register(skill: BaseTool, available_for: Optional[Callable] = None) -> None:
    """注册 skill。重名直接报错（拒绝静默覆盖）。

    available_for: (session) -> bool，None 表示全会话可用。
    """
    if skill.name in _registry:
        raise ValueError(f"skill 重名: {skill.name}")
    _registry[skill.name] = skill
    _availability[skill.name] = available_for
    logger.debug(f"注册 skill: {skill.name}")


def get_tools(session=None) -> List[BaseTool]:
    """按会话取可用工具集。session=None 返回全量。"""
    tools = []
    for name, skill in _registry.items():
        gate = _availability.get(name)
        if session is None or gate is None or gate(session):
            tools.append(skill)
    return tools


def clear() -> None:
    """仅供测试。"""
    _registry.clear()
    _availability.clear()


def load_builtin() -> None:
    """加载内置 skill（幂等）。"""
    if _registry:
        return
    from junjun_skills.builtin.get_time import get_time
    from junjun_skills.builtin.do_not_reply import do_not_reply
    from junjun_skills.builtin.memory_skills import (
        recall_memory, save_memory, manage_user_profile, query_jargon, learn_jargon,
    )

    register(get_time)
    register(do_not_reply)
    register(recall_memory)
    register(save_memory)
    register(manage_user_profile)
    register(query_jargon)
    register(learn_jargon)
    logger.info(f"内置 skill 已加载: {[t.name for t in get_tools()]}")
