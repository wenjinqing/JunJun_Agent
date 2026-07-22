"""消息拦截器总线：ON_MESSAGE 型自动检测（插件迁移基础设施）。

旧插件的「消息里出现 B 站/抖音/网盘链接就自动解析」在新架构
不该占用 LLM 决策——正则命中后在漏斗前直接处理。

注册：
    from junjun_agent.interceptors import register_interceptor

    @register_interceptor(r"v\\.douyin\\.com/\\w+", plugin="douyin",
                          group_at_only=True)
    async def douyin_hit(ctx) -> bool:
        ...  # 处理（可直接 reply，或后台任务下载）
        return True   # True=消费掉，不再进漏斗；False=继续正常决策

InterceptorContext 与 CommandContext 相同（session/meta/args/reply/send），
args 为正则命中的整个匹配串。

processor 在命令总线之后、L1 之前调用 dispatch()。
"""

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Pattern

from junjun_core.observability import get_logger

logger = get_logger("interceptors")

Handler = Callable[[object], Awaitable[bool]]


@dataclass
class Interceptor:
    name: str
    pattern: Pattern
    handler: Handler
    plugin: str = "builtin"
    group_at_only: bool = False      # 群聊里只在 @bot 时触发
    admin_only: bool = False


_interceptors: list = []


def register_interceptor(pattern: str, *, name: str = "", plugin: str = "builtin",
                         group_at_only: bool = False, admin_only: bool = False,
                         flags: int = re.I):
    """装饰器：注册消息拦截器（按注册顺序匹配，先中先得）。"""
    def deco(fn: Handler) -> Handler:
        _interceptors.append(Interceptor(
            name=name or fn.__name__, pattern=re.compile(pattern, flags),
            handler=fn, plugin=plugin, group_at_only=group_at_only,
            admin_only=admin_only,
        ))
        logger.debug(f"注册拦截器: {name or fn.__name__} [{plugin}] /{pattern}/")
        return fn
    return deco


def list_interceptors() -> list:
    return [{"name": i.name, "plugin": i.plugin, "pattern": i.pattern.pattern}
            for i in _interceptors]


async def dispatch(session, meta) -> bool:
    """依次尝试拦截器。True=消息已被消费（不进漏斗）。"""
    text = (meta.text or "").strip()
    if not text:
        return False
    for it in _interceptors:
        m = it.pattern.search(text)
        if not m:
            continue
        if it.group_at_only and session.is_group and not meta.at_bot:
            continue
        from junjun_skills import registry
        if not registry.is_plugin_enabled(it.plugin):
            continue
        if it.admin_only:
            from junjun_core.security import is_admin_privileged, report_violation
            if not is_admin_privileged():
                report_violation(f"管理员拦截器 {it.name}", meta.user_id or "",
                                 meta.nickname, session.chat_id, text[:60])
                return True

        from junjun_agent.commands import CommandContext
        ctx = CommandContext(session=session, meta=meta, args=m.group(0))
        try:
            consumed = await it.handler(ctx)
        except Exception as e:
            logger.error(f"拦截器 {it.name} 执行异常: {type(e).__name__}: {e}")
            consumed = False
        if consumed:
            logger.info(f"拦截器已处理 [{it.name}] [{session.chat_id}] {meta.nickname}")
            return True
    return False


def clear_interceptors() -> None:
    """仅供测试。"""
    _interceptors.clear()
