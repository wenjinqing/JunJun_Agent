"""主动出站统一入口（严厉审查 P0-5/S2）。

系统曾有两条出站路径：主管线（processor：分条/清洗/记忆回填/落库）与
后台直发（tasks/async_jobs/订阅推送各写各的 gateway.send_reply，护栏全漏）。
每加一个出站护栏就得记得在直发路径重做一遍，漏一个就是一次线上事故
（「图呢」「一直说还在画」两次幻觉都源于这条裂缝）。

本模块是唯一指定直发口：路由解析 + 文本清洗 + 发送 + 记忆回填 + 出站落库。
所有「非回复当前消息」的主动发送（后台任务成品/任务汇报/订阅推送/定时推送）
都必须走这里。拦截器/命令路径由 CommandContext.send 自带回填（同语义）。
"""

from typing import List, Optional, Tuple

from junjun_core.contracts import ReplySegment, ReplySet
from junjun_core.observability import get_logger

logger = get_logger("agent.outbound")


def parse_route(chat_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    """chat_id（qq:ID:group|private）-> (platform, target_user_id, target_group_id)。"""
    parts = (chat_id or "").split(":")
    if len(parts) < 3:
        return parts[0] if parts and parts[0] else "qq", None, None
    platform, target, kind = parts[0], parts[1], parts[2]
    if kind == "group":
        return platform, None, target
    return platform, target, None


def _remember(chat_id: str, text: str) -> None:
    """出站回填：短期记忆（add_bot）+ 落库（_store_outbound）。

    直发消息不经过 inbound 管线，不手动记的话模型会忘了自己说过
    「画好啦/画砸了」，下轮被问「图呢」只能装傻（2026-08-04 实战）。
    """
    if not text:
        return
    try:
        from junjun_core.gateway.session_manager import get_session_manager
        s = get_session_manager().all_sessions().get(chat_id)
        if s is not None:
            if getattr(s, "memory", None) is not None:
                s.memory.add_bot(text)
            from junjun_agent.processor import _store_outbound
            _store_outbound(s, text)
    except Exception:
        pass


async def send_proactive(chat_id: str, segments: List[ReplySegment], *,
                         source: str = "proactive", remember: bool = True) -> bool:
    """主动推送到会话。返回是否送达。

    - 文本段过主管线同款清洗（clean_markdown：去 markdown/星号表演）
    - 发送成功且 remember=True 时回填记忆 + 落库（bot 记得自己说过什么）
    - 任何异常静默返回 False（直发绝不能炸主流程）
    """
    cleaned: List[ReplySegment] = []
    for s in segments:
        if s.type == "text" and isinstance(s.data, str):
            try:
                from junjun_agent.postprocess.cleaner import clean_markdown
                s = ReplySegment(type="text", data=clean_markdown(s.data))
            except Exception:
                pass
        cleaned.append(s)
    try:
        from junjun_core.gateway.router import get_gateway
        platform, user_id, group_id = parse_route(chat_id)
        await get_gateway().send_reply(ReplySet(
            platform=platform,
            target_user_id=user_id,
            target_group_id=group_id,
            segments=cleaned,
            should_reply=True,
        ))
    except Exception as e:
        logger.warning(f"[{chat_id}] 主动推送失败({source}): {type(e).__name__}: {e}")
        return False
    if remember:
        text = "\n".join(s.data for s in cleaned
                         if s.type == "text" and isinstance(s.data, str)).strip()
        _remember(chat_id, text)
    return True
