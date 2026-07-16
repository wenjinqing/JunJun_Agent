"""君君消息处理器：把决策漏斗接到网关。

流程：L1 规则门 -> L2 语义门 -> L3 主 Agent -> ReplySet。
由 run_junjun.py 注入 gateway.set_processor(junjun_processor)。
"""

from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.contracts import ReplySet, ReplySegment
from junjun_core.gateway.router import InboundMeta
from junjun_core.gateway.session_manager import ChatSession
from junjun_core.observability import get_logger

from junjun_memory.short_term import ShortTermMemory
from junjun_agent.funnel import (
    L1Config, L1Result, rule_gate, GateDecision, llm_gate,
)

logger = get_logger("processor")


def _l1_config() -> L1Config:
    cfg = get_global_config()
    chat = cfg.raw.get("chat", {})
    return L1Config(
        talk_value=float(chat.get("talk_value", 0.9)),
        mentioned_bot_reply=bool(chat.get("mentioned_bot_reply", True)),
        nickname=cfg.bot.nickname,
        alias_names=tuple(cfg.bot.alias_names or ()),
    )


def _ensure_session_ready(session: ChatSession) -> None:
    """惰性注入 memory 与 agent（每会话独立）。"""
    if session.memory is None:
        max_ctx = int(get_global_config().raw.get("chat", {}).get("max_context_size", 80))
        session.memory = ShortTermMemory(max_size=max_ctx)
    if session.agent is None:
        from junjun_agent.agent import JunJunAgent
        session.agent = JunJunAgent(session)


async def junjun_processor(session: ChatSession, meta: InboundMeta) -> Optional[ReplySet]:
    """网关 processor：决策漏斗全流程。返回 None 表示沉默。"""
    _ensure_session_ready(session)
    cfg = _l1_config()

    # 先入短期记忆（无论是否回复，上下文都要积累）
    session.memory.add_user(
        meta.text, meta.nickname,
        user_id=meta.user_id or "", message_id=meta.message_id, at_bot=meta.at_bot,
    )

    # ---- L1 规则门（0 token）----
    l1 = rule_gate(
        text=meta.text,
        is_group=session.is_group,
        at_bot=meta.at_bot,
        is_self=meta.is_self,
        silenced_until_call=session.silenced_until_call,
        cfg=cfg,
    )
    if l1 is L1Result.DROP:
        logger.debug(f"[{session.chat_id}] L1 拦截")
        return None
    if session.silenced_until_call:
        # 走到这说明被呼唤，解除沉默模式
        session.silenced_until_call = False
        logger.info(f"[{session.chat_id}] 沉默模式解除（被呼唤）")

    from junjun_llm import get_callbacks
    callbacks = get_callbacks()

    # ---- L2 语义门（小模型，@ 旁路时跳过）----
    if l1 is L1Result.TO_GATE:
        decision = await llm_gate(
            session.memory.render(limit=10), cfg.nickname, callbacks=callbacks,
        )
        if decision is GateDecision.NO_REPLY:
            logger.debug(f"[{session.chat_id}] L2 判定不回复")
            return None
        if decision is GateDecision.NO_REPLY_UNTIL_CALL:
            session.silenced_until_call = True
            logger.info(f"[{session.chat_id}] 进入沉默模式（直到被呼唤）")
            return None

    # ---- L3 主 Agent ----
    text = await session.agent.process(session.memory.render(), callbacks=callbacks)
    if not text:
        return None

    session.memory.add_bot(text)
    return ReplySet(
        platform=session.platform,
        target_user_id=meta.user_id if not session.is_group else None,
        target_group_id=session.group_id,
        segments=[ReplySegment(type="text", data=text)],
        should_reply=True,
    )
