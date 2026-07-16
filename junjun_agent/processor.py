"""君君消息处理器：决策漏斗 + 拟人化回复全流程（阶段 3）。

流程：
  入站 -> 消息入库 -> 短期记忆 -> [会话队列串行]
  L1 规则门(talk_value 时段+动态因子) -> L2 语义门 -> L3 主 Agent
  -> 回复后处理(分条/错别字/引用) -> 逐条延迟发送 -> 回复入库

由 run_junjun.py 注入 gateway.set_processor(junjun_processor)。
"""

import asyncio
import time
from typing import List, Optional

from junjun_core.config import get_global_config
from junjun_core.contracts import ReplySet, ReplySegment
from junjun_core.gateway.router import InboundMeta
from junjun_core.gateway.session_manager import ChatSession
from junjun_core.observability import get_logger

from junjun_memory.short_term import ShortTermMemory
from junjun_agent.funnel import (
    L1Config, L1Result, rule_gate, GateDecision, llm_gate,
)
from junjun_agent.funnel.frequency import frequency_control
from junjun_agent.postprocess import process_response

logger = get_logger("processor")


def _l1_config(session: ChatSession) -> L1Config:
    cfg = get_global_config()
    chat = cfg.raw.get("chat", {})
    return L1Config(
        # talk_value = 时段规则解析 * LLM 动态调节因子
        talk_value=frequency_control.effective_talk_value(session.chat_id),
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


def _store_inbound(session: ChatSession, meta: InboundMeta) -> None:
    """入站消息落库（fire-and-forget）。"""
    try:
        from junjun_core.database import Messages, db_writer
        db_writer.submit(
            Messages.create,
            message_id=meta.message_id, chat_id=session.chat_id, time=time.time(),
            user_id=meta.user_id or "", user_nickname=meta.nickname,
            group_id=session.group_id or "", processed_plain_text=meta.text,
            is_bot=False, is_at=meta.at_bot, is_mentioned=meta.at_bot,
        )
    except Exception as e:
        logger.debug(f"入站消息落库失败（忽略）: {e}")


def _store_outbound(session: ChatSession, text: str) -> None:
    try:
        from junjun_core.database import Messages, db_writer
        db_writer.submit(
            Messages.create,
            message_id="", chat_id=session.chat_id, time=time.time(),
            user_id="", user_nickname="", group_id=session.group_id or "",
            processed_plain_text=text, is_bot=True,
        )
    except Exception as e:
        logger.debug(f"回复落库失败（忽略）: {e}")


def _quote_message_id(session: ChatSession, meta: InboundMeta) -> Optional[str]:
    """引用回复决策（reply_message_quote 简化实现）：

    群聊中被 @ 且距离该消息已有他人插话时带引用，避免歧义；私聊不引用。
    """
    mode = str(get_global_config().raw.get("chat", {}).get("reply_message_quote", "llm"))
    if mode == "never" or not session.is_group:
        return None
    if not meta.at_bot:
        return None
    entries = session.memory.entries
    # 最后一条 user 消息之后若还有别人发言，回复带引用
    for e in reversed(entries[:-1]):
        if e.role == "user" and e.message_id == meta.message_id:
            break
        if e.role == "user" and e.user_id != meta.user_id:
            return meta.message_id or None
    return None


async def _handle(session: ChatSession, meta: InboundMeta) -> None:
    """会话队列内串行执行的核心处理。发送直接走 gateway（分条延迟）。"""
    cfg = _l1_config(session)
    frequency_control.note_message(session.chat_id)

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
        logger.debug(f"[{session.chat_id}] L1 拦截 (talk_value={cfg.talk_value:.2f})")
        await _maybe_adjust_frequency(session)
        return
    if session.silenced_until_call:
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
            await _maybe_adjust_frequency(session)
            return
        if decision is GateDecision.NO_REPLY_UNTIL_CALL:
            session.silenced_until_call = True
            logger.info(f"[{session.chat_id}] 进入沉默模式（直到被呼唤）")
            return

    # ---- L3 主 Agent ----
    text = await session.agent.process(
        session.memory.render(), callbacks=callbacks, latest_text=meta.text,
        addressed=(l1 is L1Result.TO_AGENT),
    )
    if not text:
        return

    session.memory.add_bot(text)
    _store_outbound(session, text)

    # ---- 回复后处理：分条 + 错别字 + 引用 ----
    outbound = process_response(text)
    if not outbound:
        return
    quote_id = _quote_message_id(session, meta)

    from junjun_core.gateway.router import get_gateway
    gateway = get_gateway()
    for i, msg in enumerate(outbound):
        if msg.delay > 0:
            await asyncio.sleep(msg.delay)
        await gateway.send_reply(ReplySet(
            platform=session.platform,
            target_user_id=meta.user_id if not session.is_group else None,
            target_group_id=session.group_id,
            segments=[ReplySegment(type="text", data=msg.text)],
            should_reply=True,
            reply_to_message_id=quote_id if i == 0 else None,  # 只首条带引用
        ))

    await _maybe_adjust_frequency(session)


async def _maybe_adjust_frequency(session: ChatSession) -> None:
    """满足冷却与消息数条件时触发 LLM 频率评估。"""
    if not session.is_group:
        return
    if frequency_control.should_evaluate(session.chat_id):
        await frequency_control.evaluate_with_llm(
            session.chat_id, session.memory.render(limit=20),
        )


async def junjun_processor(session: ChatSession, meta: InboundMeta) -> Optional[ReplySet]:
    """网关 processor 入口：记忆/入库即时完成，决策投递会话队列串行处理。

    返回 None（发送由 _handle 内部走 gateway 分条执行）。
    """
    _ensure_session_ready(session)
    # 记忆与入库不排队：堆积消息也要进上下文
    session.memory.add_user(
        meta.text, meta.nickname,
        user_id=meta.user_id or "", message_id=meta.message_id, at_bot=meta.at_bot,
    )
    _store_inbound(session, meta)

    from junjun_agent.funnel.session_queue import session_queues
    session_queues.dispatch(session, meta, _handle)
    return None
