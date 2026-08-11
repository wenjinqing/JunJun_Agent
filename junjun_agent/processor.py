"""君君消息处理器：决策门 + 拟人化回复全流程。

流程：
  入站 -> 消息入库 -> 短期记忆 -> [会话队列串行]
  决策前段（命令/拦截器/预热，0 token）-> 决策门（私聊直通，群聊仅 @/直呼）
  -> 主 Agent -> 回复后处理(分条/错别字/引用) -> 逐条延迟发送 -> 回复入库

历史：L1 规则门/L2 语义门/talk_value 频率控制在生产路径从未被调用
（死代码空转烧 token），2026-08-04 严厉审查后删除，git 历史可查。

由 run_junjun.py 注入 gateway.set_processor(junjun_processor)。
"""

import asyncio
import re
import time
from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.contracts import ReplySet, ReplySegment
from junjun_core.gateway.router import InboundMeta
from junjun_core.gateway.session_manager import ChatSession
from junjun_core.observability import get_logger

from junjun_memory.short_term import ShortTermMemory
from junjun_agent.funnel import L1Config
from junjun_agent.postprocess import process_response

logger = get_logger("processor")


def _l1_config(session: ChatSession) -> L1Config:
    cfg = get_global_config()
    chat = cfg.raw.get("chat", {})
    return L1Config(
        mentioned_bot_reply=bool(chat.get("mentioned_bot_reply", True)),
        nickname=cfg.bot.nickname,
        alias_names=tuple(cfg.bot.alias_names or ()),
    )


def _ensure_session_ready(session: ChatSession) -> None:
    """惰性注入 memory 与 agent（每会话独立）。"""
    if session.memory is None:
        max_ctx = int(get_global_config().raw.get("chat", {}).get("max_context_size", 80))
        persist_stm = bool(get_global_config().raw.get("memory", {}).get("persist_short_term", False))
        session.memory = ShortTermMemory(
            max_size=max_ctx,
            chat_id=session.chat_id if persist_stm else "",
            persist=persist_stm,
        )
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

    群聊中被 @ **且是追问/指令类**（长度>10 字或含问号），且距离该消息已有他人
    插话时带引用避免歧义；闲聊/短促回应不带。私聊不引用。
    """
    mode = str(get_global_config().raw.get("chat", {}).get("reply_message_quote", "llm"))
    if mode == "never" or not session.is_group:
        return None
    if not meta.at_bot:
        return None
    # 追问/指令类才引用：短促闲聊不引用（防满屏引用气泡）
    is_question = len(meta.text) > 10 or "?" in meta.text or "？" in meta.text
    if not is_question:
        return None
    entries = session.memory.entries
    # 最后一条 user 消息之后若还有别人发言，回复带引用
    for e in reversed(entries[:-1]):
        if e.role == "user" and e.message_id == meta.message_id:
            break
        if e.role == "user" and e.user_id != meta.user_id:
            return meta.message_id or None
    return None


async def _pre_decision(session: ChatSession, meta: InboundMeta) -> None:
    """决策前段（0 token 段）：身份注入/反射器/命令总线/拦截器/复读/预热/批次记录。

    会话队列合并连发消息时，被合并的消息也必须过这一段——否则斜杠命令、
    链接拦截、图片/语音/视频预热会被静默吞掉（严厉审查 P1-7：
    「/sub add xxx」+「你在吗」连发，命令被 drain 丢弃零日志）。
    """
    cfg = _l1_config(session)
    session.last_active_ts = time.time()  # 主动系统空闲判定

    # ---- 调用者身份注入（真实 QQ，工具层/prompt 鉴权锚点）----
    # 权限激活位：管理员本人 +（@bot 或私聊）才激活；
    # 管理员平时=普通群友（走正常漏斗，不直通、不特殊）
    from junjun_core.security import set_caller
    set_caller(meta.user_id, at_bot=meta.at_bot, is_group=session.is_group,
               nickname=meta.nickname)

    # ---- 好感度累计（0 token，异步入库；@/直呼加权）----
    if not meta.is_self and meta.user_id:
        from junjun_express.intimacy import note_interaction
        note_interaction(meta.user_id, addressed=meta.at_bot)

    # ---- 表达反思：管理员回复「删除 N」拦截（0 token）----
    if not meta.is_self:
        try:
            from junjun_express.reflector import expression_reflector
            receipt = expression_reflector.handle_operator_reply(session.chat_id, meta.text)
            if receipt:
                from junjun_core.gateway.router import get_gateway
                await get_gateway().send_reply(ReplySet(
                    platform=session.platform,
                    target_group_id=session.group_id,
                    target_user_id=meta.user_id if not session.is_group else None,
                    segments=[ReplySegment(type="text", data=receipt)], should_reply=True,
                ))
                return
        except Exception:
            pass

    # ---- 命令总线（0 token，旧插件 /cmd 命令的新形态）----
    from junjun_agent.commands import dispatch as dispatch_command
    if await dispatch_command(session, meta):
        return

    # ---- 消息拦截器（0 token，链接自动解析类：B站/抖音/网盘）----
    from junjun_agent.interceptors import dispatch as dispatch_interceptor
    if await dispatch_interceptor(session, meta):
        return

    # ---- 表情包偷图（fire-and-forget，失败静默；只偷表情包，不偷普通图片）----
    if meta.sticker_urls and session.is_group and not meta.is_self:
        try:
            from junjun_express.emoji import emoji_manager
            await emoji_manager.steal(meta.sticker_urls)
        except Exception:
            pass

    # ---- 复读参与（0 token，先于漏斗；跟读不挡正常决策）----
    if session.is_group:
        from junjun_agent.loop.repeat import repeat_detector
        echo = repeat_detector.note(session.chat_id, meta.user_id or "", meta.text,
                                    is_self=meta.is_self)
        if echo == "[STEAL]":
            # 热图偷图：只保存不发送（重复出现的表情包说明是热图）
            if meta.sticker_urls:
                try:
                    from junjun_express.emoji import emoji_manager
                    await emoji_manager.steal(meta.sticker_urls)
                    logger.info(f"[{session.chat_id}] 热图偷图成功")
                except Exception:
                    pass
            return  # 偷图本身就是本条消息的处理，不再进漏斗
        elif echo and echo.startswith("[INTERRUPT:"):
            # 打断复读：发送固定打断语句（随机风格）
            phrase = echo[len("[INTERRUPT:"):-1]  # 去掉前缀和结尾 ]
            from junjun_core.gateway.router import get_gateway
            await get_gateway().send_reply(ReplySet(
                platform=session.platform, target_group_id=session.group_id,
                segments=[ReplySegment(type="text", data=phrase)], should_reply=True,
            ))
            session.memory.add_bot(phrase)
            return  # 打断本身就是本条消息的回应，不再进漏斗
        elif echo:
            from junjun_core.gateway.router import get_gateway
            await get_gateway().send_reply(ReplySet(
                platform=session.platform, target_group_id=session.group_id,
                segments=[ReplySegment(type="text", data=echo)], should_reply=True,
            ))
            session.memory.add_bot(echo)
            return  # 跟读本身就是本条消息的回应，不再进漏斗

    # ---- 中期记忆：批次记录，满批触发摘要 ----
    # bot 自己的消息不喂摘要器——自己的话被蒸馏成「群里发生的事」固化进长期
    # 记忆，是自我污染的第二条补给线（严厉审查 P0-3）
    from junjun_memory.summarizer import get_summarizer
    summarizer = get_summarizer()
    if not meta.is_self and summarizer.note(session.chat_id, meta.nickname or meta.user_id or "?", meta.text):
        await summarizer.summarize(session.chat_id)

    # ---- 事件雷达：群消息里的未来安排自动登记（预过滤 0 token，不阻塞） ----
    if session.is_group and not meta.is_self:
        try:
            from junjun_agent.loop.event_radar import maybe_scan
            maybe_scan(session.chat_id, meta.user_id, meta.nickname, meta.text)
        except Exception:
            pass

    # ---- 表达学习：积累群友消息，满批学习 ----
    if session.is_group and not meta.is_self:
        from junjun_express.expression import expression_learner
        if expression_learner.note(session.chat_id, meta.nickname, meta.text):
            await expression_learner.learn(session.chat_id)

    # ---- 图片预热识图（不管是否 @bot）：发图 -> 再 @君君看 的场景，
    # 等 Agent 被叫到时 VLM 描述已就绪/在途，不再「看不到图」----
    if not meta.is_self and (meta.image_urls or getattr(meta, "sticker_urls", None)):
        try:
            from junjun_memory.vision import prewarm_images
            prewarm_images(session.chat_id, meta.image_urls,
                           getattr(meta, "sticker_urls", None))
        except Exception:
            pass
    # ---- 语音预热转写（同构）：发语音 -> 再 @君君，ASR 已就绪/在途 ----
    if not meta.is_self and getattr(meta, "voice_records", None):
        try:
            from junjun_memory.voice import prewarm_voice
            prewarm_voice(meta.voice_records)
        except Exception:
            pass
    # ---- 视频预热感知（同构，但更重：下载+抽帧+ASR 10-30s，预热几乎是唯一
    # 能及时用完的路径——发视频 -> 过会儿 @君君，描述已就绪）----
    if not meta.is_self and getattr(meta, "video_urls", None):
        try:
            from junjun_memory.video import prewarm_videos
            prewarm_videos(meta.video_urls)
        except Exception:
            pass

    return


async def _handle(session: ChatSession, meta: InboundMeta) -> None:
    """会话队列内串行执行的核心处理。发送直接走 gateway（分条延迟）。"""
    import uuid
    await _pre_decision(session, meta)
    trace_id = uuid.uuid4().hex[:12]  # 本轮决策 ID：结构化日志 + Langfuse metadata 互查
    cfg = _l1_config(session)

    # ---- 决策门（0 token）：私聊直通，群聊仅 @/直呼进思考 ----
    if meta.is_self:
        logger.debug(f"[{session.chat_id}] 自消息，沉默")
        return
    addressed = True  # 私聊默认直通；群聊按 is_addressed 判定
    if session.is_group:
        from junjun_agent.funnel.rule_gate import is_addressed
        addressed = is_addressed(meta.text, cfg, meta.at_bot)
        if not addressed:
            logger.debug(f"[{session.chat_id}] 非 @/直呼，沉默")
            return
    # 私聊：直通（对齐原 Brain 语义：私聊基本都回）
    if session.silenced_until_call:
        session.silenced_until_call = False
        logger.info(f"[{session.chat_id}] 沉默模式解除（被呼唤）")

    from junjun_llm import get_callbacks
    callbacks = get_callbacks()

    # ---- skill 上下文注入（memory skill 执行时读）----
    from junjun_skills.builtin.memory_skills import current_chat_id, current_platform
    current_chat_id.set(session.chat_id)
    current_platform.set(session.platform)

    # ---- 记忆/关系/情绪/表达块（检索失败降级空串，不阻塞回复）----
    memory_block, pending_perception = await _build_memory_block(session, meta)
    relation_block = _build_relation_block(session, meta)

    from junjun_express.mood import mood_manager
    mood_block = mood_manager.build_mood_block(session.chat_id)

    expression_block = ""
    try:
        from junjun_express.expression import build_expression_block
        expression_block = build_expression_block(session.chat_id, meta.text)
    except Exception:
        pass
    if expression_block:
        memory_block = f"{memory_block}\n{expression_block}" if memory_block else expression_block

    # ---- 路由层：复杂任务 -> 任务通道（TaskKernel），其余走对话通道 ----
    # 0-token 严格规则，宁漏勿错；[task_kernel] enable 关闭时 try_submit 直接
    # 返回 None，等于回到现状（灰度开关）。
    text: Optional[str] = None
    from junjun_agent.router import route_to_task
    if route_to_task(meta.text, chat_id=session.chat_id):
        # 路由命中留 Langfuse span：accepted=false 的样本是阶段 4 误路由/规划失败抽检素材。
        # reject_reason 区分三种回退：disabled（灰度开关关）/ planner_none（规划失败）/
        # exception——混在一起可疑清单会被开关噪声淹没（2026-08-06 审查实锤）
        from junjun_core.observability import lf
        with lf.start_span(
            name=f"router.{session.chat_id}",
            input={"latest_text": meta.text},
            metadata={"trace_id": trace_id, "route": "task"},
        ) as _rspan:
            reject_reason = ""
            try:
                from junjun_agent.task_kernel import enabled as _tk_enabled
                if not _tk_enabled():
                    reject_reason = "disabled"
                else:
                    from junjun_agent.task_kernel import kernel
                    text = await kernel.try_submit(
                        meta.text, chat_id=session.chat_id,
                        user_id=meta.user_id or "", callbacks=callbacks)
                    if not text:
                        reject_reason = "planner_none"
            except Exception as e:
                logger.warning(f"[{session.chat_id}] 任务内核接单异常，回退对话通道: {e}")
                text = None
                reject_reason = f"exception:{type(e).__name__}"
            try:
                _rspan.update(metadata={"accepted": bool(text),
                                        "reject_reason": reject_reason})
            except Exception:
                pass
        if text:
            logger.info(f"[{session.chat_id}] 路由->任务通道，已接单 [trace={trace_id}]")

    # ---- L3 主 Agent（Langfuse span：漏斗决策在后台可见）----
    if text is None:
        from junjun_core.observability import lf
        logger.info(f"[{session.chat_id}] 进入 L3 决策 [trace={trace_id}]")
        # system prompt 快照写 span metadata——Langfuse UI 渲染 bug 时 WebUI 日志页可直接查
        # Phase 2：ContextBudget 启用时，prompt 在 agent 内按预算重组，processor 不再预构建；
        # span 仍留一个 core 快照（不占用 agent 预算决策）用于调试。
        cfg = get_global_config().raw
        budget_enabled = bool(cfg.get("context_budget", {}).get("enable", False))
        _prompt_snapshot = ""
        if budget_enabled:
            from junjun_agent.persona import build_admin_block, build_prompt_blocks
            core, _ = build_prompt_blocks(
                is_group=session.is_group, latest_text=meta.text)
            _prompt_snapshot = core + "\n\n" + build_admin_block()
        else:
            from junjun_agent.persona import build_system_prompt
            _prompt_snapshot = build_system_prompt(
                is_group=session.is_group, latest_text=meta.text,
                mood_block=mood_block, memory_block=memory_block, relation_block=relation_block,
            )
        with lf.start_span(
            name=f"agent.{session.chat_id}",
            input={"latest_text": meta.text, "context_preview": session.memory.render(limit=5, for_security=True)[:500]},
            metadata={
                "trace_id": trace_id, "addressed": addressed, "at_bot": meta.at_bot,
                "system_prompt": _prompt_snapshot[:2000],
                "context_budget_enabled": budget_enabled,
            },
        ) as _span:
            text = await session.agent.process(
                # 群聊 30 条上下文（提高长度）+ 标记最后一条 + 发言者画像注入
                # for_security=True：保留（管理员）标记供安全验证锚点
                session.memory.render(limit=30, mark_latest=True, for_security=True),
                callbacks=callbacks, latest_text=meta.text,
                addressed=True,  # 只有 @/直呼才走到这里
                memory_block=memory_block, relation_block=relation_block,
                mood_block=mood_block, trace_id=trace_id,
                system_prompt=None if budget_enabled else _prompt_snapshot,
            )
            # ---- Phase 3：发送前 HonestyGuard 代码层诚实校验（span 内做）----
            # 曾经 span 先记录原文、校验在 span 外替换——trace 里是原文，用户
            # 收到的是替换稿，「实发文本 trace 里找不到」（2026-08-06 实锤）。
            # 现在：拦截的原文/理由/实发稿全部进 span output。
            hg_issues: list = []
            hg_original: Optional[str] = None
            if text:
                try:
                    from junjun_agent.honesty_guard import (
                        enabled as hg_enabled, verify as hg_verify)
                    if hg_enabled():
                        ok, fixed, hg_issues = hg_verify(session, text)
                        if not ok:
                            hg_original = text
                            text = fixed
                            logger.warning(f"[{session.chat_id}] HonestyGuard 拦截: "
                                           f"{hg_issues} [trace={trace_id}]")
                except Exception:
                    pass

            # span output：回复内容或沉默标记，后台直接可见
            _out = {"reply": text[:500] if text else None, "silenced": text is None}
            if hg_issues:
                _out["honesty_guard"] = {
                    "intercepted": True, "issues": hg_issues,
                    "original": (hg_original or "")[:500]}
            _span.update(output=_out)
            if not text:
                logger.info(f"[{session.chat_id}] L3 沉默 [trace={trace_id}]")

    # 情绪重评（跟随 L3，冷却内跳过；不阻塞发送——先发再评）
    if not text:
        return

    # ---- 感知后续：决策时「还在看」的图/语音/视频，看完后主动补一句 ----
    # （治「我看不到图片」：Agent 已接话 + 有在途感知 -> 完成后观后感推上门）
    if pending_perception:
        try:
            from junjun_agent.loop import perception_followup
            perception_followup.schedule(session, pending_perception)
        except Exception:
            pass

    session.memory.add_bot(text)
    _store_outbound(session, text)
    # 自我反思计数（到阈值后台自评并私聊管理员，失败静默）
    try:
        from junjun_agent.loop.reflection import reflection_loop
        reflection_loop.note_reply()
    except Exception:
        pass

    # ---- 回复后处理：分条 + 错别字 + 引用 ----
    quote_id = _quote_message_id(session, meta)  # 提前定义（image/forward 分支也要用）
    # 特殊标记：[IMAGE:url] -> 提取为 image 段单独发送（ai_draw 等工具产出）
    import re as _re
    _img_match = _re.search(r"\[IMAGE:(https?://[^\]]+)\]", text)
    if _img_match:
        img_url = _img_match.group(1)
        # 文本部分去掉标记，图片单独发
        clean_text = text.replace(_img_match.group(0), "").strip()
        from junjun_core.gateway.router import get_gateway
        gateway = get_gateway()
        if clean_text:
            outbound = process_response(clean_text, incoming=meta.text)
            for i, msg in enumerate(outbound):
                if msg.delay > 0:
                    await asyncio.sleep(msg.delay)
                await gateway.send_reply(ReplySet(
                    platform=session.platform,
                    target_user_id=meta.user_id if not session.is_group else None,
                    target_group_id=session.group_id,
                    segments=[ReplySegment(type="text", data=msg.text)],
                    should_reply=True,
                    reply_to_message_id=quote_id if i == 0 else None,
                ))
        await gateway.send_reply(ReplySet(
            platform=session.platform,
            target_user_id=meta.user_id if not session.is_group else None,
            target_group_id=session.group_id,
            segments=[ReplySegment(type="image", data=img_url)],
            should_reply=True,
        ))
        logger.info(f"[{session.chat_id}] 图片已发送: {img_url[:60]}")
        return

    # MCP 长结果特殊处理：被包装为 forward JSON 的不走分条，直接合并转发发出
    if text.strip().startswith('{"type": "forward"'):
        try:
            import json
            fwd = json.loads(text)
            from junjun_core.gateway.router import get_gateway
            gateway = get_gateway()
            await gateway.send_reply(ReplySet(
                platform=session.platform,
                target_user_id=meta.user_id if not session.is_group else None,
                target_group_id=session.group_id,
                segments=[ReplySegment(type="text", data=fwd.get("text", "")),
                          ReplySegment(type="forward", data=json.dumps(fwd.get("nodes", []), ensure_ascii=False))],
                should_reply=True,
            ))
            return
        except Exception as e:
            logger.warning(f"forward 消息解析失败，降级分条: {e}")

    outbound = process_response(text, incoming=meta.text)
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

    # 情绪重评（发送后进行，不阻塞回复；冷却内跳过）
    if mood_manager.should_evaluate(session.chat_id):
        await mood_manager.evaluate(
            session.chat_id, session.memory.render(limit=12), callbacks=callbacks,
        )


# 记忆召回失败告警节流（5 分钟一次）：API 抖动期召回静默全失不能毫无痕迹，
# 但每条消息都 warning 会刷屏
_last_recall_warn = 0.0


def _warn_recall_throttled(e: BaseException) -> None:
    global _last_recall_warn
    import time as _t
    now = _t.monotonic()
    if now - _last_recall_warn >= 300:
        _last_recall_warn = now
        logger.warning(f"记忆召回失败（降级无记忆注入，5 分钟内不再重复告警）: "
                       f"{type(e).__name__}: {e}")


# 「你忽然想起」注入限流（P6-1）：每会话每小时上限，防每轮都注入回忆块
# 稀释人设 + 省 embedding 调用。命中注入才占额度（检索失败/空结果不占）。
_RECALL_LOG: dict = {}  # chat_id -> deque(注入时间戳，1h 滑窗)


def _recall_capped(chat_id: str, max_per_hour: int) -> bool:
    """本小时注入额度是否已用完（只读检查，同时清过期）。"""
    from collections import deque
    now = time.time()
    dq = _RECALL_LOG.setdefault(chat_id, deque())
    while dq and now - dq[0] > 3600:
        dq.popleft()
    return len(dq) >= max_per_hour


def _recall_consume(chat_id: str) -> None:
    _RECALL_LOG[chat_id].append(time.time())


async def _build_memory_block(session: ChatSession, meta: InboundMeta) -> tuple:
    """被动记忆注入 + 感知在途清单。

    返回 (记忆块文本, 在途感知条目 [{"kind","task"}])。
    在途感知（3s 决策窗内没看完的图/语音/视频）必须让 Agent 知道「还在看」
    而不是「看不到」——2026-08-02 生产反馈：占位符与「没有图」无法区分，
    Agent 只能说「看不到图片」，体验像瞎子。条目交给 perception_followup
    在完成后主动补观后感。
    """
    parts = []
    pending: list = []
    pending_kinds: list = []
    if meta.image_urls:
        try:
            from junjun_memory.vision import describe_images_full, render_image_block
            descs, pend = await describe_images_full(meta.image_urls)
            block = render_image_block(descs)
            if block:
                parts.append(block)
            if pend:
                pending += [{"kind": "image", "task": t} for t in pend]
                pending_kinds.append(f"{len(pend)} 张图片")
        except Exception:
            pass
    if getattr(meta, "sticker_urls", None):
        try:
            from junjun_memory.vision import describe_stickers, render_sticker_block
            block = render_sticker_block(await describe_stickers(meta.sticker_urls))
            if block:
                parts.append(block)
        except Exception:
            pass
    # 语音转写：record 段 -> ASR（预热已就绪则秒回；在途有界等待）
    if getattr(meta, "voice_records", None):
        try:
            from junjun_memory.voice import transcribe_voices_full, render_voice_block
            texts, pend = await transcribe_voices_full(meta.voice_records)
            block = render_voice_block(texts)
            if block:
                parts.append(block)
            if pend:
                pending += [{"kind": "voice", "task": t} for t in pend]
                pending_kinds.append(f"{len(pend)} 段语音")
        except Exception:
            pass
    # 视频感知：抽帧 VLM + 抽音 ASR（预热就绪才注入；首轮多半还在看）
    if getattr(meta, "video_urls", None):
        try:
            from junjun_memory.video import describe_videos_full, render_video_block
            descs, pend = await describe_videos_full(meta.video_urls)
            block = render_video_block(descs)
            if block:
                parts.append(block)
            if pend:
                pending += [{"kind": "video", "task": t} for t in pend]
                pending_kinds.append(f"{len(pend)} 个视频")
        except Exception:
            pass
    if pending_kinds:
        parts.append("注意：对方刚发的" + "、".join(pending_kinds) +
                     "你还在看（后台解析中，还没看完）——先自然地说你在看，"
                     "看完你会主动补一句；绝不要说「看不到/没收到」。")
    # 近期图片补充：图是前几条消息发的（当时没 @bot），被 @ 时把最近 10 分钟
    # 群里的图片描述也注入（预热任务已就续则秒回；在途则 await 同一任务）
    try:
        from junjun_memory.vision import describe_images, recent_image_urls
        current = set(meta.image_urls or []) | set(getattr(meta, "sticker_urls", None) or [])
        older = [(k, u) for k, u in recent_image_urls(session.chat_id) if u not in current]
        if older:
            img_urls = [u for k, u in older if k == "image"]
            if img_urls:
                descs = await describe_images(img_urls)
                lines = [f"- {d}" for d in descs.values() if d and d != "[图片]"]
                if lines:
                    parts.append("群里最近发的图片：\n" + "\n".join(lines))
    except Exception:
        pass
    # 链接内容感知：消息含网页链接时抓正文摘要注入（4s 超时，失败静默）
    try:
        from junjun_memory.link_preview import fetch_link_preview
        preview = await fetch_link_preview(meta.text or "")
        if preview:
            parts.append(f"对方分享的链接内容：{preview}")
    except Exception:
        pass
    # B站视频理解：群里最近分享的视频字幕摘要（bilibili 插件后台「看懂」）
    try:
        from junjun_skills.plugins.bilibili import content as _bili_content
        vb = _bili_content.render_recent_block(session.chat_id)
        if vb:
            parts.append(vb)
    except Exception:
        pass
    # 抖音视频理解：群里最近分享的抖音摘要（douyin 插件后台「看懂」）
    try:
        from junjun_skills.plugins.douyin import content as _dy_content
        db = _dy_content.render_recent_block(session.chat_id)
        if db:
            parts.append(db)
    except Exception:
        pass
    # 订阅感知：本会话生效中的订阅（重启后 Agent 依然知道自己在盯梢）
    try:
        from junjun_skills.plugins.subscription.tools import subscriptions_block
        sb = subscriptions_block(session.chat_id)
        if sb:
            parts.append(sb)
    except Exception:
        pass
    # 后台任务近况（2026-08-04「图呢」事件）：成品任务（画图/语音/视频）的
    # 在途与结局注入——Agent 必须记得自己答应的事办成了没有，
    # 失败要主动提补救，不能「说了在画了然后永远没下文」
    try:
        from junjun_agent.tasks import task_manager
        tb = task_manager.task_status_block(session.chat_id)
        if not tb:
            # 否定证据（同日「一直说还在画」幻觉）：用户在问进度但没有
            # 任何在途/记录——不注入的话模型只能顺着历史里的旧话续编
            import re as _re2
            if _re2.search(r"图呢|画好|还没|画完|画得|做好了|进度|还没弄", meta.text or ""):
                tb = task_manager.negative_status_block(session.chat_id)
        if tb:
            parts.append(tb)
    except Exception:
        pass
    # 钉住记忆（P6-2）：对方明确说「记住」钉下的事，每轮优先注入，
    # 不占语义召回额度——用户显式意志 > 检索运气
    try:
        from junjun_memory.long_term import get_long_term_memory as _gltm
        _pins = _gltm().pinned(session.chat_id)
        if _pins:
            parts.append("对方明确要求你记住的事（钉住的，务必当回事）：\n"
                         + "\n".join(f"- {it.text}" for it in _pins))
    except Exception:
        pass
    # 语义召回（P6-1）：faiss 向量检索（top-3 + 0.3 阈值）注入「你忽然想起」块，
    # 每会话每小时限流（默认 5 次，[memory] recall_max_per_hour），超限整段跳过
    try:
        from junjun_core.config import get_global_config as _ggc
        _max_recall = int(_ggc().raw.get("memory", {}).get("recall_max_per_hour", 5))
    except Exception:
        _max_recall = 5
    if _max_recall <= 0 or not _recall_capped(session.chat_id, _max_recall):
        try:
            import asyncio as _aio
            import re as _re
            from junjun_memory.long_term import get_long_term_memory
            # 检索查询清洗：剥掉 [回复...]/[图片] 占位与 @昵称 前缀，避免噪声稀释相似度
            query = _re.sub(r"\[[^\]]{0,220}\]", " ", meta.text or "")
            query = _re.sub(r"@\S+\s*", " ", query).strip() or (meta.text or "")
            items = await _aio.wait_for(
                # chat_id 多值过滤：本会话记忆 + 知识库条目（"knowledge"）
                # + 自我日记（"self:diary"，第一人称自我叙事）。
                # 只传会话 id 时知识库永远召不回——导入的知识在日常聊天里是死功能。
                # 含私聊素材的日记（"self:diary:private"）只在私聊会话召回——
                # 私聊内容经日记转述进群聊是广播级泄露（严厉审查 P0-4）。
                # 场景不明时按群聊处理（保守收紧，宁少召回不泄露）
                get_long_term_memory().search(
                    query, top_k=3,
                    chat_id=((session.chat_id, "knowledge", "self:diary", "self:diary:private")
                             if not getattr(session, "is_group", True)
                             else (session.chat_id, "knowledge", "self:diary"))),
                timeout=1.5,
            )
            if items:
                _recall_consume(session.chat_id)
                parts.append(
                    "你忽然想起这些相关的事（可能不完全相关——和当前话题搭就顺着自然提一句，"
                    "不搭就当没想起，别逐条转述）：\n"
                    + "\n".join(f"- {it.text}" for it in items))
        except Exception as e:
            _warn_recall_throttled(e)
    try:
        from junjun_express.jargon import build_jargon_block
        jb = build_jargon_block(meta.text, session.chat_id)
        if jb:
            parts.append(jb)
    except Exception:
        pass
    return "\n".join(parts), pending


def _build_relation_block(session: ChatSession, meta: InboundMeta) -> str:
    """发言者画像 + 好感度档位注入（P2-19 关系驱动行为）。失败降级空串。"""
    if not meta.user_id:
        return ""
    parts = []
    # 好感度档位 + 行为指导：语气亲疏由真实互动数据驱动，不再千人一面
    try:
        from junjun_core.config import get_global_config
        if get_global_config().raw.get("relationship", {}).get("enable", True):
            from junjun_express.intimacy import behavior_hint, get_intimacy
            score, count, level = get_intimacy(meta.user_id)
            parts.append(
                f"和 {meta.nickname or '对方'} 的关系：{level}"
                f"（好感度 {score:.0f}/100，互动过 {count} 次）——{behavior_hint(level)}")
    except Exception:
        pass
    try:
        from junjun_memory.user_profile import get_profile_store
        block = get_profile_store().build_relation_block(
            session.platform, meta.user_id, meta.nickname,
        )
        if block:
            parts.append(block)
    except Exception:
        pass
    # 跨场景用户档案（P6-4）：知道，但分得清场合——过滤逻辑在 build_scene_block 强制
    try:
        from junjun_memory.scene_profile import build_scene_block
        sb = build_scene_block(session.platform, meta.user_id,
                               session.chat_id, session.is_group)
        if sb:
            parts.append(sb)
    except Exception:
        pass
    return "\n".join(parts)


def _queue_addressed(session: ChatSession, meta: InboundMeta) -> bool:
    """会话队列决策目标判定（Q1）：私聊一律 addressed（取最新即最新 addressed，
    行为不变）；群聊复用决策门同一套规则（@/昵称/别名直呼），不在队列层重写。"""
    if not session.is_group:
        return True
    from junjun_agent.funnel.rule_gate import is_addressed
    return is_addressed(meta.text, _l1_config(session), meta.at_bot)


async def junjun_processor(session: ChatSession, meta: InboundMeta) -> Optional[ReplySet]:
    """网关 processor 入口：记忆/入库即时完成，决策投递会话队列串行处理。

    返回 None（发送由 _handle 内部走 gateway 分条执行）。
    """
    _ensure_session_ready(session)
    # 记忆与入库不排队：堆积消息也要进上下文
    if meta.is_self:
        # bot 自己的消息（NapCat 回传）：以 bot 身份进短期记忆——以 user 身份
        # 写入等于亲手把「自己说的话」伪装成「别人说的话」喂回模型，是自我
        # 模仿/复读的输入侧补给线（严厉审查 P0-3）
        session.memory.add_bot(meta.text)
    else:
        session.memory.add_user(
            meta.text, meta.nickname,
            user_id=meta.user_id or "", message_id=meta.message_id, at_bot=meta.at_bot,
        )
    _store_inbound(session, meta)
    # 复杂任务人审（LangGraph 引擎）：管理员的「发/算了」最优先消费，
    # 不进决策队列（在记忆入库之后——审批回复也是真实消息，如实记录）
    try:
        from junjun_agent.task_kernel.graph import approval_hook
        if await approval_hook(session, meta):
            return None
    except Exception:
        pass
    # 热点日报人审：同一套「发/算了」，只在自己的待审批非空时接单
    try:
        from junjun_skills.plugins.daily_report.graph import approval_hook as dr_hook
        if await dr_hook(session, meta):
            return None
    except Exception:
        pass
    # 群周报社发人审：同上，pending 独立
    try:
        from junjun_skills.plugins.weekly_report.tools import approval_hook as wr_hook
        if await wr_hook(session, meta):
            return None
    except Exception:
        pass
    # 意向系统事件钩子（P7）：emo 规则预筛 -> 关心意向（0 token，
    # [intention] enable=false 时零开销直接返回）
    try:
        from junjun_agent.loop.intention import spawn_care_if_needed
        spawn_care_if_needed(session.chat_id, meta.user_id or "",
                             meta.nickname or "", meta.text or "")
    except Exception:
        pass
    # 「别烦我」（P7-4）：只有对 bot 说的才静音本会话 24h——群里两人互怼不能误伤
    try:
        from junjun_agent.loop.intention import detect_mute_request, mute_chat
        from junjun_core.config import get_global_config as _ggc2
        _nick = _ggc2().bot.nickname
        _addr = meta.at_bot or (_nick and _nick in (meta.text or ""))
        if _addr and detect_mute_request(meta.text):
            mute_chat(session.chat_id, hours=24)
            logger.info(f"[{session.chat_id}] 收到「别烦我」，主动消息静音 24h")
    except Exception:
        pass

    from junjun_agent.funnel.session_queue import session_queues
    session_queues.dispatch(session, meta, _handle, pre_handler=_pre_decision,
                            addressed_fn=_queue_addressed)
    return None
