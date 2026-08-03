"""君君消息处理器：决策漏斗 + 拟人化回复全流程（阶段 3）。

流程：
  入站 -> 消息入库 -> 短期记忆 -> [会话队列串行]
  L1 规则门(talk_value 时段+动态因子) -> L2 语义门 -> L3 主 Agent
  -> 回复后处理(分条/错别字/引用) -> 逐条延迟发送 -> 回复入库

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


async def _handle(session: ChatSession, meta: InboundMeta) -> None:
    """会话队列内串行执行的核心处理。发送直接走 gateway（分条延迟）。"""
    import uuid
    trace_id = uuid.uuid4().hex[:12]  # 本轮决策 ID：结构化日志 + Langfuse metadata 互查
    cfg = _l1_config(session)
    frequency_control.note_message(session.chat_id)
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
    from junjun_memory.summarizer import get_summarizer
    summarizer = get_summarizer()
    if summarizer.note(session.chat_id, meta.nickname or meta.user_id or "?", meta.text):
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

    # ---- L3 主 Agent（Langfuse span：漏斗决策在后台可见）----
    from junjun_core.observability import lf
    logger.info(f"[{session.chat_id}] 进入 L3 决策 [trace={trace_id}]")
    # system prompt 快照写 span metadata——Langfuse UI 渲染 bug 时 WebUI 日志页可直接查
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
        )
        # span output：回复内容或沉默标记，后台直接可见
        _span.update(output={"reply": text[:500] if text else None, "silenced": text is None})
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

    await _maybe_adjust_frequency(session)


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
            # 只传会话 id 时知识库永远召不回——导入的知识在日常聊天里是死功能
            get_long_term_memory().search(query, top_k=3,
                                          chat_id=(session.chat_id, "knowledge", "self:diary")),
            timeout=1.5,
        )
        if items:
            parts.append("相关记忆：\n" + "\n".join(f"- {it.text}" for it in items))
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
    return "\n".join(parts)


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
