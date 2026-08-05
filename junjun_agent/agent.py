"""Agent 核心：LangChain 1.x create_agent（LangGraph runtime）。

每会话独立 agent 实例 + 独立消息历史，防跨会话串味。
system prompt 每轮动态构建（时间/keyword_reaction/情绪/记忆块都是活的），
通过 SystemMessage 前置注入而非 create_agent(system_prompt=...) 冻结。
决策语义：reply -> 文本输出；no_reply -> do_not_reply 工具置沉默。
"""

import time
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger
from junjun_skills.registry import get_tools, load_builtin, intent_groups
from junjun_skills.builtin.do_not_reply import SILENCE_TOOL_NAME
from junjun_agent.loop.plan_tracker import (
    PlanMiddleware, detect_complexity, make_plan, set_plan, reset_plan,
)
from junjun_agent.persona import build_prompt_blocks, build_system_prompt
from junjun_agent.context_budget import BudgetBlock, ContextBudget

logger = get_logger("agent")


def _called_silence_tool(messages: list) -> bool:
    """扫描本轮消息里是否调过 do_not_reply（工具在独立 context 执行，
    contextvar 状态写不回，用 tool_call 记录本身作为沉默信号）。"""
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                if tc.get("name") == SILENCE_TOOL_NAME:
                    return True
    return False


# 意图自检规则：强意图词 -> 必须真正调用的工具。元数据与注册表 INTENT 层
# 挂载共用单一数据源（顺序敏感，先长后短，「取消订阅」含「订阅」必须先匹配）。
_INTENT_RULES = [(kws, primary) for kws, _group, primary in intent_groups() if primary]

_NUDGE_PROMPT = (
    "（系统追问）对方的请求包含明确的「{intent}」意图，必须调用 {tool} 工具"
    "才能真正生效，你刚才没有调用它。请现在调用该工具完成请求；"
    "如果工具调用失败或确实办不到，直接如实告诉对方办不到，不要口头承诺。"
    "注意：你上一轮的回复还没有发送出去，对方目前什么都没看到——"
    "不要说「如上所述」「上面那份」之类的话。"
)


def _background_to_messages(background: str) -> list:
    """背景文本 -> 结构化多轮消息：bot 的历史发言必须是 AIMessage。

    曾把全部背景拼成一个 HumanMessage，bot 自己的话以「你(历史):」文本
    前缀混在 user role 里——模型对它只有「语气模仿」一种处理方式，
    这是 echo 复读的根因（四层补丁都在给它交利息，严厉审查 P0-3）。
    role 边界是防自我模仿的第一道结构防线。
    """
    msgs: list = []
    buf: list = []
    buf_is_bot = False

    def flush():
        nonlocal buf
        if not buf:
            return
        content = "\n".join(buf).strip()
        if content:
            msgs.append(AIMessage(content=content) if buf_is_bot
                        else HumanMessage(content=content))
        buf = []

    for line in background.split("\n"):
        s = line.strip()
        if s.startswith("你(历史):"):
            if not buf_is_bot:
                flush()
            buf_is_bot = True
            buf.append(s[len("你(历史):"):].strip())
        elif s and (":" in s or "：" in s):
            if buf_is_bot:
                flush()
            buf_is_bot = False
            buf.append(line.rstrip())
        else:
            # 续行（多行消息的后续行）并入当前缓冲
            buf.append(line.rstrip())
    flush()
    return msgs


def _called_tool_names(messages: list) -> set:
    names = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                names.add(tc.get("name"))
    return names


_ECHO_NUDGE = (
    "（系统提示）你刚才的回复在复读你最近已经说过的话（「{hit}」），"
    "真人不会复读自己。请换一个完全不同的说法或角度重新回应；"
    "如果实在没有新内容可说，调用 do_not_reply。"
    "注意：你上一轮的回复还没有发送出去，对方目前什么都没看到。"
)


def _plain_reply_text(msg) -> str:
    """轻量提取 AIMessage 文本（echo 补救轮用）：拼接分块 + 砍 think 链。"""
    if msg is None:
        return ""
    text = msg.content or ""
    if isinstance(text, list):
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        return ""
    return text


def _intent_nudge(latest_text: str, result_messages: list, available: set):
    """强意图命中但对应工具没调 -> (系统追问文本, 是否需全绑补救)，否则 None。

    背景：弱模型常把「帮我盯着xxx」当成记忆任务只调 save_memory 或纯口头
    答应——动作没生效用户却以为办好了。给它一次补救轮比换贵模型便宜。
    工具被掩码裁掉（漏绑）时 full_bind=True：补救轮用全量工具重建 agent
    （P5-2 兜底——2026-08-01 实战：模型被追问一个没绑定的工具，如实答「没有」）。
    """
    text = (latest_text or "").strip()
    if not text:
        return None
    called = _called_tool_names(result_messages)
    for keywords, tool_name in _INTENT_RULES:
        if any(kw in text for kw in keywords):
            if tool_name in called:
                continue             # 该意图组已调用——继续检查其余意图组
                                       # （曾 return None 短路：「提醒我开会，顺便画只猫」
                                       #  只查了第一个意图，ai_draw 没调也永不追问）
            intent = next(kw for kw in keywords if kw in text)
            return (_NUDGE_PROMPT.format(intent=intent, tool=tool_name),
                    tool_name not in available)
    return None


def _record_usage(messages: list, chat_id: str, request_type: str = "agent") -> None:
    """从 AIMessage.usage_metadata 提取 token 用量落库（失败静默）。"""
    try:
        from junjun_core.database import LLMUsage, db_writer
        for m in messages:
            if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None):
                u = m.usage_metadata
                model_name = (getattr(m, "response_metadata", {}) or {}).get("model_name", "")
                db_writer.submit(
                    LLMUsage.create,
                    time=time.time(), model_name=model_name, request_type=request_type,
                    prompt_tokens=int(u.get("input_tokens", 0)),
                    completion_tokens=int(u.get("output_tokens", 0)),
                    chat_id=chat_id,
                )
    except Exception as e:
        logger.debug(f"token 用量记录失败（忽略）: {e}")


def _apply_context_budget(
    *,
    is_group: bool,
    latest_text: str,
    mood_block: str,
    memory_block: str,
    relation_block: str,
    background: str,
    latest_msg: str,
    addressed: bool,
) -> tuple[list, str, dict]:
    """用 ContextBudget 组装 messages，返回 (messages, final_system_text, metrics)。"""
    from junjun_core.config import get_global_config
    cfg = get_global_config().raw.get("context_budget", {})
    max_tokens = int(cfg.get("max_total_tokens", 6000))
    reserve = int(cfg.get("reserve_tokens", 500))
    budget = ContextBudget(max_total_tokens=max_tokens, reserve_tokens=reserve)

    core_text, dynamic_blocks = build_prompt_blocks(
        is_group=is_group, latest_text=latest_text,
        mood_block=mood_block, memory_block=memory_block,
        relation_block=relation_block,
    )
    blocks: list[BudgetBlock] = [
        BudgetBlock(name="system", content=core_text, priority=1, required=True),
    ]
    for b in dynamic_blocks:
        blocks.append(BudgetBlock(
            name=b["name"], content=b["content"],
            priority=b.get("priority", 5),
            required=b.get("required", False),
        ))
    if background:
        blocks.append(BudgetBlock(name="background", content=background, priority=6))
    if latest_msg:
        blocks.append(BudgetBlock(
            name="latest", content=f"[你要回复的消息]\n{latest_msg}",
            priority=1, required=True,
        ))
    if addressed:
        blocks.append(BudgetBlock(
            name="addressed", content="最后一条消息明确 @ 你或直呼你的名字，你必须正面回应，禁止调用 do_not_reply。",
            priority=1, required=True,
        ))

    kept, metrics = budget.build_messages(blocks)
    # 组装 messages
    messages: list = []
    system_parts = []
    dynamic_parts = []
    bg_block = None
    latest_block = None
    for b in kept:
        if b.name == "system":
            system_parts.append(b.content)
        elif b.name in ("mood", "memory", "relation", "health"):
            dynamic_parts.append(b.content)
        elif b.name == "background":
            bg_block = b.content
        elif b.name == "latest":
            latest_block = b.content
        elif b.name == "addressed":
            system_parts.append(b.content)
    system_text = "\n\n".join(system_parts)
    if dynamic_parts:
        system_text += f"\n\n<state>\n{'\n\n'.join(dynamic_parts)}\n</state>"

    messages.append(SystemMessage(content=system_text))
    if bg_block:
        bg_msgs = _background_to_messages(bg_block)
        if bg_msgs:
            if isinstance(bg_msgs[0], HumanMessage):
                bg_msgs[0] = HumanMessage(content="[群聊背景，仅供参考]\n" + (bg_msgs[0].content or ""))
            else:
                bg_msgs.insert(0, HumanMessage(content="[群聊背景，仅供参考]"))
            messages.extend(bg_msgs)
    if latest_block:
        messages.append(HumanMessage(content=latest_block))
    elif not bg_block:
        # 没有背景也没有最新消息时，把整个 context 塞进去兜底
        messages.append(HumanMessage(content=latest_text or ""))

    return messages, system_text, metrics


class JunJunAgent:
    """单会话 Agent 封装。"""

    def __init__(self, session, model=None):
        self.session = session
        load_builtin()
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("agent")
        self._model = model  # 留引用：会话淘汰时关闭 httpx 连接池（防泄漏）

    def _build_agent(self, full: bool = False, allow_silence: bool = True):
        """每轮重建 agent 图：工具集按「当前」会话话题实时掩码。

        曾经只在 __init__ 绑一次——那时 memory 是空的，关键词钉不住任何工具，
        非 CORE 工具（如 run_background_task）被掩码裁掉后整个会话生命周期
        都不可用；而意图自检按实时掩码判定「可用」去追问，模型却被追问一个
        没绑定的工具（2026-08-01 实战 trace：模型如实回答「没有这个工具」）。
        重建成本是毫秒级图编译，相对秒级 LLM 调用可忽略；顺便让话题变化
        后掩码真正生效（设计本意就是按轮动态）。
        full=True：漏绑补救轮用全量工具（意图自检发现目标工具被裁掉时，
        P5-2 兜底）。
        allow_silence=False（必回场景）：从工具集摘除 do_not_reply——
        prompt 里「禁止调用 do_not_reply」只是劝告，模型不听话就真沉默了
        （2026-08-04 trace：管理员私聊问话被 do_not_reply 吞掉，output=null）。
        必回语义必须结构性强制：想沉默？没这个工具，只能出声。
        """
        tools = get_tools() if full else get_tools(self.session)
        if not allow_silence:
            tools = [t for t in tools if t.name != SILENCE_TOOL_NAME]
        if not full:
            # 后台预热 embedding 缓存：同步掩码路径只读缓存，这里喂它——
            # 本轮可能来不及，下一轮起语义相关性补满生效（失败静默）
            try:
                import asyncio
                from junjun_skills.registry import warm_tool_embeddings
                asyncio.create_task(warm_tool_embeddings(self.session))
            except Exception:
                pass
        return create_agent(model=self._model, tools=tools,
                            middleware=[PlanMiddleware()])

    async def aclose(self) -> None:
        """关闭模型客户端连接池（会话淘汰时调用）。best-effort，失败静默。"""
        import asyncio as _aio
        models = [self._model]
        models.extend(getattr(self._model, "fallbacks", None) or [])
        for m in models:
            ac = getattr(m, "async_client", None)
            try:
                if ac is not None and hasattr(ac, "close"):
                    res = ac.close()
                    if _aio.iscoroutine(res):
                        await res
            except Exception:
                pass

    async def process(
        self,
        context_text: str,
        callbacks: Optional[list] = None,
        latest_text: str = "",
        addressed: bool = False,
        mood_block: str = "",
        memory_block: str = "",
        relation_block: str = "",
        trace_id: str = "",
        system_prompt: str = "",
    ) -> Optional[str]:
        """跑一轮决策。返回回复文本；None 表示沉默。

        addressed: 被 @/直呼（mentioned_bot_reply 必回语义，禁用 do_not_reply）。
        trace_id: 本轮决策 ID（processor 生成），写结构化日志并进 Langfuse metadata，
                  供 WebUI 日志页与 Langfuse trace 互查。
        system_prompt: processor 已构建好的 prompt（它要为 Langfuse 快照构建一次）——
                  传入则复用，避免每轮构建两次且快照与实发不一致（严厉审查 P2-9）。
        """
        cfg = get_global_config()
        max_iter = int(cfg.raw.get("memory", {}).get("max_agent_iterations", 5))
        budget_cfg = cfg.raw.get("context_budget", {})
        budget_enabled = bool(budget_cfg.get("enable", False))

        # context_text 包含历史消息（可能含最新消息）。把最新消息剥离单独作为
        # HumanMessage 传入，context 只作为背景参考——模型明确知道「这是背景，这是你要回的」。
        context_lines = context_text.strip().split("\n") if context_text.strip() else []
        # 最后一条 user 消息（排除 bot 的「你(历史):」前缀和空行）作为最新指令
        latest_msg = ""
        background_lines = []
        for line in reversed(context_lines):
            stripped = line.strip()
            if not stripped:
                background_lines.insert(0, line)
                continue
            # 排除 bot 历史输出（「你(历史):」前缀）——它不是 user 消息
            if stripped.startswith("你(历史):"):
                background_lines.insert(0, line)
                continue
            # 排除 bot 输出的续行（不以「昵称:」或「你(历史):」开头的行）
            # user 消息格式为「昵称: 内容」或「昵称 [@你]: 内容」
            if not latest_msg and (":" in stripped or "：" in stripped):
                # 判定为 user 消息：有「昵称:」前缀且不是「你(历史):」
                latest_msg = line
            else:
                background_lines.insert(0, line)
        # 背景条数预算（[chat] background_context_lines，默认 30 对齐 processor 的
        # render(limit=30)）。曾经硬编 10 行——processor 渲染的 30 条在这被砍到只剩
        # 5-7 条可见（2026-08-03 用户实测反馈上下文太短）。群聊消息短，30 行约千级
        # token，换上下文完整度值；弱模型分心就调低这个配置，别在代码里砍。
        bg_budget = int(cfg.raw.get("chat", {}).get("background_context_lines", 30))
        background = "\n".join(background_lines[-bg_budget:])

        if budget_enabled and not system_prompt:
            messages, final_system, budget_metrics = _apply_context_budget(
                is_group=self.session.is_group,
                latest_text=latest_text,
                mood_block=mood_block,
                memory_block=memory_block,
                relation_block=relation_block,
                background=background,
                latest_msg=latest_msg.replace("【最新】", "").strip() if latest_msg else "",
                addressed=addressed,
            )
        else:
            system = system_prompt or build_system_prompt(
                is_group=self.session.is_group,
                latest_text=latest_text,
                mood_block=mood_block,
                memory_block=memory_block,
                relation_block=relation_block,
            )
            if addressed:
                system += "\n最后一条消息明确 @ 你或直呼你的名字，你必须正面回应，禁止调用 do_not_reply。"
            messages = [SystemMessage(content=system)]
            if background:
                bg_msgs = _background_to_messages(background)
                if bg_msgs:
                    if isinstance(bg_msgs[0], HumanMessage):
                        bg_msgs[0] = HumanMessage(
                            content="[群聊背景，仅供参考]\n" + (bg_msgs[0].content or ""))
                    else:
                        # 首轮是 bot 历史发言：标记单独成条，别让模型以为这话是它说的
                        bg_msgs.insert(0, HumanMessage(content="[群聊背景，仅供参考]"))
                    messages.extend(bg_msgs)
            if latest_msg:
                # 去掉「【最新】」前缀（processor 加的标记），还原原始消息
                clean_latest = latest_msg.replace("【最新】", "").strip()
                messages.append(HumanMessage(content=f"[你要回复的消息]\n{clean_latest}"))
            else:
                messages.append(HumanMessage(content=context_text))
            final_system = system
            budget_metrics = {}

        # 轻量规划循环（P0-12）：疑似复合任务先拆清单，工具循环里持续注入 ----
        plan_token = None
        plan_steps = None
        if bool(cfg.raw.get("plan", {}).get("enable", True)):
            target_text = latest_text or context_text
            if detect_complexity(target_text):
                plan_steps = await make_plan(target_text)
                if plan_steps:
                    logger.info(f"[{self.session.chat_id}] 任务清单({len(plan_steps)}步): "
                                f"{' → '.join(plan_steps)}")
                    plan_token = set_plan(plan_steps)
        # 多步任务加迭代预算：每步可能 1-2 次工具调用
        eff_iter = max_iter + len(plan_steps or [])

        agent = self._build_agent(allow_silence=not addressed)  # 每轮重建：工具掩码按当前话题实时生效；必回场景摘除沉默工具
        try:
            result = await agent.ainvoke(
                {"messages": messages},
                config={
                    "callbacks": callbacks or [],
                    "recursion_limit": 2 * eff_iter + 1,
                    "metadata": {
                        "chat_id": self.session.chat_id,
                        "trace_id": trace_id,
                        # Langfuse v3 CallbackHandler 识别的元数据：trace 按会话归组
                        "langfuse_session_id": self.session.chat_id,
                        "langfuse_tags": ["junjun", "agent"],
                        "context_budget": budget_metrics,
                        "system_prompt": final_system[:2000],
                    },
                },
            )
        except Exception as e:
            # 含 GraphRecursionError：超限兜底。被 @ 必回语义下沉默像装死，
            # 回一句实话（2026-08-01 trace：工具参数校验连错 5 次烧穿上限，用户被晾）
            logger.warning(f"agent 执行异常 [trace={trace_id}]: {type(e).__name__}: {e}")
            if addressed:
                return "……刚才那个请求把我绕进去了，没办成。换个说法再叫我一次？"
            return None
        finally:
            if plan_token is not None:
                reset_plan(plan_token)

        messages = result.get("messages", [])
        _record_usage(messages, self.session.chat_id)

        if _called_silence_tool(messages):
            logger.debug(f"[{self.session.chat_id}] agent 选择沉默")
            return None

        # ---- 意图自检（一轮补救）：强意图词命中但对应工具没调 -> 追问重来 ----
        if bool(cfg.raw.get("agent", {}).get("intent_retry", True)):
            try:
                available = {t.name for t in get_tools(self.session)}
                nudge_info = _intent_nudge(latest_text, messages, available)
            except Exception:
                nudge_info = None
            if nudge_info:
                nudge, full_bind = nudge_info
                logger.info(f"[{self.session.chat_id}] 意图自检：追问补调工具"
                            f"{'（全绑补救）' if full_bind else ''} [trace={trace_id}]")
                try:
                    retry_agent = (self._build_agent(full=True, allow_silence=not addressed)
                                   if full_bind else agent)
                    retry = await retry_agent.ainvoke(
                        {"messages": messages + [HumanMessage(content=nudge)]},
                        config={
                            "callbacks": callbacks or [],
                            "recursion_limit": 2 * eff_iter + 1,
                            "metadata": {
                                "chat_id": self.session.chat_id,
                                "trace_id": trace_id,
                                "langfuse_session_id": self.session.chat_id,
                                "langfuse_tags": ["junjun", "agent", "intent-retry"],
                            },
                        },
                    )
                    messages = retry.get("messages", messages)
                    _record_usage(messages, self.session.chat_id)
                    if _called_silence_tool(messages):
                        return None
                except Exception as e:
                    logger.warning(f"意图补救轮异常（沿用首轮结果）: {type(e).__name__}: {e}")

        # DeepSeek 官方规则（调研确认）：
        # - 无工具调用时：上一轮的 reasoning_content 禁止拼入后续 context（传了也被忽略）
        # - 有工具调用时：reasoning_content 必须完整回传，缺失直接 400
        # 实现：检测本轮是否有 tool_call，有则保留 reasoning 链；无则确保不发 reasoning
        has_tool_call = any(
            isinstance(m, AIMessage) and (m.tool_calls or [])
            for m in messages[-3:]  # 最近 3 条内有无 tool_call
        )
        last_msg = messages[-1] if messages else None
        text = ""
        if last_msg:
            reasoning = (getattr(last_msg, "additional_kwargs", {}) or {}).get("reasoning_content")
            if reasoning and has_tool_call:
                # 工具调用链内：reasoning 保留（DeepSeek 要求回传，否则 400）
                logger.debug(f"[{self.session.chat_id}] 工具链内 reasoning_content 保留 ({len(reasoning)} 字)")
            elif reasoning:
                # 无工具调用：reasoning 已分离，content 是最终答案
                logger.debug(f"[{self.session.chat_id}] reasoning_content 已分离 ({len(reasoning)} 字)")
            text = last_msg.content or ""
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        text = (text or "").strip()

        # 直接截断：<think>...</think> 之间的内容全砍（无 reasoning_content 字段时的兜底）
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            logger.warning(f"[{self.session.chat_id}] 未闭合 <think> 思考链泄漏，本轮沉默")
            return None

        # GLM-5 工具参数格式泄漏：模型把 do_not_reply 的 XML 参数格式当文本输出
        # （<arg_key>reason</arg_key><arg_value>...</arg_value>），不是真正调用工具
        if "<arg_key>" in text or "</arg_key>" in text:
            logger.warning(f"[{self.session.chat_id}] GLM-5 工具参数格式泄漏，本轮沉默")
            return None

        # 推理结构检测（无 reasoning_content 字段且 text 仍含推理时的最后保险）
        if text and len(text) > 200:
            _REASONING_STARTS = ("这个问题", "让我", "我需要", "首先", "根据系统",
                                 "根据提示", "用户在问", "对方在问", "分析一下")
            first_line = text.split("\n")[0].strip()
            if any(first_line.startswith(s) for s in _REASONING_STARTS):
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                if len(lines) >= 2:
                    tail = lines[-1]
                    if len(tail) < 150 and not any(tail.startswith(s) for s in _REASONING_STARTS):
                        logger.info(f"[{self.session.chat_id}] 推理结构检测，取尾部: {tail[:40]}")
                        text = tail
                    else:
                        logger.warning(f"[{self.session.chat_id}] 推理结构无法提取，本轮沉默")
                        return None
                else:
                    logger.warning(f"[{self.session.chat_id}] 推理结构无法提取，本轮沉默")
                    return None

        # ---- 复读自检（echo guard，2026-08-04）：与近期自身发言撞车 -> 追问重说 ----
        # 背景：bot 复读的话术落进短期记忆，下一轮 context 里同一句话堆 N 次，
        # 模型把「自己老说这句」当成说话习惯继续复读——自我污染正反馈。
        # 输入端 render 已去重，这里守出口：撞车追问一轮，仍撞车则沉默
        # （被 @ 必回场景发重试稿——至少模型挣扎过一次）。
        agent_cfg = cfg.raw.get("agent", {})
        if text and bool(agent_cfg.get("echo_guard", True)):
            try:
                from junjun_memory.echo import (
                    extract_catchphrases, is_echo, normalize_echo)
                sim = float(agent_cfg.get("echo_similarity", 0.85))
                k = int(agent_cfg.get("echo_recent_k", 8))
                memory = getattr(self.session, "memory", None)
                all_bot = ([e.text for e in memory.entries if e.role == "bot"]
                           if memory is not None else [])
                recent_bot = all_bot[-k:]
                # 口头禅检测：整句不像但嵌着近期用滥的词组（「姐姐疼你」式）。
                # 挖掘窗口比整句撞车宽（默认 30 条）——口头禅是跨消息模式
                cp_min = int(agent_cfg.get("echo_catchphrase_count", 3))
                cp_k = int(agent_cfg.get("echo_catchphrase_k", 30))
                catchphrases = extract_catchphrases(all_bot[-cp_k:],
                                                    min_count=cp_min) if all_bot else []

                def _echo_hit(t: str):
                    h = is_echo(t, recent_bot, similarity=sim) if recent_bot else None
                    if h is not None:
                        return h
                    norm_t = normalize_echo(t)
                    for cp in catchphrases:
                        if cp in norm_t:
                            return cp
                    return None

                hit = _echo_hit(text)
            except Exception:
                hit = None
            if hit is not None:
                logger.info(f"[{self.session.chat_id}] 复读自检命中，追问重说 "
                            f"[trace={trace_id}]: {text[:30]} ≈ {hit[:30]}")
                try:
                    retry = await agent.ainvoke(
                        {"messages": messages + [
                            HumanMessage(content=_ECHO_NUDGE.format(hit=hit[:60]))]},
                        config={
                            "callbacks": callbacks or [],
                            "recursion_limit": 2 * eff_iter + 1,
                            "metadata": {
                                "chat_id": self.session.chat_id,
                                "trace_id": trace_id,
                                "langfuse_session_id": self.session.chat_id,
                                "langfuse_tags": ["junjun", "agent", "echo-retry"],
                            },
                        },
                    )
                    rmsgs = retry.get("messages", messages)
                    _record_usage(rmsgs, self.session.chat_id)
                    if _called_silence_tool(rmsgs):
                        return None
                    rtext = _plain_reply_text(rmsgs[-1] if rmsgs else None)
                    if rtext and is_echo(rtext, recent_bot, similarity=sim) is None:
                        text = rtext                       # 重说成功
                    elif not addressed:
                        logger.info(f"[{self.session.chat_id}] 重说仍复读，本轮沉默")
                        return None                        # 非必回：宁可沉默不复读
                    elif rtext:
                        logger.warning(f"[{self.session.chat_id}] 被@必回但重说仍复读，"
                                       f"发重试稿: {rtext[:30]}")
                        text = rtext
                except Exception as e:
                    logger.warning(f"复读补救轮异常（按原稿处理）: {type(e).__name__}: {e}")
                    if not addressed:
                        return None
        return text or None
