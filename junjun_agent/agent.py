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
from junjun_skills.registry import get_tools, load_builtin
from junjun_skills.builtin.do_not_reply import SILENCE_TOOL_NAME
from junjun_agent.persona import build_system_prompt

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


class JunJunAgent:
    """单会话 Agent 封装。"""

    def __init__(self, session, model=None):
        self.session = session
        load_builtin()
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("agent")
        # system prompt 留空，每轮由 process() 动态注入 SystemMessage
        self._agent = create_agent(model=model, tools=get_tools(session))

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
    ) -> Optional[str]:
        """跑一轮决策。返回回复文本；None 表示沉默。

        addressed: 被 @/直呼（mentioned_bot_reply 必回语义，禁用 do_not_reply）。
        trace_id: 本轮决策 ID（processor 生成），写结构化日志并进 Langfuse metadata，
                  供 WebUI 日志页与 Langfuse trace 互查。
        """
        cfg = get_global_config()
        max_iter = int(cfg.raw.get("memory", {}).get("max_agent_iterations", 5))
        system = build_system_prompt(
            is_group=self.session.is_group,
            latest_text=latest_text,
            mood_block=mood_block,
            memory_block=memory_block,
            relation_block=relation_block,
        )
        if addressed:
            system += "\n最后一条消息明确 @ 你或直呼你的名字，你必须正面回应，禁止调用 do_not_reply。"
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
        background = "\n".join(background_lines[-10:])  # 背景留最近 10 轮（记忆效果 vs 防分心平衡）

        messages = [SystemMessage(content=system)]
        if background:
            messages.append(HumanMessage(content=f"[群聊背景，仅供参考]\n{background}"))
        if latest_msg:
            # 去掉「【最新】」前缀（processor 加的标记），还原原始消息
            clean_latest = latest_msg.replace("【最新】", "").strip()
            messages.append(HumanMessage(content=f"[你要回复的消息]\n{clean_latest}"))
        else:
            messages.append(HumanMessage(content=context_text))

        try:
            result = await self._agent.ainvoke(
                {"messages": messages},
                config={
                    "callbacks": callbacks or [],
                    "recursion_limit": 2 * max_iter + 1,
                    "metadata": {
                        "chat_id": self.session.chat_id,
                        "trace_id": trace_id,
                        # Langfuse v3 CallbackHandler 识别的元数据：trace 按会话归组
                        "langfuse_session_id": self.session.chat_id,
                        "langfuse_tags": ["junjun", "agent"],
                    },
                },
            )
        except Exception as e:
            # 含 GraphRecursionError：超限兜底沉默，不炸会话
            logger.warning(f"agent 执行异常，本轮沉默 [trace={trace_id}]: {type(e).__name__}: {e}")
            return None

        messages = result.get("messages", [])
        _record_usage(messages, self.session.chat_id)

        if _called_silence_tool(messages):
            logger.debug(f"[{self.session.chat_id}] agent 选择沉默")
            return None

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
        return text or None
