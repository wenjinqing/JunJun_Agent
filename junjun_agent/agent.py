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
        try:
            result = await self._agent.ainvoke(
                {"messages": [SystemMessage(content=system), HumanMessage(content=context_text)]},
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

        text = messages[-1].content if messages else ""
        if isinstance(text, list):  # 部分模型返回 content blocks
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        text = (text or "").strip()

        # 防御 1：LLM 自白式思考泄漏（<think> 标签场景）
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            logger.warning(f"[{self.session.chat_id}] 未闭合 <think> 思考链泄漏，本轮沉默")
            return None

        # 防御 2：无标签推理残留（deepseek function calling 后常见）——
        # content 以推理开头词起始且后续有「正式回复」迹象的，视为推理泄漏
        _REASONING_STARTS = ("这个问题", "让我", "我需要", "首先", "根据系统", "根据提示",
                             "用户在问", "对方在问", "分析一下", "思考一下")
        if any(text.startswith(s) for s in _REASONING_STARTS):
            # 找最后一个换行后的段落作为真正回复（推理通常在前半段）
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if len(lines) >= 2:
                # 最后一段如果是短句且不像推理，取它
                tail = lines[-1]
                if len(tail) < 100 and not any(tail.startswith(s) for s in _REASONING_STARTS):
                    logger.info(f"[{self.session.chat_id}] 推理残留检测，取尾部回复: {tail[:30]}")
                    text = tail
                else:
                    logger.warning(f"[{self.session.chat_id}] 推理残留无法提取有效回复，本轮沉默")
                    return None
            else:
                logger.warning(f"[{self.session.chat_id}] 推理残留无法提取有效回复，本轮沉默")
                return None
        return text or None
