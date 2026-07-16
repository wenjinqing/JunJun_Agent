"""Agent 核心：LangChain 1.x create_agent（LangGraph runtime）。

每会话独立 agent 实例 + 独立消息历史，防跨会话串味。
决策语义：reply -> 文本输出；no_reply -> do_not_reply 工具置沉默状态。
"""

from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

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

# max_agent_iterations=5 -> LangGraph recursion_limit（2*N+1）
_RECURSION_LIMIT = 11


class JunJunAgent:
    """单会话 Agent 封装。"""

    def __init__(self, session, model=None):
        self.session = session
        load_builtin()
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("agent")
        self._agent = create_agent(
            model=model,
            tools=get_tools(session),
            system_prompt=build_system_prompt(is_group=session.group_id is not None),
        )

    async def process(self, context_text: str, callbacks: Optional[list] = None) -> Optional[str]:
        """跑一轮决策。返回回复文本；None 表示沉默。"""
        cfg = get_global_config()
        max_iter = int(cfg.raw.get("memory", {}).get("max_agent_iterations", 5))
        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=context_text)]},
                config={
                    "callbacks": callbacks or [],
                    "recursion_limit": 2 * max_iter + 1,
                    "metadata": {"chat_id": self.session.chat_id},
                },
            )
        except Exception as e:
            # 含 GraphRecursionError：超限兜底沉默，不炸会话
            logger.warning(f"agent 执行异常，本轮沉默: {type(e).__name__}: {e}")
            return None

        messages = result.get("messages", [])
        if _called_silence_tool(messages):
            logger.debug(f"[{self.session.chat_id}] agent 选择沉默")
            return None

        text = messages[-1].content if messages else ""
        if isinstance(text, list):  # 部分模型返回 content blocks
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        text = (text or "").strip()
        return text or None
