"""决策漏斗 L2: 语义门（小模型单次调用）。

对齐原 chat_v2/v2_native_planner_gate 语义：仅输出三值
reply / no_reply / no_reply_until_call，不做其他动作决策。
JSON 脏输出宽松解析，失败默认 no_reply。
"""

import json
import re
from enum import Enum
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from junjun_core.observability import get_logger

logger = get_logger("funnel.gate")


class GateDecision(Enum):
    REPLY = "reply"
    NO_REPLY = "no_reply"
    NO_REPLY_UNTIL_CALL = "no_reply_until_call"


_GATE_SYSTEM = """你是 QQ 群聊机器人"{nickname}"的发言决策器。根据最近对话判断是否应该回复最后一条消息。

规则：
- 消息明确针对你、提问、或话题你能自然切入 -> reply
- 闲聊与你无关、你刚说过话不宜刷屏、插话会尴尬 -> no_reply
- 群里明确表示让你闭嘴/别说话 -> no_reply_until_call

只输出 JSON：{{"decision": "reply|no_reply|no_reply_until_call", "reason": "简短原因"}}"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def parse_gate_output(raw: str) -> GateDecision:
    """宽松解析 LLM 输出；失败默认 no_reply。"""
    try:
        m = _JSON_RE.search(raw or "")
        if not m:
            return GateDecision.NO_REPLY
        decision = json.loads(m.group(0)).get("decision", "")
        return GateDecision(decision)
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"gate 输出解析失败，默认 no_reply: {raw[:120]!r}")
        return GateDecision.NO_REPLY


async def llm_gate(
    context_text: str,
    nickname: str,
    *,
    model=None,
    callbacks: Optional[list] = None,
) -> GateDecision:
    """小模型判断是否回复。model 可注入（测试用 fake）。"""
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("gate")
    messages = [
        SystemMessage(content=_GATE_SYSTEM.format(nickname=nickname)),
        HumanMessage(content=f"最近对话：\n{context_text}\n\n是否回复最后一条？"),
    ]
    try:
        resp = await model.ainvoke(messages, config={"callbacks": callbacks or []})
        decision = parse_gate_output(resp.content)
        logger.debug(f"L2 gate -> {decision.value}")
        return decision
    except Exception as e:
        logger.warning(f"L2 gate 调用失败，默认 no_reply: {e}")
        return GateDecision.NO_REPLY
