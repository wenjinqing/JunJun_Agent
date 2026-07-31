"""轻量规划循环（P0-12）：复杂多步任务的「清单注入 + 进度回填」。

不引入 LangGraph 子图（GLM 级 function calling 稳定性撑不起），
用模型无关的弱约束：
1. 规则粗判复杂度（0 token）：多动作域关键词 + 连接词
2. utils 槽 LLM 把请求拆成 2-4 步清单
3. PlanMiddleware.awrap_model_call 在**每次模型迭代**把清单 + 已发起
   工具调用数注入本次请求（request.override 只改本次，不污染消息历史）——
   模型在工具循环中始终「看得见清单」，半途忘步骤显著减少。

开关：bot_config [plan] enable（默认 true）。
"""

import re
from contextvars import ContextVar
from typing import List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage

from junjun_core.observability import get_logger

logger = get_logger("loop.plan_tracker")

# 本轮任务清单（agent.process 设置/复位；contextvar 保证跨会话隔离）
_current_plan: ContextVar[Optional[List[str]]] = ContextVar("plan_steps", default=None)

_CONNECTORS = ("然后", "接着", "之后", "顺便", "并且", "还要", "同时",
               "再帮", "再给", "先", "再")
_ACTION_KEYWORDS = (
    "搜", "查", "画", "发空间", "说说", "提醒", "下载", "点歌", "总结",
    "翻译", "签到", "看空间", "整理", "写", "小说", "抽签",
)
_MAX_STEPS = 4


def detect_complexity(text: str) -> bool:
    """规则粗判疑似复合任务：动作词≥2 且有连接词，或动作词≥3。"""
    text = (text or "").strip()
    if len(text) < 12:
        return False
    hits = sum(1 for k in _ACTION_KEYWORDS if k in text)
    has_conn = any(c in text for c in _CONNECTORS)
    return (hits >= 2 and has_conn) or hits >= 3


async def make_plan(text: str) -> Optional[List[str]]:
    """utils 槽 LLM 拆步骤；失败/不足 2 步返回 None（降级无清单）。"""
    prompt = (
        "把下面这个请求拆成有序执行步骤（2-4 步，每步不超过 15 个字）。\n"
        "只输出编号列表，每行一步，不要任何解释。如果其实就是一步能完成的事，"
        "只输出一行「1. 直接回复」。\n"
        f"请求：{text}"
    )
    try:
        from langchain_core.messages import HumanMessage
        from junjun_llm import get_chat_model
        resp = await get_chat_model("utils").ainvoke([HumanMessage(content=prompt)])
        content = resp.content
        if isinstance(content, list):
            content = "".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p)
                for p in content)
        steps = []
        for line in str(content or "").splitlines():
            m = re.match(r"^\s*\d+\s*[.、)]\s*(.+?)\s*$", line)
            if m:
                steps.append(m.group(1))
        steps = [s for s in steps if s and "直接回复" not in s][:_MAX_STEPS]
        if len(steps) >= 2:
            return steps
    except Exception as e:
        logger.warning(f"计划生成失败（降级无清单）: {type(e).__name__}: {e}")
    return None


def set_plan(steps: Optional[List[str]]):
    """设置本轮清单；返回 token 供 finally 复位。"""
    return _current_plan.set(steps if steps and len(steps) >= 2 else None)


def reset_plan(token) -> None:
    _current_plan.reset(token)


def _reminder(steps: List[str], tool_calls: int) -> str:
    lines = ["[任务清单] 这是一个多步任务，按顺序完成："]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    lines.append(
        f"（已发起 {tool_calls} 次工具调用。对照清单逐步执行；"
        "全部步骤完成前不要急着回复用户；某步失败可换方式重试或如实说明，"
        "然后继续后面的步骤。）")
    return "\n".join(lines)


class PlanMiddleware(AgentMiddleware):
    """每次模型迭代把任务清单 + 进度注入本次调用（不写回消息历史）。"""

    name = "plan_tracker"

    async def awrap_model_call(self, request, handler):
        steps = _current_plan.get()
        if not steps:
            return await handler(request)
        tool_calls = sum(1 for m in request.messages
                         if isinstance(m, ToolMessage))
        reminder = SystemMessage(content=_reminder(steps, tool_calls))
        try:
            request = request.override(
                messages=[*request.messages, reminder])
        except Exception as e:
            logger.debug(f"清单注入失败（本次跳过）: {e}")
            return await handler(request)
        return await handler(request)
