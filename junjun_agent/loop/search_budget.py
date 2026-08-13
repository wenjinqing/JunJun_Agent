"""检索软刹车（2026-08-13 trace 4742c8fd 实锤）：主对话循环的搜索次数预算。

病灶 trace：一次「研究笔记」请求在对话通道里 9 轮生成、20 次搜索
（web_search 三家引擎全跪只剩 bing 裸兜底出「2026 Calendar」垃圾结果、
tavily 半败 ECONNRESET，模型在三个引擎间换乘救火），递归上限 19 硬烧穿，
已经写成的报告被整体丢弃，用户只收到一句搪塞话。
对话通道的迭代预算装不下重度调研——但模型没有「够了」的信号，
唯一的停止条件是崩溃。

本中间件给检索类工具设每轮（一次 process 调用，含追问重试）软上限：
预算内正常放行；用完后短路成结构化文本，要求模型立刻基于已搜集信息
整理作答、缺口诚实说明。硬崩溃 -> 软着陆。
重度调研的正路是 deep_research 派单（独立循环与预算），不经此限制；
TaskKernel 执行器直接调注册表工具，也不经此限制。
"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from junjun_core.observability import get_logger

logger = get_logger("agent.search_budget")

_DEFAULT_BUDGET = 6


def _budget_cfg() -> int:
    """[agent] search_budget，默认 6；0 = 关闭刹车。配置坏了按默认（宁刹勿放）。"""
    try:
        from junjun_core.config import get_global_config
        return int(get_global_config().raw.get("agent", {}).get(
            "search_budget", _DEFAULT_BUDGET))
    except Exception:
        return _DEFAULT_BUDGET


def _blocked_text(budget: int) -> str:
    return (f'[TOOL_ERROR kind=预算 suggestion="本轮检索额度已用完，禁止再搜：'
            f'基于已搜集到的信息立即整理作答；仍有缺口在回答里诚实说明，'
            f'或建议对方换个说法派深研单（如「帮我深度调研一下xx」）"] '
            f'本轮对话的搜索次数已达上限（{budget} 次），本次调用未执行。')


class SearchBudgetMiddleware(AgentMiddleware):
    """检索类工具（名字含 search）每轮软上限。

    实例随 _build_agent 每轮新建 -> 计数天然按轮隔离；意图追问/复读追问等
    重试复用同一 agent 实例时预算延续——同一用户回合，追问不该重开闸。
    按「尝试」计数（失败的搜索同样烧迭代，这正是要刹的换乘救火）。
    """

    name = "search_budget"

    def __init__(self):
        self._used = 0

    async def awrap_tool_call(self, request, handler):
        name = str(request.tool_call.get("name", ""))
        if "search" not in name:
            return await handler(request)
        budget = _budget_cfg()
        if budget > 0 and self._used >= budget:
            logger.warning(f"检索预算用完（{budget}），短路 {name} 要求立即作答")
            return ToolMessage(
                content=_blocked_text(budget),
                tool_call_id=str(request.tool_call.get("id", "")),
                name=name,
            )
        self._used += 1
        return await handler(request)
