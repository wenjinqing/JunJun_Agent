"""重复调用熔断（2026-08-14 trace bc95cd3b 实锤）：同工具同参数的卡死循环检测。

病灶 trace：管理员私聊派复杂数据分析任务，模型读了 code-lab 手册后
被「run_code 需审批」的描述吓住，6 轮生成里 use_skill("code-lab")×3
与 manage_mood×2 交替空转，始终没敢真的调 run_code（其实管理员直跑），
把 max_agent_iterations=5 撞穿，用户只收到一句「没办成」。
这不是步数不够——模型根本没在推进，加大上限只会多烧一倍 token。

本中间件给「同工具+同参数」的调用设每轮总量上限：前两次放行（重复读
手册/刷新状态可能是合法的），第三次起短路成结构化文本，把上次结果摘要
拍回去并要求立刻推进。硬烧穿 -> 软纠偏。
与 SearchBudgetMiddleware 互补：那个管「检索类总次数」，这个管「任何
工具的复读机」；TaskKernel 执行器不调对话 agent，不经此限制。
"""

import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from junjun_core.observability import get_logger

logger = get_logger("agent.repeat_guard")

_DEFAULT_LIMIT = 3   # 同（工具,参数）每轮允许次数；第 limit 次起短路


def _limit_cfg() -> int:
    """[agent] repeat_call_limit，默认 3；0 = 关闭。配置坏了按默认（宁刹勿放）。"""
    try:
        from junjun_core.config import get_global_config
        return int(get_global_config().raw.get("agent", {}).get(
            "repeat_call_limit", _DEFAULT_LIMIT))
    except Exception:
        return _DEFAULT_LIMIT


def _fingerprint(name: str, args) -> str:
    """（工具名, 规范化参数）指纹：参数 JSON 化排序键，无法序列化退 repr。"""
    try:
        return f"{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
    except Exception:
        return f"{name}|{repr(args)}"


class RepeatCallGuardMiddleware(AgentMiddleware):
    """同（工具,参数）调用每轮限次：第 N 次起短路，拍回上次结果并要求推进。

    实例随 _build_agent 每轮新建 -> 计数天然按轮隔离（同 SearchBudget）。
    按「尝试」计数：复读的每一次尝试都在烧迭代，这正是要刹的。
    """

    name = "repeat_call_guard"

    def __init__(self):
        self._seen: dict = {}   # fingerprint -> (次数, 上次结果摘要)

    async def awrap_tool_call(self, request, handler):
        name = str(request.tool_call.get("name", ""))
        fp = _fingerprint(name, request.tool_call.get("args"))
        limit = _limit_cfg()
        if limit <= 0:
            return await handler(request)
        count, last_digest = self._seen.get(fp, (0, ""))
        if count >= limit - 1 and count > 0:
            logger.warning(f"复读熔断：{name} 同参第 {count + 1} 次，短路要求推进")
            return ToolMessage(
                content=(f'[TOOL_ERROR kind=复读熔断 suggestion="相同的调用已执行 '
                         f'{count} 次，结果不会有新变化。禁止再重复调用它：'
                         f'基于已有结果立刻推进任务（该派单派单、该作答作答），'
                         f'如果上次的执行失败了，换一个不同的做法而不是原样重试"] '
                         f'上次结果摘要：{last_digest or "（无）"}'),
                tool_call_id=str(request.tool_call.get("id", "")),
                name=name,
            )
        resp = await handler(request)
        digest = str(getattr(resp, "content", ""))[:120]
        self._seen[fp] = (count + 1, digest)
        return resp
