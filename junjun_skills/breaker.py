"""工具连续失败熔断（P2，2026-08-09 反馈闭环）。

同工具在同轮决策里连错 N 次（默认 2）后，后续调用短路成「熔断」错误文本，
不再真执行——防模型拿着网络错误无限换乘烧 token（对齐 12-Factor Agents
Factor 9 的 consecutive error counter：连错到阈值就强制升级，不许死磕）。

作用域纪律：
- 按 (chat_id, tool_name) 计**连续**失败，成功即清零；
- 每轮决策开始（agent.process）整会话清零——熔断只管「同一轮里的死磕」，
  不拿上一轮的历史失败惩罚新一轮（保守方向：宁可下一轮再试一次）。
- 只统计异常（registry 错误包装层看到的）；业务性拒绝文案是正常返回，
  不计数（拒绝不是故障）。
"""

from collections import defaultdict

from junjun_core.observability import get_logger

logger = get_logger("skills.breaker")

_failures: dict = defaultdict(int)  # (chat_id, tool_name) -> 连续失败次数


def _threshold() -> int:
    try:
        from junjun_core.config import get_global_config
        return int(get_global_config().raw.get("tools", {}).get("max_consecutive_failures", 2))
    except Exception:
        return 2


def note_success(chat_id: str, tool: str) -> None:
    _failures.pop((chat_id, tool), None)


def note_failure(chat_id: str, tool: str) -> int:
    key = (chat_id, tool)
    _failures[key] += 1
    return _failures[key]


def is_open(chat_id: str, tool: str) -> bool:
    return _failures.get((chat_id, tool), 0) >= _threshold()


def reset_chat(chat_id: str) -> None:
    """每轮决策开始清零该会话所有计数。"""
    for key in [k for k in _failures if k[0] == chat_id]:
        _failures.pop(key, None)
