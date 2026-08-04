"""决策门：私聊直通，群聊仅 @/直呼进主 Agent（2026-08-04 起收敛为此）。"""

from junjun_agent.funnel.rule_gate import L1Config, is_addressed

__all__ = ["L1Config", "is_addressed"]
