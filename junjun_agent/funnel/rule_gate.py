"""决策门（0 token，纯函数）。

现状语义（2026-08-04 严厉审查 P0-2 后收敛）：
- 私聊 -> 直通主 Agent
- 群聊 -> 仅 @/昵称直呼 进主 Agent，其余沉默
- 自消息永远丢弃（防回环）

历史：曾有三级漏斗（L1 规则门 talk_value 概率 + L2 LLM 语义门 + L3 主 Agent），
L1/L2 在生产路径从未被调用，frequency.py 的 LLM 频率评估每 160s 烧一次
LLM 调用更新无人消费的 adjust_factor——死代码已删（git 历史可查）。
"""

from dataclasses import dataclass


@dataclass
class L1Config:
    mentioned_bot_reply: bool = True
    nickname: str = ""
    alias_names: tuple = ()


def is_addressed(text: str, cfg: L1Config, at_bot: bool) -> bool:
    """被 @ 或昵称/别名直呼。"""
    if at_bot:
        return True
    names = [cfg.nickname, *cfg.alias_names]
    return any(n and n in text for n in names)
