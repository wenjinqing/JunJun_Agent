"""决策漏斗 L1: 规则门（0 token，纯函数）。

语义对齐原项目：
- mentioned_bot_reply=true 时被 @/昵称直呼 -> 直通 L3（旁路 L2）
- 私聊 -> 直通 L2
- 群聊非 @ -> talk_value 概率未命中 -> 拦截
- 沉默模式（no_reply_until_call）中仅被呼唤可解除
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class L1Result(Enum):
    DROP = "drop"                # 拦截，不再处理
    TO_GATE = "to_gate"          # 进 L2 语义门
    TO_AGENT = "to_agent"        # 直通 L3 主 Agent（@ 旁路）


@dataclass
class L1Config:
    talk_value: float = 0.9
    mentioned_bot_reply: bool = True
    nickname: str = ""
    alias_names: tuple = ()


def is_addressed(text: str, cfg: L1Config, at_bot: bool) -> bool:
    """被 @ 或昵称/别名直呼。"""
    if at_bot:
        return True
    names = [cfg.nickname, *cfg.alias_names]
    return any(n and n in text for n in names)


def rule_gate(
    *,
    text: str,
    is_group: bool,
    at_bot: bool,
    is_self: bool,
    silenced_until_call: bool,
    cfg: L1Config,
    rand: Optional[random.Random] = None,
) -> L1Result:
    rng = rand or random
    # 1. 自消息永远丢弃（防回环）
    if is_self:
        return L1Result.DROP

    addressed = is_addressed(text, cfg, at_bot)

    # 2. 沉默模式：只有被呼唤才解除并处理
    if silenced_until_call and not addressed:
        return L1Result.DROP

    # 3. @ / 直呼旁路（mentioned_bot_reply 必回，不受 talk_value 影响）
    if addressed and cfg.mentioned_bot_reply:
        return L1Result.TO_AGENT

    # 4. 私聊直通语义门（原 Brain 语义：基本都回，仍过 gate 防刷）
    if not is_group:
        return L1Result.TO_GATE

    # 5. 群聊非 @：talk_value 概率
    if rng.random() < cfg.talk_value:
        return L1Result.TO_GATE
    return L1Result.DROP
