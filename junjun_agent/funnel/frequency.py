"""频率控制：talk_value 时段规则 + LLM 动态调节因子。

对齐原 chat/frequency_control：
- talk_value_rules: [{target, time, value}] 按时段/会话覆盖基础 talk_value，
  具体 chat 优先于全局，支持跨夜区间
- 动态调节：>=160s 冷却且新消息 >=20 条时小模型评「过于频繁/过少/正常」，
  调节因子 *0.8/*1.2 夹在 [0.1, 1.5]
"""

import time as time_mod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("funnel.freq")

_ADJUST_COOLDOWN = 160.0
_ADJUST_MIN_MSGS = 20


def _parse_hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _in_range(now_min: int, rng: str) -> bool:
    """time="HH:MM-HH:MM"，支持跨夜（23:00-02:00）。"""
    try:
        start_s, end_s = rng.split("-")
        start, end = _parse_hhmm(start_s), _parse_hhmm(end_s)
    except (ValueError, AttributeError):
        return False
    if start <= end:
        return start <= now_min <= end
    return now_min >= start or now_min <= end  # 跨夜


def resolve_talk_value(chat_id: str, *, now: Optional[datetime] = None) -> float:
    """基础 talk_value + 时段规则（具体 chat 优先于全局 target=""）。"""
    chat_cfg = get_global_config().raw.get("chat", {})
    base = float(chat_cfg.get("talk_value", 0.9))
    if not chat_cfg.get("enable_talk_value_rules", False):
        return base
    rules = chat_cfg.get("talk_value_rules", []) or []
    now_min = (now or datetime.now()).hour * 60 + (now or datetime.now()).minute

    matched_global: Optional[float] = None
    for rule in rules:
        target = str(rule.get("target", ""))
        if not _in_range(now_min, str(rule.get("time", ""))):
            continue
        value = float(rule.get("value", base))
        if target == chat_id:
            return value  # 具体会话规则直接生效
        if target == "":
            matched_global = value
    return matched_global if matched_global is not None else base


@dataclass
class FrequencyState:
    """单会话频率调节状态。"""
    adjust_factor: float = 1.0
    last_adjust_time: float = 0.0
    msgs_since_adjust: int = 0


class FrequencyControl:
    """会话级频率控制器（调节因子由 LLM 评估驱动）。"""

    def __init__(self):
        self._states: Dict[str, FrequencyState] = {}

    def state(self, chat_id: str) -> FrequencyState:
        if chat_id not in self._states:
            self._states[chat_id] = FrequencyState()
        return self._states[chat_id]

    def effective_talk_value(self, chat_id: str, *, now: Optional[datetime] = None) -> float:
        v = resolve_talk_value(chat_id, now=now) * self.state(chat_id).adjust_factor
        return max(0.0, min(1.0, v))

    def note_message(self, chat_id: str) -> None:
        self.state(chat_id).msgs_since_adjust += 1

    def should_evaluate(self, chat_id: str, *, now_ts: Optional[float] = None) -> bool:
        st = self.state(chat_id)
        now = now_ts if now_ts is not None else time_mod.time()
        return (now - st.last_adjust_time) >= _ADJUST_COOLDOWN and st.msgs_since_adjust >= _ADJUST_MIN_MSGS

    def apply_evaluation(self, chat_id: str, verdict: str, *, now_ts: Optional[float] = None) -> None:
        """verdict: 过于频繁 / 过少 / 正常。"""
        st = self.state(chat_id)
        if "频繁" in verdict:
            st.adjust_factor *= 0.8
        elif "过少" in verdict:
            st.adjust_factor *= 1.2
        st.adjust_factor = max(0.1, min(1.5, st.adjust_factor))
        st.last_adjust_time = now_ts if now_ts is not None else time_mod.time()
        st.msgs_since_adjust = 0
        logger.info(f"[{chat_id}] 频率调节: {verdict} -> factor={st.adjust_factor:.2f}")

    async def evaluate_with_llm(self, chat_id: str, recent_text: str, *, model=None, callbacks=None) -> None:
        """小模型评估最近发言频率并应用调节。失败静默。"""
        from langchain_core.messages import HumanMessage
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        prompt = (
            "以下是 QQ 群最近的聊天记录，其中「你:」开头的是机器人发言。"
            "评估机器人发言频率，只输出一个词：正常 / 过于频繁 / 过少。\n\n" + recent_text
        )
        try:
            resp = await model.ainvoke([HumanMessage(content=prompt)], config={"callbacks": callbacks or []})
            self.apply_evaluation(chat_id, str(resp.content).strip())
        except Exception as e:
            logger.warning(f"频率评估失败（忽略）: {e}")


frequency_control = FrequencyControl()
