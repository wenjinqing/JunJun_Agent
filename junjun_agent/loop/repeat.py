"""复读参与：对齐原 repeat_plugin 语义（主动跟复读，不是防复读）。

- 群内连续 threshold 条相同消息（不同人发）-> 冷却允许 -> 君君跟一条
- 自消息不计入；跟读后进冷却防跟自己复读的风暴
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("loop.repeat")


def _cfg() -> dict:
    return get_global_config().raw.get("repeat", {})


@dataclass
class RepeatState:
    content: str = ""
    count: int = 0
    users: set = field(default_factory=set)
    last_repeat_content: str = ""
    last_repeat_time: float = 0.0


class RepeatDetector:
    def __init__(self):
        self._states: Dict[str, RepeatState] = {}

    def note(self, chat_id: str, user_id: str, text: str, *, is_self: bool = False,
             now: Optional[float] = None) -> Optional[str]:
        """记录消息。返回应跟读的内容（不触发返回 None）。"""
        cfg = _cfg()
        if not cfg.get("enable", True):
            return None
        text = (text or "").strip()
        min_len = int(cfg.get("min_message_length", 1))
        max_len = int(cfg.get("max_message_length", 50))
        st = self._states.setdefault(chat_id, RepeatState())

        if is_self or not text or not (min_len <= len(text) <= max_len):
            st.content, st.count, st.users = "", 0, set()  # 断链
            return None

        if text == st.content:
            st.users.add(user_id)
            st.count += 1
        else:
            st.content, st.count, st.users = text, 1, {user_id}

        threshold = int(cfg.get("threshold", 4))
        now = now if now is not None else time.time()
        interval = float(cfg.get("min_interval_seconds", 60))
        if (st.count >= threshold
                and len(st.users) >= 2                      # 至少两个不同人在复读
                and text != st.last_repeat_content           # 同内容不二跟
                and (now - st.last_repeat_time) >= interval):
            st.last_repeat_content = text
            st.last_repeat_time = now
            st.content, st.count, st.users = "", 0, set()   # 跟完断链
            logger.info(f"[{chat_id}] 参与复读: {text[:30]}")
            return text
        return None


repeat_detector = RepeatDetector()
