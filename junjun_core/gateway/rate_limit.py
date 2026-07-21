"""网关速率限制：每会话令牌桶，防刷屏（阶段 1 交付物）。

配置 [gateway]：
  rate_limit_capacity = 8        # 桶容量（突发上限）
  rate_limit_refill_per_sec = 0.5  # 每秒补充令牌数（持续速率）
"""

import time
from typing import Dict

from junjun_core.config import get_global_config


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = capacity
        self.refill = refill_per_sec
        self.tokens = capacity
        self.updated = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_buckets: Dict[str, TokenBucket] = {}


def allow_message(chat_id: str) -> bool:
    """该会话这条消息是否放行（超速率丢弃）。"""
    gw = get_global_config().raw.get("gateway", {})
    capacity = float(gw.get("rate_limit_capacity", 8))
    refill = float(gw.get("rate_limit_refill_per_sec", 0.5))
    bucket = _buckets.get(chat_id)
    if bucket is None or bucket.capacity != capacity or bucket.refill != refill:
        bucket = _buckets[chat_id] = TokenBucket(capacity, refill)
    return bucket.allow()


def reset() -> None:
    """仅供测试。"""
    _buckets.clear()
