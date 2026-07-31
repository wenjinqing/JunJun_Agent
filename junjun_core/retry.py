"""瞬态失败重试助手：指数退避，默认 3 次，都失败抛原异常（调用方走原有降级）。

适用：网络抖动/限流/连接重置类瞬态错误（MCP 调用、VLM、HTTP 抓取）。
不适用：确定性错误（参数错误、权限不足、内容不存在）——重试只是浪费时间，
       所以 retry_on 默认排除 ValueError/KeyError/TypeError 等编程错误。
       TimeoutError 同样不重试：30s 超时已经发生，重试只会把阻塞放大 3 倍
       （MCP 挂死工具曾因此阻塞 agent 93s+）。
"""

import asyncio
from typing import Awaitable, Callable, Optional, Tuple, Type

from junjun_core.observability import get_logger

logger = get_logger("core.retry")

# 编程/确定性错误 + 超时：重试无意义，直接抛出
_NO_RETRY = (ValueError, KeyError, TypeError, AttributeError, TimeoutError)


async def retry_async(fn: Callable[[], Awaitable], *, attempts: int = 3,
                      base_delay: float = 0.5, label: str = "",
                      retry_on: Tuple[Type[BaseException], ...] = (Exception,)):
    """调用 fn()，失败按 base_delay*2^i 退避重试，attempts 次都失败抛最后一次异常。

    命中 _NO_RETRY（编程错误）时不重试直接抛。
    """
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return await fn()
        except _NO_RETRY:
            raise
        except retry_on as e:
            last = e
            if i < attempts - 1:
                delay = base_delay * (2 ** i)
                logger.debug(f"{label} 第 {i + 1}/{attempts} 次失败，{delay:.1f}s 后重试: "
                             f"{type(e).__name__}: {e}")
                await asyncio.sleep(delay)
    raise last  # type: ignore[misc]
