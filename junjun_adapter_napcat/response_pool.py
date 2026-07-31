"""NapCat 响应池：echo/id 请求-响应匹配。

2026-07-31 重写：0.1s 轮询改 asyncio.Future——原实现每个 OneBot 调用
平均白等 50ms、最差 100ms+。响应先于等待注册（罕见竞态）走暂存兜底。
"""

import asyncio
import time
from typing import Dict, Tuple

_pending: Dict[str, asyncio.Future] = {}           # echo -> 等待中的 Future
_stash: Dict[str, Tuple[float, dict]] = {}         # 响应先于等待注册的暂存（60s TTL）


async def get_response(request_id: str, timeout: int = 10) -> dict:
    hit = _stash.pop(request_id, None)
    if hit is not None:
        return hit[1]
    fut = asyncio.get_running_loop().create_future()
    _pending[request_id] = fut
    try:
        return await asyncio.wait_for(asyncio.shield(fut), timeout)
    finally:
        _pending.pop(request_id, None)


async def put_response(response: dict) -> None:
    echo_id = response.get("echo")
    fut = _pending.get(echo_id)
    if fut is not None and not fut.done():
        fut.set_result(response)
    else:
        _stash[echo_id] = (time.time(), response)


async def check_timeout_response() -> None:
    """定期清理暂存的超时响应（等待方已放弃）。"""
    while True:
        now = time.time()
        for echo_id in list(_stash.keys()):
            if now - _stash[echo_id][0] > 60:
                _stash.pop(echo_id, None)
        await asyncio.sleep(30)
