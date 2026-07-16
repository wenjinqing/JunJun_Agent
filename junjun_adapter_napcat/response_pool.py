"""NapCat 响应池：echo/id 请求-响应匹配。"""

import asyncio
import time
from typing import Dict

_response_dict: Dict = {}
_response_time_dict: Dict = {}


async def get_response(request_id: str, timeout: int = 10) -> dict:
    return await asyncio.wait_for(_get_response(request_id), timeout)


async def _get_response(request_id: str) -> dict:
    while request_id not in _response_dict:
        await asyncio.sleep(0.1)
    return _response_dict.pop(request_id)


async def put_response(response: dict) -> None:
    echo_id = response.get("echo")
    _response_dict[echo_id] = response
    _response_time_dict[echo_id] = time.time()


async def check_timeout_response() -> None:
    while True:
        now = time.time()
        for echo_id in list(_response_time_dict.keys()):
            if now - _response_time_dict[echo_id] > 60:
                _response_dict.pop(echo_id, None)
                _response_time_dict.pop(echo_id, None)
        await asyncio.sleep(30)
