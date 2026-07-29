"""NapCat 发送器：通过 WS 连接调用 OneBot API。

断线韧性（2026-07-29）：NapCat 重启/网络抖动断连时，
发送方等待重连（默认 10s）而不是直接丢消息；连接关闭事件
驱动 connected Event，message_recv 断开时清连接。
"""

import asyncio
import json
import uuid

from ..logger import logger
from ..response_pool import get_response

_RECONNECT_WAIT = 10.0  # 断线后等待 NapCat 重连的秒数


class NCMessageSender:
    def __init__(self):
        self.server_connection = None
        self._connected = asyncio.Event()

    async def set_server_connection(self, conn) -> None:
        self.server_connection = conn
        self._connected.set()

    def clear_server_connection(self) -> None:
        self.server_connection = None
        self._connected.clear()

    async def wait_connected(self, timeout: float = None) -> bool:
        """等待 NapCat 重连；已连接立即返回 True。"""
        if self.server_connection is not None:
            return True
        if timeout is None:
            timeout = _RECONNECT_WAIT
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            return False

    async def send_message_to_napcat(self, action: str, params: dict) -> dict:
        if self.server_connection is None:
            # 短暂掉线：等重连再发，不直接丢消息
            if not await self.wait_connected():
                logger.error("NapCat 连接未建立（等待重连超时）")
                return {"status": "error", "message": "no connection"}
        request_uuid = str(uuid.uuid4())
        payload = json.dumps({"action": action, "params": params, "echo": request_uuid})
        try:
            await self.server_connection.send(payload)
        except Exception as e:
            logger.error(f"发送失败（连接已断开）: {type(e).__name__}: {e}")
            return {"status": "error", "message": str(e)}
        try:
            return await get_response(request_uuid, timeout=15)
        except TimeoutError:
            logger.error("等待 NapCat 响应超时")
            return {"status": "error", "message": "timeout"}
        except Exception as e:
            logger.error(f"发送失败: {e}")
            return {"status": "error", "message": str(e)}


nc_message_sender = NCMessageSender()
