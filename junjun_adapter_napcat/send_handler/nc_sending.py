"""NapCat 发送器：通过 WS 连接调用 OneBot API。"""

import json
import uuid
from typing import Optional

from ..logger import logger
from ..response_pool import get_response


class NCMessageSender:
    def __init__(self):
        self.server_connection = None

    async def set_server_connection(self, conn) -> None:
        self.server_connection = conn

    async def send_message_to_napcat(self, action: str, params: dict) -> dict:
        if self.server_connection is None:
            logger.error("NapCat 连接未建立")
            return {"status": "error", "message": "no connection"}
        request_uuid = str(uuid.uuid4())
        payload = json.dumps({"action": action, "params": params, "echo": request_uuid})
        await self.server_connection.send(payload)
        try:
            return await get_response(request_uuid, timeout=15)
        except TimeoutError:
            logger.error("等待 NapCat 响应超时")
            return {"status": "error", "message": "timeout"}
        except Exception as e:
            logger.error(f"发送失败: {e}")
            return {"status": "error", "message": str(e)}


nc_message_sender = NCMessageSender()
