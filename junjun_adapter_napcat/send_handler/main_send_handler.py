"""发消息处理：网关回复 MessageBase -> NapCat OneBot API。"""

import json
import uuid
from typing import List

from maim_message import MessageBase, Seg

from ..logger import logger
from ..response_pool import get_response
from .nc_sending import nc_message_sender


class SendHandler:
    async def handle_message(self, raw_message_base_dict: dict) -> None:
        try:
            msg = MessageBase.from_dict(raw_message_base_dict)
        except Exception as e:
            logger.error(f"反序列化网关消息失败: {e}")
            return
        seg = msg.message_segment
        info = msg.message_info
        group_info = info.group_info
        platform = info.platform

        processed = await self._process_seg(seg)
        if not processed:
            logger.warning("无有效发送内容")
            return

        if group_info:
            params = {
                "group_id": int(group_info.group_id),
                "message": processed,
            }
            action = "send_group_msg"
        else:
            params = {
                "user_id": int(info.user_info.user_id),
                "message": processed,
            }
            action = "send_private_msg"

        resp = await nc_message_sender.send_message_to_napcat(action, params)
        if resp.get("status") == "ok":
            logger.info(f"已发送到 NapCat [{action}]")
        else:
            logger.warning(f"NapCat 发送失败: {resp}")

    async def _process_seg(self, seg: Seg) -> List[dict]:
        """Seg -> OneBot message 数组。"""
        payload: List[dict] = []
        if seg.type == "seglist" and isinstance(seg.data, list):
            for sub in seg.data:
                payload = self._process_one(sub, payload)
        else:
            payload = self._process_one(seg, payload)
        return payload

    def _process_one(self, seg: Seg, payload: List[dict]) -> List[dict]:
        if seg.type == "text":
            text = seg.data
            if text:
                payload.append({"type": "text", "data": {"text": text}})
        elif seg.type == "image":
            payload.append({"type": "image", "data": {"file": seg.data}})
        elif seg.type == "emoji":
            payload.append({"type": "face", "data": {"id": str(seg.data)}})
        elif seg.type == "reply":
            payload.append({"type": "reply", "data": {"id": str(seg.data)}})
        return payload


send_handler = SendHandler()
