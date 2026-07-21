"""notice 事件处理：戳一戳（notify/poke）入站。

对齐总目标功能清单 #1「戳一戳」：群友/私聊戳 bot -> 合成一条 addressed 文本消息
进正常决策链（L1 @ 旁路必回），由 persona 决定如何俏皮回应，而非硬编码回复。
"""

import time

from maim_message import (
    UserInfo, GroupInfo, Seg, BaseMessageInfo, MessageBase, FormatInfo,
)

from ..config import get_config
from ..logger import logger
from ..message_sending import message_send_instance


class NoticeHandler:
    async def handle_notice(self, raw: dict) -> None:
        notice_type = raw.get("notice_type")
        if notice_type == "notify" and raw.get("sub_type") == "poke":
            await self._handle_poke(raw)

    async def _handle_poke(self, raw: dict) -> None:
        self_id = str(raw.get("self_id", ""))
        target_id = str(raw.get("target_id", ""))
        user_id = str(raw.get("user_id", ""))
        group_id = raw.get("group_id")
        # 只响应「戳的是 bot 自己」（群友互戳不打扰）
        if not self_id or target_id != self_id or not user_id or user_id == self_id:
            return
        if not await message_handler_allow(user_id, group_id):
            return

        platform = get_config().maibot_server.platform_name
        user_info = UserInfo(platform=platform, user_id=user_id, user_nickname="", user_cardname=None)
        group_info = (
            GroupInfo(platform=platform, group_id=str(group_id), group_name="") if group_id else None
        )
        msg_info = BaseMessageInfo(
            platform=platform,
            message_id=f"poke-{user_id}-{int(time.time())}",
            time=time.time(),
            user_info=user_info,
            group_info=group_info,
            template_info=None,
            format_info=FormatInfo(content_format=["text"], accept_format=["text"]),
            additional_config={"at_bot": True},  # 戳一戳 = 直呼，走 L1 @ 旁路
        )
        msg_base = MessageBase(
            message_info=msg_info,
            message_segment=Seg(type="text", data="（戳了戳你）"),
            raw_message="（戳了戳你）",
        )
        logger.info(f"收到戳一戳 [user={user_id} group={group_id}]，转决策链")
        await message_send_instance.message_send(msg_base)


async def message_handler_allow(user_id: str, group_id) -> bool:
    """复用 message_handler 的黑白名单判定。"""
    from .message_handler import message_handler
    return await message_handler.check_allow_to_chat(user_id, group_id)


notice_handler = NoticeHandler()
