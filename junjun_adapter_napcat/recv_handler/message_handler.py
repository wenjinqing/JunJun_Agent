"""收消息处理：NapCat OneBot 事件 -> maim_message MessageBase -> 发往君君网关。"""

import time

from maim_message import (
    UserInfo, GroupInfo, Seg, BaseMessageInfo, MessageBase, FormatInfo,
)

from ..config import get_config
from ..message_sending import message_send_instance

ACCEPT_FORMAT = ["text", "image", "emoji", "reply", "voice"]


class MessageHandler:
    def __init__(self):
        self.server_connection = None

    async def set_server_connection(self, conn) -> None:
        self.server_connection = conn

    async def check_allow_to_chat(self, user_id, group_id=None) -> bool:
        cfg = get_config().chat
        uid = int(user_id) if user_id is not None else None
        gid = int(group_id) if group_id is not None else None
        if gid is not None:
            if cfg.group_list_type == "whitelist" and gid not in cfg.group_list:
                return False
            if cfg.group_list_type == "blacklist" and gid in cfg.group_list:
                return False
        elif uid is not None:
            if cfg.private_list_type == "whitelist" and uid not in cfg.private_list:
                return False
            if cfg.private_list_type == "blacklist" and uid in cfg.private_list:
                return False
        if uid is not None and uid in cfg.ban_user_id:
            return False
        return True

    async def handle_raw_message(self, raw_message: dict) -> None:
        message_type = raw_message.get("message_type")
        message_id = raw_message.get("message_id")
        message_time = time.time()
        platform = get_config().maibot_server.platform_name

        if message_type == "private":
            sender = raw_message.get("sender", {})
            if not await self.check_allow_to_chat(sender.get("user_id"), None):
                return
            user_info = UserInfo(
                platform=platform,
                user_id=str(sender.get("user_id")),
                user_nickname=sender.get("nickname", ""),
                user_cardname=sender.get("card"),
            )
            group_info = None
        elif message_type == "group":
            sender = raw_message.get("sender", {})
            if not await self.check_allow_to_chat(sender.get("user_id"), raw_message.get("group_id")):
                return
            user_info = UserInfo(
                platform=platform,
                user_id=str(sender.get("user_id")),
                user_nickname=sender.get("nickname", ""),
                user_cardname=sender.get("card"),
            )
            group_info = GroupInfo(
                platform=platform,
                group_id=str(raw_message.get("group_id")),
                group_name="",
            )
        else:
            return

        seg_list, at_bot = self._parse_message_segments(
            raw_message.get("message", []),
            self_id=str(raw_message.get("self_id", "")),
        )
        if not seg_list:
            return

        submit_seg = Seg(type="seglist", data=seg_list) if len(seg_list) > 1 else seg_list[0]
        msg_info = BaseMessageInfo(
            platform=platform,
            message_id=str(message_id),
            time=message_time,
            user_info=user_info,
            group_info=group_info,
            template_info=None,
            format_info=FormatInfo(content_format=["text", "image", "emoji"], accept_format=ACCEPT_FORMAT),
            additional_config={"at_bot": at_bot},  # 供网关 L1 规则门 @ 旁路（对齐原 adapter 语义）
        )
        msg_base = MessageBase(
            message_info=msg_info,
            message_segment=submit_seg,
            raw_message=raw_message.get("raw_message"),
        )
        await message_send_instance.message_send(msg_base)

    def _parse_message_segments(self, real_message: list, self_id: str = "") -> tuple:
        """解析 OneBot array 消息段为 Seg 列表（阶段 1 只处理 text/at/image）。

        返回 (segs, at_bot)：at_bot 表示消息中 @ 了 bot 自己。
        """
        segs = []
        at_bot = False
        for sub in real_message or []:
            t = sub.get("type")
            d = sub.get("data", {})
            if t == "text":
                segs.append(Seg(type="text", data=d.get("text", "")))
            elif t == "at":
                # @ 信息转成 text 提示
                qq = str(d.get("qq", ""))
                if self_id and qq == self_id:
                    at_bot = True
                segs.append(Seg(type="text", data=f"@{qq} "))
            elif t == "image":
                segs.append(Seg(type="image", data=d.get("url", "")))
            elif t == "face":
                segs.append(Seg(type="emoji", data=str(d.get("id", ""))))
            elif t == "reply":
                segs.append(Seg(type="reply", data=str(d.get("id", ""))))
        return segs, at_bot


message_handler = MessageHandler()
