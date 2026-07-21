"""君君消息数据契约。

基于 maim_message 的 MessageBase 协议，层间只传递 MessageBase 实例，
不重造数据类，避免与 Adapter 协议错位。
"""

from dataclasses import dataclass, field
from typing import List, Optional

from maim_message import (
    MessageBase,
    Seg,
    UserInfo,
    GroupInfo,
    BaseMessageInfo,
    FormatInfo,
)

ACCEPT_FORMAT = [
    "text", "image", "emoji", "reply", "voice", "command",
    "voiceurl", "music", "videourl", "file", "imageurl", "forward", "video",
]


@dataclass
class ReplySegment:
    """Agent 回复的单个片段（简化版，由 send_handler 转 Seg/OneBot）。

    支持的 type（data 一律 str，复杂负载用 JSON 字符串）：
      text   纯文本
      image  图片（http URL 或 base64://...）
      voice  语音（URL 或本地路径，OneBot record 段）
      video  视频（URL 或本地路径，OneBot video 段）
      at     @某人（data=QQ号）
      poke   戳一戳（data=QQ号；独立 action，不占消息段）
      music  音乐卡片（JSON: url/audio/title/content/image）
      forward 合并转发（JSON: OneBot node 列表；独立 action）
      emoji  QQ 原生表情（data=face id）
      reply  引用（data=message_id，一般由 ReplySet.reply_to_message_id 自动加）
    注意：发「文件」不走消息段——用 junjun_core.napcat_client.upload_group/private_file。
    """
    type: str
    data: str


@dataclass
class ReplySet:
    """Agent 一次决策产出的回复集合。"""
    platform: str = "qq"
    target_user_id: Optional[str] = None
    target_group_id: Optional[str] = None
    segments: List[ReplySegment] = field(default_factory=list)
    should_reply: bool = True
    reply_to_message_id: Optional[str] = None

    def to_message_base(self, bot_user_id: str = "") -> MessageBase:
        segs: List[Seg] = []
        if self.reply_to_message_id:
            segs.append(Seg(type="reply", data=self.reply_to_message_id))
        for s in self.segments:
            segs.append(Seg(type=s.type, data=s.data))
        if len(segs) > 1:
            submit_seg = Seg(type="seglist", data=segs)
        elif segs:
            submit_seg = segs[0]
        else:
            submit_seg = Seg(type="text", data="")

        user_info = UserInfo(
            platform=self.platform,
            user_id=self.target_user_id or "",
            user_nickname="",
            user_cardname=None,
        )
        group_info = None
        if self.target_group_id:
            group_info = GroupInfo(
                platform=self.platform,
                group_id=self.target_group_id,
                group_name="",
            )
        msg_info = BaseMessageInfo(
            platform=self.platform,
            message_id="",
            time=0.0,
            user_info=user_info,
            group_info=group_info,
            template_info=None,
            format_info=FormatInfo(content_format=["text", "image", "emoji"], accept_format=ACCEPT_FORMAT),
            additional_config={},
        )
        return MessageBase(message_info=msg_info, message_segment=submit_seg, raw_message=None)
