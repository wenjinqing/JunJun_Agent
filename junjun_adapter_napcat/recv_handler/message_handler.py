"""收消息处理：NapCat OneBot 事件 -> maim_message MessageBase -> 发往君君网关。"""

import time

from maim_message import (
    UserInfo, GroupInfo, Seg, BaseMessageInfo, MessageBase, FormatInfo,
)

from ..config import get_config
from ..message_sending import message_send_instance

ACCEPT_FORMAT = ["text", "image", "emoji", "reply", "voice"]

# @ 昵称解析缓存：{(group_id, qq): (nickname, ts)}，TTL 1 小时
_NICK_CACHE: dict = {}
_NICK_TTL = 3600.0
_NICK_CACHE_MAX = 5000  # 硬上限：满了先清过期项，还满丢最旧（防长跑无界增长）


def _nick_cache_put(key, name: str) -> None:
    now = time.time()
    if len(_NICK_CACHE) >= _NICK_CACHE_MAX:
        expired = [k for k, (_, ts) in _NICK_CACHE.items() if now - ts >= _NICK_TTL]
        for k in expired:
            _NICK_CACHE.pop(k, None)
        while len(_NICK_CACHE) >= _NICK_CACHE_MAX:
            _NICK_CACHE.pop(next(iter(_NICK_CACHE)), None)  # 还满就丢最旧
    _NICK_CACHE[key] = (name, now)


async def _resolve_nickname(qq: str, group_id: str) -> str:
    """@ 目标 QQ -> 昵称。群成员 API（card 优先，缓存 1h）-> Messages 历史昵称 -> 某人。"""
    key = (str(group_id), str(qq))
    hit = _NICK_CACHE.get(key)
    if hit and time.time() - hit[1] < _NICK_TTL:
        return hit[0]
    if group_id:
        try:
            from ..send_handler.nc_sending import nc_message_sender
            resp = await nc_message_sender.send_message_to_napcat(
                "get_group_member_info",
                {"group_id": int(group_id), "user_id": int(qq), "no_cache": False},
            )
            data = resp.get("data") or {}
            name = (data.get("card") or data.get("nickname") or "").strip()
            if name:
                _nick_cache_put(key, name)
                return name
        except Exception:
            pass
    # 降级：历史消息里见过的昵称（跨群转发/私聊/API 失败兜底）
    try:
        from junjun_core.database import Messages
        row = (Messages.select(Messages.user_nickname)
               .where((Messages.user_id == str(qq)) & (Messages.user_nickname != ""))
               .order_by(Messages.time.desc()).first())
        if row and row.user_nickname:
            _NICK_CACHE[key] = (row.user_nickname, time.time())
            return row.user_nickname
    except Exception:
        pass
    return "某人"


async def _resolve_reply(reply_id: str) -> str:
    """引用消息 id -> 「[回复 昵称: 内容]」文本；失败降级占位。内容截断 200 字。"""
    try:
        from ..send_handler.nc_sending import nc_message_sender
        resp = await nc_message_sender.send_message_to_napcat(
            "get_msg", {"message_id": int(reply_id)})
        data = resp.get("data") or {}
        if not data:
            return "[回复某条消息]"
        nickname = ((data.get("sender") or {}).get("card")
                    or (data.get("sender") or {}).get("nickname") or "某人")
        text = await _plain_text_of(data.get("message") or data.get("raw_message") or "",
                                    group_id=str(data.get("group_id") or ""))
        if not text:
            return f"[回复 {nickname} 的消息]"
        if len(text) > 200:
            text = text[:200] + "…"
        return f"[回复 {nickname}: {text}]"
    except Exception:
        return "[回复某条消息]"


async def _plain_text_of(message, group_id: str = "") -> str:
    """从 OneBot 消息（array 或 string）提取纯文本（图片/表情转占位，@ 解析昵称）。"""
    if isinstance(message, str):
        return message.strip()
    parts = []
    for seg in message if isinstance(message, list) else []:
        t, d = seg.get("type"), seg.get("data", {})
        if t == "text":
            parts.append(d.get("text", ""))
        elif t == "image":
            parts.append("[图片]")
        elif t == "face":
            parts.append("[表情]")
        elif t == "at":
            name = await _resolve_nickname(str(d.get("qq", "")), group_id)
            parts.append(f"@{name} ")
    return "".join(parts).strip()


class MessageHandler:
    def __init__(self):
        self.server_connection = None

    async def set_server_connection(self, conn) -> None:
        self.server_connection = conn

    async def check_allow_to_chat(self, user_id, group_id=None) -> bool:
        cfg = get_config().chat
        # 畸形 id（非数字）按拒绝处理，不让 ValueError 炸到消费循环
        try:
            uid = int(user_id) if user_id is not None else None
            gid = int(group_id) if group_id is not None else None
        except (TypeError, ValueError):
            logger.warning(f"黑白名单判定收到畸形 id: user={user_id} group={group_id}，按拒绝处理")
            return False
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
        platform = get_config().junjun_server.platform_name

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

        seg_list, at_bot = await self._parse_message_segments(
            raw_message.get("message", []),
            self_id=str(raw_message.get("self_id", "")),
            group_id=str(raw_message.get("group_id", "") or ""),
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

    async def _parse_message_segments(self, real_message: list, self_id: str = "",
                                      group_id: str = "") -> tuple:
        """解析 OneBot array 消息段为 Seg 列表。

        返回 (segs, at_bot)：at_bot 表示消息中 @ 了 bot 自己。
        合并转发（forward）经 get_forward_msg 展开为文本（递归深度限 2，截断 500 字）。
        @ 解析为真实群昵称（缓存 1h，查不到降级 @某人）；@bot 显示为 @你。
        引用消息经 get_msg 展开为「[回复 昵称: 内容]」（截断 200 字，失败降级占位）。
        """
        segs = []
        at_bot = False
        for sub in real_message or []:
            t = sub.get("type")
            d = sub.get("data", {})
            if t == "text":
                segs.append(Seg(type="text", data=d.get("text", "")))
            elif t == "at":
                # @ 解析为真实昵称（防 QQ 号误判断，同时保留指向性）
                qq = str(d.get("qq", ""))
                if self_id and qq == self_id:
                    at_bot = True
                    segs.append(Seg(type="text", data="@你 "))
                else:
                    name = await _resolve_nickname(qq, group_id)
                    segs.append(Seg(type="text", data=f"@{name} "))
            elif t == "image":
                # sub_type=1 是收藏表情/贴纸（可偷），0 是普通图片（不偷，只给 VLM 看）
                if str(d.get("sub_type", "0")) == "1":
                    segs.append(Seg(type="sticker", data=d.get("url", "")))
                else:
                    segs.append(Seg(type="image", data=d.get("url", "")))
            elif t == "mface":
                # 商城表情：也算表情包
                segs.append(Seg(type="sticker", data=d.get("url", "")))
            elif t == "face":
                segs.append(Seg(type="emoji", data=str(d.get("id", ""))))
            elif t == "reply":
                # 引用消息展开为可读文本（君君需要看到被回复的内容才能接话）
                segs.append(Seg(type="text", data=await _resolve_reply(str(d.get("id", "")))))
            elif t == "forward":
                segs.append(Seg(type="text", data=await self._expand_forward(d)))
            elif t == "video":
                # 视频段：转占位文本（VLM 暂不支持视频，让 Agent 知道有视频）
                segs.append(Seg(type="text", data="[视频]"))
            elif t == "record":
                # 语音段：文本占位（保上下文形态）+ voice 段（转写管线用，url 优先其次 file id）
                segs.append(Seg(type="text", data="[语音]"))
                ref = str(d.get("url") or d.get("file") or "")
                if ref:
                    segs.append(Seg(type="voice", data=ref))
            elif t == "file":
                # 文件段：转占位文本
                segs.append(Seg(type="text", data="[文件]"))
        return segs, at_bot

    async def _expand_forward(self, data: dict, depth: int = 1) -> str:
        """展开合并转发消息为可读文本。失败降级占位，不阻塞主链路。"""
        if depth > 2:
            return "[嵌套合并转发]"
        forward_id = data.get("id")
        if not forward_id:
            return "[合并转发消息]"
        try:
            from ..send_handler.nc_sending import nc_message_sender
            resp = await nc_message_sender.send_message_to_napcat("get_forward_msg", {"id": str(forward_id)})
            nodes = (resp.get("data") or {}).get("message") or (resp.get("data") or {}).get("messages") or []
            if not nodes:
                return "[合并转发消息]"
            parts = []
            total = 0
            for node in nodes:
                nickname = (node.get("sender") or {}).get("nickname", "??")
                texts = []
                for seg in node.get("message", []) or []:
                    st, sd = seg.get("type"), seg.get("data", {})
                    if st == "text":
                        texts.append(sd.get("text", ""))
                    elif st == "image":
                        texts.append("[图片]")
                    elif st == "at":
                        name = await _resolve_nickname(str(sd.get("qq", "")), "")
                        texts.append(f"@{name} ")
                    elif st == "forward":
                        texts.append(await self._expand_forward(sd, depth + 1))
                line = f"{nickname}: {''.join(texts).strip()}"
                budget = 500 - total  # 防炸上下文（对齐踩坑清单：展开文本截断 500 字）
                if len(line) > budget:
                    if budget > 0:
                        parts.append(line[:budget])
                    parts.append("……（转发内容过长已截断）")
                    break
                parts.append(line)
                total += len(line)
            return "[合并转发]\n" + "\n".join(parts)
        except Exception as e:
            from ..logger import logger
            logger.warning(f"合并转发展开失败（降级占位）: {e}")
            return "[合并转发消息]"


message_handler = MessageHandler()
