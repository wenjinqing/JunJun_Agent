"""notice 事件处理：戳一戳（notify/poke）入站。

对齐总目标功能清单 #1「戳一戳」：群友/私聊戳 bot -> 合成一条 addressed 文本消息
进正常决策链（L1 @ 旁路必回），由 persona 决定如何俏皮回应，而非硬编码回复。

防刷屏（2026-08-03）：被连戳时每条都回太烦人，也不像真人。三层抑制：
1. 同人同事窗口内只放一条（poke_min_interval，默认 60s）；
2. 会话地板：全会话最多每条 poke_chat_min_interval（默认 20s）放一条，
   防多人同时戳把回复刷爆；
3. 连戳升级：同一人被连续抑制 poke_escalate_count（默认 5）次后，放一条
   「（连续戳了你好几下）」让她名正言顺地吐槽一次，然后进入更长冷却
   （poke_escalate_cooldown，默认 600s）——像真人：先不理，烦了就哼一声。
"""

import time

from maim_message import (
    UserInfo, GroupInfo, Seg, BaseMessageInfo, MessageBase, FormatInfo,
)

from ..config import get_config
from ..logger import logger

# (chat_key, user_id) -> ts / 计数 / 是否已升级过；chat_key -> ts
_last: dict = {}
_suppressed: dict = {}
_escalated: dict = {}
_chat_last: dict = {}

_POKE_TEXT = "（戳了戳你）"
_POKE_ESCALATE_TEXT = "（连续戳了你好几下）"


def _poke_cfg() -> tuple:
    """(同人最小间隔, 会话地板, 升级阈值, 升级后冷却)。配置缺失时用默认值。"""
    chat = getattr(get_config(), "chat", None)
    return (
        int(getattr(chat, "poke_min_interval", 60) or 60),
        int(getattr(chat, "poke_chat_min_interval", 20) or 20),
        int(getattr(chat, "poke_escalate_count", 5) or 5),
        int(getattr(chat, "poke_escalate_cooldown", 600) or 600),
    )


def _gc_state(now: float) -> None:
    """状态表防膨胀：超 2000 条时清掉一天没动的。"""
    if len(_last) <= 2000:
        return
    cutoff = now - 86400
    for d in (_last, _suppressed, _escalated):
        for k in [k for k, v in d.items() if (v if isinstance(v, float) else 0) < cutoff]:
            d.pop(k, None)


def _reset_for_test() -> None:
    _last.clear()
    _suppressed.clear()
    _escalated.clear()
    _chat_last.clear()


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

        text = self._throttle(user_id, group_id)
        if text is None:
            return  # 被防抖抑制（连戳刷屏）

        platform = get_config().junjun_server.platform_name
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
            message_segment=Seg(type="text", data=text),
            raw_message=text,
        )
        logger.info(f"收到戳一戳 [user={user_id} group={group_id}]，转决策链")
        # adapter 是独立进程，必须和普通消息一样走 WS 发给核心网关——
        # 直接调本进程的 gateway 只会拿到 echo 占位处理器，poke 会被静默丢弃
        from ..message_sending import message_send_instance
        await message_send_instance.message_send(msg_base)

    @staticmethod
    def _throttle(user_id: str, group_id) -> "str | None":
        """防刷屏闸门。返回要合成的文本，None = 本条抑制。"""
        now = time.time()
        _gc_state(now)
        min_interval, chat_floor, escalate_n, esc_cooldown = _poke_cfg()
        chat_key = f"g:{group_id}" if group_id else f"u:{user_id}"
        key = (chat_key, user_id)
        interval = esc_cooldown if _escalated.get(key) else min_interval
        within_user = now - _last.get(key, 0.0) < interval
        within_chat = now - _chat_last.get(chat_key, 0.0) < chat_floor

        if not within_user and not within_chat:
            _last[key] = _chat_last[chat_key] = now
            _suppressed[key] = 0
            _escalated[key] = False
            return _POKE_TEXT

        # 只有「同一人连戳」计入升级计数；撞会话地板的只是顺被带挡
        if within_user:
            _suppressed[key] = _suppressed.get(key, 0) + 1
            if _suppressed[key] >= escalate_n and not _escalated.get(key):
                _last[key] = _chat_last[chat_key] = now
                _suppressed[key] = 0
                _escalated[key] = True
                logger.info(f"戳一戳连刷 {escalate_n} 次 [user={user_id} group={group_id}]，放一条吐槽")
                return _POKE_ESCALATE_TEXT
        logger.debug(f"戳一戳被防抖抑制 [user={user_id} group={group_id}]")
        return None


async def message_handler_allow(user_id: str, group_id) -> bool:
    """复用 message_handler 的黑白名单判定。"""
    from .message_handler import message_handler
    return await message_handler.check_allow_to_chat(user_id, group_id)


notice_handler = NoticeHandler()
