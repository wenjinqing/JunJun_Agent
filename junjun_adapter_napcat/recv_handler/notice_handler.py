"""notice 事件处理：戳一戳（notify/poke）入站。

群聊（2026-08-13 用户裁决）：戳一戳一律【不进决策链】——每条戳都过 LLM
token 消耗巨大，群里还有别的 bot 时会互戳滚雪球。同一人在同群每天前
poke_group_daily_replies 次（默认 3）直接廉价回敬：反戳回去或发张库存
表情包（各半随机，2026-08-13 起不再发内置小黄豆），adapter 本地直发
NapCat，0 token；当日额度用完直接无视。

私聊维持原样：合成一条 addressed 文本消息进正常决策链（L1 @ 旁路必回）——
量小，且私聊戳是亲昵行为，值得 persona 认真回应。

防刷屏（2026-08-03，群聊私聊都生效）：三层抑制——
1. 同人同事窗口内只放一条（poke_min_interval，默认 60s）；
2. 会话地板：全会话最多每条 poke_chat_min_interval（默认 20s）放一条，
   防多人同时戳把回复刷爆；
3. 连戳升级：同一人被连续抑制 poke_escalate_count（默认 5）次后放一条，
   然后进入更长冷却（poke_escalate_cooldown，默认 600s）。
"""

import random
import time
from pathlib import Path

from maim_message import (
    UserInfo, Seg, BaseMessageInfo, MessageBase, FormatInfo,
)

from ..config import get_config
from ..logger import logger

# (chat_key, user_id) -> ts / 计数 / 是否已升级过；chat_key -> ts
_last: dict = {}
_suppressed: dict = {}
_escalated: dict = {}
_chat_last: dict = {}
# (chat_key, user_id) -> (日期串, 当日已回敬次数)：群戳廉价回敬的日额度
_group_daily: dict = {}

_POKE_TEXT = "（戳了戳你）"
_POKE_ESCALATE_TEXT = "（连续戳了你好几下）"

# 戳一戳回敬用的表情包库存（2026-08-13 用户裁决：回敬发库存表情包，
# 不再发 QQ 内置小黄豆 emoji）。与 bot 核心 express.emoji 的注册池同一目录；
# adapter 是独立进程，不碰 peewee/DB（守则：生产库只读走 ro 连接），
# 直接扫目录随机抽一张，0 token 语义不变。
_EMOJI_REG_DIR = Path(__file__).resolve().parents[2] / "data" / "emoji_registed"
_STICKER_EXTS = (".jpg", ".jpeg", ".gif", ".png", ".webp")


def _pick_sticker() -> "Path | None":
    """注册池随机抽一张表情包；目录空/不存在返回 None（调用方兜底反戳）。"""
    try:
        files = [p for p in _EMOJI_REG_DIR.iterdir()
                 if p.suffix.lower() in _STICKER_EXTS]
    except OSError:
        return None
    return random.choice(files) if files else None


def _poke_cfg() -> tuple:
    """(同人最小间隔, 会话地板, 升级阈值, 升级后冷却, 群戳日回敬额度)。配置缺失用默认值。"""
    chat = getattr(get_config(), "chat", None)
    return (
        int(getattr(chat, "poke_min_interval", 60) or 60),
        int(getattr(chat, "poke_chat_min_interval", 20) or 20),
        int(getattr(chat, "poke_escalate_count", 5) or 5),
        int(getattr(chat, "poke_escalate_cooldown", 600) or 600),
        int(getattr(chat, "poke_group_daily_replies", 3) or 3),
    )


def _gc_state(now: float) -> None:
    """状态表防膨胀：超 2000 条时清掉一天没动的。"""
    if len(_last) <= 2000:
        return
    cutoff = now - 86400
    for d in (_last, _suppressed, _escalated):
        for k in [k for k, v in d.items() if (v if isinstance(v, float) else 0) < cutoff]:
            d.pop(k, None)
    for k in [k for k, (day, _n) in _group_daily.items()
              if day != time.strftime("%Y-%m-%d", time.localtime(cutoff))]:
        _group_daily.pop(k, None)


def _reset_for_test() -> None:
    _last.clear()
    _suppressed.clear()
    _escalated.clear()
    _chat_last.clear()
    _group_daily.clear()


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

        if group_id:
            # 群聊：廉价回敬（反戳/表情），不进决策链（0 token）
            if not self._consume_group_budget(user_id, group_id):
                logger.info(f"戳一戳当日回敬额度用完，无视 [user={user_id} group={group_id}]")
                return
            await self._cheap_reply(user_id, str(group_id))
            return

        # 私聊：合成 addressed 文本进正常决策链
        platform = get_config().junjun_server.platform_name
        user_info = UserInfo(platform=platform, user_id=user_id, user_nickname="", user_cardname=None)
        msg_info = BaseMessageInfo(
            platform=platform,
            message_id=f"poke-{user_id}-{int(time.time())}",
            time=time.time(),
            user_info=user_info,
            group_info=None,
            template_info=None,
            format_info=FormatInfo(content_format=["text"], accept_format=["text"]),
            additional_config={"at_bot": True},  # 戳一戳 = 直呼，走 L1 @ 旁路
        )
        msg_base = MessageBase(
            message_info=msg_info,
            message_segment=Seg(type="text", data=text),
            raw_message=text,
        )
        logger.info(f"收到私聊戳一戳 [user={user_id}]，转决策链")
        # adapter 是独立进程，必须和普通消息一样走 WS 发给核心网关——
        # 直接调本进程的 gateway 只会拿到 echo 占位处理器，poke 会被静默丢弃
        from ..message_sending import message_send_instance
        await message_send_instance.message_send(msg_base)

    @staticmethod
    def _consume_group_budget(user_id: str, group_id) -> bool:
        """群戳日额度：同群同人每天最多 N 次廉价回敬。True=本次放行（已计数）。"""
        budget = _poke_cfg()[4]
        day = time.strftime("%Y-%m-%d")
        key = (f"g:{group_id}", user_id)
        d, n = _group_daily.get(key, ("", 0))
        if d != day:
            d, n = day, 0
        if n >= budget:
            return False
        _group_daily[key] = (d, n + 1)
        return True

    @classmethod
    async def _poke_back(cls, user_id: str, group_id: str) -> bool:
        """反戳回去（新旧 action 兜底）。True=成功。"""
        from ..send_handler.nc_sending import nc_message_sender
        for action, params in (
                ("send_poke", {"user_id": int(user_id), "group_id": int(group_id)}),
                ("send_group_poke", {"group_id": int(group_id),
                                     "user_id": int(user_id)})):
            try:
                resp = await nc_message_sender.send_message_to_napcat(action, params)
            except Exception as e:
                logger.warning(f"戳一戳回敬异常 [{action}]: {e}")
                continue
            if resp.get("status") == "ok":
                logger.info(f"戳一戳已回敬 [{action} user={user_id} group={group_id}]")
                return True
            logger.warning(f"戳一戳回敬 [{action}] 失败: {resp}")
        return False

    @classmethod
    async def _send_sticker(cls, user_id: str, group_id: str) -> bool:
        """发张库存表情包（本地 file:// URI，NapCat 同机直读）。True=成功。"""
        sticker = _pick_sticker()
        if sticker is None:
            return False
        from ..send_handler.nc_sending import nc_message_sender
        try:
            resp = await nc_message_sender.send_message_to_napcat(
                "send_group_msg",
                {"group_id": int(group_id),
                 "message": [{"type": "image",
                              "data": {"file": sticker.as_uri()}}]})
        except Exception as e:
            logger.warning(f"戳一戳回表情包异常: {e}")
            return False
        if resp.get("status") == "ok":
            logger.info(f"戳一戳回表情包 [{sticker.name} user={user_id} group={group_id}]")
            return True
        logger.warning(f"戳一戳回表情包失败: {resp}")
        return False

    @classmethod
    async def _cheap_reply(cls, user_id: str, group_id: str) -> None:
        """廉价回敬：反戳 / 库存表情包各半随机，首选失败兜底另一种——
        adapter 本地直发 NapCat，0 token。都失败就静默（下次再戳再说）。"""
        first, second = (cls._poke_back, cls._send_sticker) \
            if random.random() < 0.5 else (cls._send_sticker, cls._poke_back)
        if not await first(user_id, group_id):
            await second(user_id, group_id)

    @staticmethod
    def _throttle(user_id: str, group_id) -> "str | None":
        """防刷屏闸门。返回要合成/放行的文本，None = 本条抑制。"""
        now = time.time()
        _gc_state(now)
        min_interval, chat_floor, escalate_n, esc_cooldown, _budget = _poke_cfg()
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
