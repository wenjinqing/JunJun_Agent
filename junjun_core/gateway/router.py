"""君君消息网关：基于 maim_message.MessageServer。

职责：
1. 作为 WS server 监听（Adapter 作为 client 连入）。
2. 收到 MessageBase 后：黑白名单过滤 -> 会话登记 -> 交给 processor。
3. 把 processor 产出的 ReplySet 转回 MessageBase 并广播给 Adapter。

分层：processor 由上层（junjun_agent）注入，core 不 import 上层。
未注入时用 echo 占位（阶段 1 语义，测试可用）。
"""

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from maim_message import MessageServer, MessageBase, Seg

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger
from junjun_core.contracts import ReplySet, ReplySegment
from junjun_core.gateway.blacklist import ChatListConfig
from junjun_core.gateway.session_manager import ChatSession, get_session_manager

logger = get_logger("gateway")


@dataclass
class InboundMeta:
    """入站消息元信息（processor 入参）。"""
    text: str
    user_id: Optional[str]
    nickname: str
    group_id: Optional[str]
    message_id: str
    at_bot: bool
    is_self: bool
    image_urls: list = None      # 消息中的图片/表情包 URL（偷图用）
    has_emoji: bool = False      # 含 QQ 原生表情


# processor 签名: async (session, meta) -> Optional[ReplySet]
Processor = Callable[[ChatSession, InboundMeta], Awaitable[Optional[ReplySet]]]


async def _echo_processor(session: ChatSession, meta: InboundMeta) -> Optional[ReplySet]:
    """阶段 1 占位：原样复读。"""
    return ReplySet(
        platform=session.platform,
        target_user_id=meta.user_id if meta.group_id is None else None,
        target_group_id=meta.group_id,
        segments=[ReplySegment(type="text", data=f"[echo] {meta.text}")],
        should_reply=True,
    )


class Gateway:
    def __init__(self, host: str = "127.0.0.1", port: int = 8092, bot_user_id: str = ""):
        self.host = host
        self.port = port
        self.bot_user_id = bot_user_id
        self.server: Optional[MessageServer] = None
        self._server_task: Optional[asyncio.Task] = None
        self._chat_list: Optional[ChatListConfig] = None
        self._processor: Processor = _echo_processor

    def set_processor(self, processor: Processor) -> None:
        """由上层注入消息处理器（junjun_agent 的决策漏斗）。"""
        self._processor = processor
        logger.info(f"消息处理器已注入: {getattr(processor, '__name__', type(processor).__name__)}")

    def _get_chat_list(self) -> ChatListConfig:
        if self._chat_list is None:
            raw = get_global_config().raw.get("chat", {})
            self._chat_list = ChatListConfig.from_raw(raw)
        return self._chat_list

    # 只允许无鉴权监听本机回环；对外监听必须配置 GATEWAY_TOKEN
    _LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

    async def start(self) -> None:
        import os
        token = os.environ.get("GATEWAY_TOKEN", "").strip()
        if token:
            self.server = MessageServer(host=self.host, port=self.port, enable_token=True)
            self.server.add_valid_token(token)
            logger.info("网关鉴权：已启用 token 认证（GATEWAY_TOKEN）")
        else:
            if self.host not in self._LOCAL_HOSTS:
                raise RuntimeError(
                    f"网关监听 {self.host} 但未配置 GATEWAY_TOKEN——"
                    "任何能连到该地址的人都能伪造消息控制 bot，拒绝启动。"
                    "请在 .env 配置 GATEWAY_TOKEN，或把 [gateway] host 改回 127.0.0.1。"
                )
            self.server = MessageServer(host=self.host, port=self.port, enable_token=False)
            logger.warning("网关鉴权：未配置 GATEWAY_TOKEN，仅本机回环连接可用（建议配置）")
        self.server.register_message_handler(self.handle_inbound)
        logger.info(f"网关启动中 ws://{self.host}:{self.port}/ws （等待 Adapter 连接）")
        # start_server() 内部 await Event().wait() 永不返回，必须放后台任务
        self._server_task = asyncio.create_task(self.server.start_server(), name="gateway-server")
        await asyncio.sleep(0.5)  # 等底层 aiohttp site 起监听
        if self._server_task.done() and (exc := self._server_task.exception()):
            raise RuntimeError(f"网关启动失败: {exc}") from exc
        logger.info(f"网关 WS 已就绪 ws://{self.host}:{self.port}/ws")

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.stop()
        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("网关已关闭")

    async def handle_inbound(self, message_data: dict) -> None:
        try:
            msg = MessageBase.from_dict(message_data)
        except Exception as e:
            logger.warning(f"消息反序列化失败: {e}")
            return

        info = msg.message_info
        group_info = info.group_info
        user_info = info.user_info
        group_id = group_info.group_id if group_info else None
        user_id = user_info.user_id if user_info else None

        if not self._get_chat_list().allow(user_id, group_id):
            logger.debug(f"消息被名单过滤: user={user_id} group={group_id}")
            return

        text = _extract_text(msg.message_segment)
        image_urls = _extract_images(msg.message_segment)
        if not text and not image_urls:
            logger.debug("消息无文本/图片内容，跳过")
            return

        chat_id = f"{info.platform}:{group_id if group_info else user_id}:{'group' if group_info else 'private'}"
        from junjun_core.gateway.rate_limit import allow_message
        if not allow_message(chat_id):
            logger.debug(f"[{chat_id}] 触发速率限制，消息丢弃")
            return

        add_cfg = info.additional_config or {}
        meta = InboundMeta(
            text=text,
            user_id=str(user_id) if user_id is not None else None,
            nickname=(user_info.user_nickname or "") if user_info else "",
            group_id=str(group_id) if group_id is not None else None,
            message_id=str(info.message_id or ""),
            at_bot=bool(add_cfg.get("at_bot")),
            is_self=bool(self.bot_user_id and str(user_id) == str(self.bot_user_id)),
            image_urls=image_urls,
            has_emoji=_has_emoji(msg.message_segment),
        )

        session = get_session_manager().get_or_create(msg)
        logger.info(f"收到消息 [{session.chat_id}] {meta.nickname}: {text[:80]}")

        try:
            reply = await self._processor(session, meta)
        except Exception as e:
            logger.error(f"processor 异常，本条消息忽略: {type(e).__name__}: {e}")
            return
        if reply is not None:
            await self.send_reply(reply)

    async def send_reply(self, reply: ReplySet) -> None:
        if not reply.should_reply:
            return
        msg_base = reply.to_message_base(self.bot_user_id)
        if self.server is not None:
            await self.server.broadcast_message(msg_base.to_dict())
            logger.info(f"已发送回复 -> {reply.target_group_id or reply.target_user_id}")


def _extract_text(seg: Seg) -> str:
    parts = []
    if seg.type == "seglist" and isinstance(seg.data, list):
        for sub in seg.data:
            parts.append(_extract_text(sub))
    elif seg.type == "text" and isinstance(seg.data, str):
        parts.append(seg.data)
    return "".join(parts)


def _extract_images(seg: Seg) -> list:
    urls = []
    if seg.type == "seglist" and isinstance(seg.data, list):
        for sub in seg.data:
            urls.extend(_extract_images(sub))
    elif seg.type == "image" and isinstance(seg.data, str) and seg.data:
        urls.append(seg.data)
    return urls


def _has_emoji(seg: Seg) -> bool:
    if seg.type == "seglist" and isinstance(seg.data, list):
        return any(_has_emoji(sub) for sub in seg.data)
    return seg.type == "emoji"


_gateway: Optional[Gateway] = None


def get_gateway() -> Gateway:
    global _gateway
    if _gateway is None:
        cfg = get_global_config()
        host = cfg.raw.get("gateway", {}).get("host", "127.0.0.1")
        port = int(cfg.raw.get("gateway", {}).get("port", 8092))
        bot_id = cfg.bot.qq_account or ""
        _gateway = Gateway(host=host, port=port, bot_user_id=bot_id)
    return _gateway


def get_router() -> Gateway:
    return get_gateway()
