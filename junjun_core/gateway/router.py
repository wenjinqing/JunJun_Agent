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

    async def start(self) -> None:
        self.server = MessageServer(host=self.host, port=self.port, enable_token=False)
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
        if not text:
            logger.debug("消息无文本内容，跳过")
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
