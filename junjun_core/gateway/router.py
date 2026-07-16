"""君君消息网关：基于 maim_message.MessageServer。

职责：
1. 作为 WS server 监听（Adapter 作为 client 连入）。
2. 收到 MessageBase 后：黑白名单过滤 -> 会话登记 -> 调 Agent（阶段 1 echo 占位）。
3. 把 Agent 产出的 ReplySet 转回 MessageBase 并广播给 Adapter。
"""

import asyncio
import time
from typing import Optional

from maim_message import MessageServer, MessageBase, Seg

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger
from junjun_core.contracts import ReplySet, ReplySegment
from junjun_core.gateway.blacklist import ChatListConfig
from junjun_core.gateway.session_manager import get_session_manager

logger = get_logger("gateway")


class Gateway:
    def __init__(self, host: str = "127.0.0.1", port: int = 8092, bot_user_id: str = ""):
        self.host = host
        self.port = port
        self.bot_user_id = bot_user_id
        self.server: Optional[MessageServer] = None
        self._server_task: Optional[asyncio.Task] = None
        self._chat_list: Optional[ChatListConfig] = None

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
        seg = msg.message_segment
        group_info = info.group_info
        user_info = info.user_info
        group_id = group_info.group_id if group_info else None
        user_id = user_info.user_id if user_info else None

        if not self._get_chat_list().allow(user_id, group_id):
            logger.debug(f"消息被名单过滤: user={user_id} group={group_id}")
            return

        text = _extract_text(seg)
        if not text:
            logger.debug("消息无文本内容，跳过")
            return

        session = get_session_manager().get_or_create(msg)
        session.add_message(text)
        logger.info(f"收到消息 [{session.chat_id}] {user_info.user_nickname}: {text[:80]}")

        reply = ReplySet(
            platform=info.platform,
            target_user_id=user_id if group_id is None else None,
            target_group_id=str(group_id) if group_id else None,
            segments=[ReplySegment(type="text", data=f"[echo] {text}")],
            should_reply=True,
        )
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
