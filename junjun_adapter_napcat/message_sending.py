"""消息发送：Adapter -> 君君网关（maim_message client）。"""

from maim_message import Router, RouteConfig, TargetConfig

from .logger import logger
from .config import get_config


class MessageSending:
    def __init__(self):
        self.maibot_router: Router = None

    async def message_send(self, message_base) -> bool:
        try:
            ok = await self.maibot_router.send_message(message_base)
            return bool(ok)
        except Exception as e:
            logger.error(f"发送消息到网关失败: {e}")
            return False


message_send_instance = MessageSending()


async def build_router() -> Router:
    cfg = get_config().maibot_server
    route = RouteConfig(route_config={
        cfg.platform_name: TargetConfig(
            url=f"ws://{cfg.host}:{cfg.port}/ws",
            token=cfg.token or None,  # 与核心 .env 的 GATEWAY_TOKEN 一致；网关未启用鉴权时留空
        )
    })
    r = Router(route)
    return r
