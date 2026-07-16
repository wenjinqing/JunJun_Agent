"""通信层：Adapter 作为 maim_message client 连接君君网关，并接收网关回复。"""

from maim_message import Router

from .logger import logger
from .config import get_config
from .message_sending import build_router, message_send_instance
from .send_handler import send_handler


async def mmc_start_com() -> Router:
    router = await build_router()
    router.register_class_handler(send_handler.handle_message)
    message_send_instance.maibot_router = router
    logger.info(f"已连接君君网关 ws://{get_config().maibot_server.host}:{get_config().maibot_server.port}/ws")
    await router.run()


async def mmc_stop_com(router: Router) -> None:
    await router.stop()
