"""君君 NapCat Adapter 入口。

架构：
- NapCat 作为 WS client 主动连入 Adapter 的 WS server（napcat_server.port，默认 8095）。
- Adapter 作为 maim_message client 连接君君网关（maibot_server.port，默认 8092）。
- 收：NapCat WS -> message_handler -> MessageBase -> 网关。
- 发：网关回复 -> send_handler -> NapCat OneBot API。
"""

import asyncio
import json
import http
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import websockets as Server

from junjun_adapter_napcat.logger import logger
from junjun_adapter_napcat.config import get_config
from junjun_adapter_napcat.recv_handler.message_handler import message_handler
from junjun_adapter_napcat.recv_handler.meta_event_handler import meta_event_handler
from junjun_adapter_napcat.recv_handler.notice_handler import notice_handler
from junjun_adapter_napcat.send_handler.nc_sending import nc_message_sender
from junjun_adapter_napcat.response_pool import put_response, check_timeout_response
from junjun_adapter_napcat.com_layer import mmc_start_com

message_queue = asyncio.Queue()


async def message_recv(server_connection: Server.ServerConnection):
    await message_handler.set_server_connection(server_connection)
    await nc_message_sender.set_server_connection(server_connection)
    async for raw_message in server_connection:
        try:
            decoded = json.loads(raw_message)
        except Exception as e:
            logger.warning(f"消息 JSON 解析失败: {e}")
            continue
        post_type = decoded.get("post_type")
        if post_type in ["meta_event", "message", "notice"]:
            await message_queue.put(decoded)
        elif post_type is None:
            await put_response(decoded)


async def message_process():
    while True:
        message = await message_queue.get()
        post_type = message.get("post_type")
        if post_type == "message":
            await message_handler.handle_raw_message(message)
        elif post_type == "meta_event":
            await meta_event_handler.handle_meta_event(message)
        elif post_type == "notice":
            await notice_handler.handle_notice(message)
        message_queue.task_done()
        await asyncio.sleep(0.05)


def check_napcat_server_token(conn, request):
    token = get_config().napcat_server.token
    if not token or token.strip() == "":
        return None
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {token}":
        return Server.Response(
            status=http.HTTPStatus.UNAUTHORIZED,
            headers=Server.Headers([("Content-Type", "text/plain")]),
            body=b"Unauthorized\n",
        )
    return None


async def napcat_server():
    cfg = get_config().napcat_server
    logger.info(f"启动 NapCat WS server ws://{cfg.host}:{cfg.port} 等待 NapCat 连入...")
    async with Server.serve(
        message_recv, cfg.host, cfg.port,
        max_size=2**26,
        process_request=check_napcat_server_token,
    ) as server:
        logger.info(f"Adapter 已就绪，监听 ws://{cfg.host}:{cfg.port}")
        await server.serve_forever()


async def main():
    await asyncio.gather(
        napcat_server(),
        mmc_start_com(),
        message_process(),
        check_timeout_response(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Adapter 已停止")
