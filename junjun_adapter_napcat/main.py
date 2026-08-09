"""君君 NapCat Adapter 入口。

架构：
- NapCat 作为 WS client 主动连入 Adapter 的 WS server（napcat_server.port，默认 8095）。
- Adapter 作为 maim_message client 连接君君网关（junjun_server.port，默认 8092）。
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

# 8095 上当前存活的 WS 连接集合：正常只有 NapCat 一条，但重连交错/误连探针
# 会短暂出现两条——断开时需要能回退绑定到仍存活的那条。
_active_conns: set = set()


async def message_recv(server_connection: Server.ServerConnection):
    """NapCat 连接生命周期：连入 -> 收发 -> 断开。

    断线韧性（2026-07-29）：NapCat 重启/网络抖动会断开 WS，
    websockets 对未捕获的 ConnectionClosed 打 "connection handler failed"
    全栈 traceback。这里显式捕获：正常断连打 INFO、异常断连打 WARN，
    并清空发送器连接（外发消息在断线窗口期等待重连而不是写死连接）。
    """
    _active_conns.add(server_connection)
    # 连接侧防抢绑定（2026-08-06 审查残留窗口）：误连探针【连上】时不许顶掉
    # 仍存活的真实 NapCat 绑定——只在「无绑定或绑定对象已不在存活集」时绑新连接。
    # 真实 NapCat 重连场景由断开侧的 fallback 回绑兜底。
    others = _active_conns - {server_connection}
    if nc_message_sender.server_connection not in others:
        await nc_message_sender.set_server_connection(server_connection)
    if message_handler.server_connection not in others:
        await message_handler.set_server_connection(server_connection)
    logger.info("NapCat 已连入 Adapter")
    try:
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
    except Server.exceptions.ConnectionClosedOK:
        logger.info("NapCat 正常断开连接，等待重连...")
    except Server.exceptions.ConnectionClosedError as e:
        logger.warning(f"NapCat 连接异常断开（等待重连）: {type(e).__name__}: {e}")
    finally:
        # 只动「本连接」的绑定，并回退到仍存活的其他连接：
        # 多连接交错时（NapCat 重连/误连探针），后断开者若无条件清空，
        # 会把仍在用的真实 NapCat 发送绑定清掉——入站正常、出站全丢
        # （2026-08-06 实测踩坑：探针 WS 断开清掉了真实 NapCat 的绑定）。
        _active_conns.discard(server_connection)
        fallback = next(iter(_active_conns), None)
        if nc_message_sender.server_connection is server_connection:
            await nc_message_sender.set_server_connection(fallback)
        if message_handler.server_connection is server_connection:
            await message_handler.set_server_connection(fallback)


async def message_process():
    """消息消费循环：单条坏消息只打日志跳过——不能杀死整个收信管线。"""
    while True:
        message = await message_queue.get()
        try:
            post_type = message.get("post_type")
            if post_type == "message":
                await message_handler.handle_raw_message(message)
            elif post_type == "meta_event":
                await meta_event_handler.handle_meta_event(message)
            elif post_type == "notice":
                await notice_handler.handle_notice(message)
        except Exception as e:
            logger.warning(
                f"单条消息处理失败（已跳过，收信继续）: {type(e).__name__}: {e} "
                f"| post_type={message.get('post_type')}")
        finally:
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
    # 与 core gateway 对齐：非回环监听 + 空 token = 任何人可伪造 NapCat 连入
    # 注入任意消息（含伪造管理员 QQ）——硬拒绝启动
    _LOOPBACK = ("127.0.0.1", "::1", "localhost")
    if cfg.host not in _LOOPBACK and not (cfg.token or "").strip():
        raise SystemExit(
            f"拒绝启动：napcat_server.host={cfg.host} 是对外地址但未配 token"
            f"（.env NAPCAT_TOKEN），任何人都能伪造 NapCat 连入注入消息。")
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
