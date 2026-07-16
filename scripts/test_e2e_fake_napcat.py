"""端到端链路测试（无真实 QQ）：

fake NapCat (WS client) -> Adapter(8195) -> Gateway(8192, echo) [隔离端口] -> Adapter -> fake NapCat

验证阶段 1 验收标准：消息能进能出、echo 回复、名单过滤生效。
用法：.venv/Scripts/python.exe scripts/test_e2e_fake_napcat.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import websockets

PASS = []
FAIL = []

# 隔离端口：绝不能用生产 8095/8092——真实 NapCat 若在运行会重连进测试 adapter，
# echo 回复会发进真实 QQ 群（2026-07-16 实测事故）
TEST_NAPCAT_PORT = 8195
TEST_GATEWAY_PORT = 8192


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


async def start_gateway():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from junjun_core import initialize_logging
    initialize_logging("WARNING")
    from junjun_core.gateway.router import Gateway
    import junjun_core.gateway.router as router_mod
    router = Gateway(host="127.0.0.1", port=TEST_GATEWAY_PORT, bot_user_id="")
    router_mod._gateway = router
    await router.start()
    return router


async def start_adapter():
    # 覆盖 adapter 配置到隔离端口
    from junjun_adapter_napcat.config import get_config
    cfg = get_config()
    cfg.napcat_server.port = TEST_NAPCAT_PORT
    cfg.maibot_server.port = TEST_GATEWAY_PORT
    # Adapter 的三个核心协程（不含超时清理）
    from junjun_adapter_napcat.main import napcat_server, message_process
    from junjun_adapter_napcat.com_layer import mmc_start_com
    tasks = [
        asyncio.create_task(napcat_server()),
        asyncio.create_task(mmc_start_com()),
        asyncio.create_task(message_process()),
    ]
    return tasks


def fake_group_message(text: str, user_id: int = 12345, group_id: int = 999) -> str:
    return json.dumps({
        "post_type": "message",
        "message_type": "group",
        "message_id": int(time.time() * 1000) % 10_000_000,
        "group_id": group_id,
        "sender": {"user_id": user_id, "nickname": "tester", "card": ""},
        "message": [{"type": "text", "data": {"text": text}}],
        "raw_message": text,
    })


async def main():
    print("[1] 启动网关 (8192) ...", flush=True)
    gateway = await start_gateway()
    await asyncio.sleep(1)

    print("[2] 启动 Adapter (8195 <- fake NapCat, -> 8192 网关) ...", flush=True)
    adapter_tasks = await start_adapter()
    await asyncio.sleep(2)

    print("[3] fake NapCat 连入 Adapter ...", flush=True)
    async with websockets.connect("ws://127.0.0.1:8195") as nc:
        # 发一条群消息，期待 echo 回来
        await nc.send(fake_group_message("hello junjun e2e"))
        try:
            raw = await asyncio.wait_for(nc.recv(), timeout=10)
            action = json.loads(raw)
            check("收到回发的 OneBot action", action.get("action") == "send_group_msg", f"action={action.get('action')}")
            msg_arr = action.get("params", {}).get("message", [])
            text = "".join(s["data"]["text"] for s in msg_arr if s.get("type") == "text")
            check("echo 内容正确", "hello junjun e2e" in text, f"text={text!r}")
            check("目标群正确", action.get("params", {}).get("group_id") == 999)
            # 模拟 NapCat 回 ack（echo 字段回传），避免 adapter 等待超时
            await nc.send(json.dumps({"status": "ok", "retcode": 0, "echo": action.get("echo")}))
        except asyncio.TimeoutError:
            check("收到回发的 OneBot action", False, "10s 超时未收到")

        # 心跳事件不应产生回复
        await nc.send(json.dumps({"post_type": "meta_event", "meta_type": "heartbeat"}))
        try:
            raw = await asyncio.wait_for(nc.recv(), timeout=3)
            check("心跳不触发回复", False, f"意外收到: {raw[:100]}")
        except asyncio.TimeoutError:
            check("心跳不触发回复", True)

    print("[4] 名单过滤：ban_user 消息应被拦截 ...", flush=True)
    from junjun_adapter_napcat.config import get_config
    get_config().chat.ban_user_id = [66666]
    async with websockets.connect("ws://127.0.0.1:8195") as nc:
        await nc.send(fake_group_message("banned user msg", user_id=66666))
        try:
            raw = await asyncio.wait_for(nc.recv(), timeout=4)
            check("ban_user 被拦截", False, f"意外收到: {raw[:100]}")
        except asyncio.TimeoutError:
            check("ban_user 被拦截", True)

    for t in adapter_tasks:
        t.cancel()
    await gateway.stop()

    print(f"\n结果: {len(PASS)} PASS / {len(FAIL)} FAIL", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
