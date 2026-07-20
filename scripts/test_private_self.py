# -*- coding: utf-8 -*-
"""
真机自测脚本：发给君君自己（私聊链路验证）。

用法：
  1. .env 填好 MAIBOT_QQ_ACCOUNT（君君的 QQ 号）+ DEEPSEEK_API_KEY
  2. 启动君君：.venv\Scripts\python.exe scripts\run_junjun.py
  3. 启动 adapter：.venv\Scripts\python.exe -m junjun_adapter_napcat
  4. 等 NapCat 连接日志出现后，在君君自己的 QQ 私聊窗口发消息
  5. 观察君君是否回复（回复会发到同一个私聊窗口，即君君自己）

本脚本不走 QQ，而是注入 fake gateway 模拟君君自己收消息，
验证私聊链路：自己 -> gateway -> processor -> agent -> 回复 -> fake gateway（不真发）
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from junjun_core import initialize_logging
initialize_logging("INFO")

from junjun_core.gateway.session_manager import ChatSession
from junjun_core.gateway.router import InboundMeta
import junjun_core.gateway.router as router_mod
from junjun_agent.processor import _handle, _ensure_session_ready


class FakeGateway:
    """捕获出站消息，打印到控制台（不真发 QQ）"""
    def __init__(self):
        self.sent = []

    async def send_reply(self, reply):
        for seg in reply.segments:
            quote = f" [引用:{reply.reply_to_message_id}]" if reply.reply_to_message_id else ""
            line = f"  >>> 发送: {seg.data}{quote}"
            print(line, flush=True)
            self.sent.append(line)


async def main():
    # 从 .env 读君君自己的 QQ，不硬编码
    bot_qq = os.environ.get("MAIBOT_QQ_ACCOUNT", "")
    if not bot_qq:
        print("[ERROR] .env 中 MAIBOT_QQ_ACCOUNT 未设置，请填写君君的 QQ 号后重试", flush=True)
        return

    router_mod._gateway = FakeGateway()
    session = ChatSession(f"qq:{bot_qq}:private", "qq", user_id=bot_qq)
    _ensure_session_ready(session)

    async def send(text, msg_id):
        meta = InboundMeta(
            text=text, user_id=bot_qq, nickname="君君",
            group_id=None, message_id=msg_id, at_bot=False, is_self=True,
        )
        # 君君自己发的消息：is_self=True，processor 会拦截（不回复自己）
        # 所以这里用 is_self=False 模拟"别人"给君君发私聊
        meta.is_self = False
        session.memory.add_user(text, "君君", user_id=bot_qq, message_id=msg_id, at_bot=False)
        await _handle(session, meta)

    print("=== 用例 1: 私聊问时间 ===", flush=True)
    await send("君君，现在几点了", "p1")

    print("\n=== 用例 2: 闲聊（应沉默） ===", flush=True)
    await send("今天天气不错", "p2")

    print("\n=== 用例 3: 长回复（分条+错别字） ===", flush=True)
    await send("君君，给我讲讲你最喜欢的动漫", "p3")

    print("\n=== 用例 4: 人机质疑（keyword_reaction） ===", flush=True)
    await send("君君你是不是机器人", "p4")

    print("\n[done] 自测完成。若上面出现回复文本，私聊链路正常。", flush=True)


asyncio.run(main())
