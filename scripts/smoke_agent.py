"""真实 LLM 冒烟（手动跑，不进 CI）：验证 gate + agent + skill 全链路。

用法：.venv/Scripts/python.exe scripts/smoke_agent.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


async def main():
    from junjun_core import initialize_logging
    initialize_logging("INFO")

    from junjun_core.gateway.session_manager import ChatSession
    from junjun_core.gateway.router import InboundMeta
    from junjun_agent.processor import junjun_processor

    session = ChatSession("qq:999:group", "qq", group_id="999")

    print("\n=== 用例 1: @君君 问时间（应走 L1 旁路 -> agent 调 get_time）===", flush=True)
    reply = await junjun_processor(session, InboundMeta(
        text="君君，现在几点了", user_id="111", nickname="甲",
        group_id="999", message_id="1", at_bot=True, is_self=False,
    ))
    print(f">>> 回复: {reply.segments[0].data if reply else '(沉默)'}", flush=True)

    print("\n=== 用例 2: 无关闲聊（应被 L2 gate 或 agent 判沉默）===", flush=True)
    reply = await junjun_processor(session, InboundMeta(
        text="昨天那家火锅真不错", user_id="222", nickname="乙",
        group_id="999", message_id="2", at_bot=False, is_self=False,
    ))
    print(f">>> 回复: {reply.segments[0].data if reply else '(沉默)'}", flush=True)

    print("\n=== 用例 3: @君君 闲聊（应正常接话）===", flush=True)
    reply = await junjun_processor(session, InboundMeta(
        text="君君你喜欢吃火锅吗", user_id="111", nickname="甲",
        group_id="999", message_id="3", at_bot=True, is_self=False,
    ))
    print(f">>> 回复: {reply.segments[0].data if reply else '(沉默)'}", flush=True)


asyncio.run(main())
