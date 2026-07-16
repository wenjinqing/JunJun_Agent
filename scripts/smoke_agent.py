"""真实 LLM 冒烟（手动跑，不进 CI）：验证 gate + agent + skill + 后处理全链路。

阶段 3 起决策在会话队列内执行、发送走 gateway.send_reply——
这里注入 fake gateway 捕获出站消息，直接调 _handle 保持同步语义。

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
    import junjun_core.gateway.router as router_mod
    from junjun_agent.processor import _handle

    class FakeGateway:
        async def send_reply(self, reply):
            for seg in reply.segments:
                quote = f" [引用:{reply.reply_to_message_id}]" if reply.reply_to_message_id else ""
                print(f"  >>> 发送: {seg.data}{quote}", flush=True)

    router_mod._gateway = FakeGateway()

    session = ChatSession("qq:999:group", "qq", group_id="999")

    async def send(text, *, at_bot=False, user_id="111", nickname="甲", msg_id="1"):
        meta = InboundMeta(
            text=text, user_id=user_id, nickname=nickname,
            group_id="999", message_id=msg_id, at_bot=at_bot, is_self=False,
        )
        from junjun_agent.processor import _ensure_session_ready
        _ensure_session_ready(session)
        session.memory.add_user(text, nickname, user_id=user_id, message_id=msg_id, at_bot=at_bot)
        await _handle(session, meta)

    print("\n=== 用例 1: @君君 问时间（L1 旁路 -> get_time 工具）===", flush=True)
    await send("君君，现在几点了", at_bot=True, msg_id="1")

    print("\n=== 用例 2: 无关闲聊（应被 L2 gate 或 agent 判沉默）===", flush=True)
    await send("昨天那家火锅真不错", user_id="222", nickname="乙", msg_id="2")

    print("\n=== 用例 3: @君君 长回复（验证分条+延迟+错别字）===", flush=True)
    await send("君君，给我讲讲你最喜欢的动漫，多说几句", at_bot=True, msg_id="3")

    print("\n=== 用例 4: 关键词反应（人机质疑）===", flush=True)
    await send("君君你是不是人机啊", at_bot=True, user_id="222", nickname="乙", msg_id="4")

    print("\n[done]", flush=True)


asyncio.run(main())
