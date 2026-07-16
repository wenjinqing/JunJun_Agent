"""阶段 4 记忆冒烟（真实 LLM + 真实 embedding，手动跑）。

验证：save_memory / recall_memory / manage_user_profile / 摘要 全链路。
用法：.venv/Scripts/python.exe scripts/smoke_memory.py
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
    from junjun_core.database import init_database
    init_database()

    from junjun_core.gateway.session_manager import ChatSession
    from junjun_core.gateway.router import InboundMeta
    import junjun_core.gateway.router as router_mod
    from junjun_agent.processor import _handle, _ensure_session_ready

    class FakeGateway:
        async def send_reply(self, reply):
            for seg in reply.segments:
                print(f"  >>> 发送: {seg.data}", flush=True)

    router_mod._gateway = FakeGateway()
    session = ChatSession("qq:999:group", "qq", group_id="999")

    async def send(text, *, at_bot=True, user_id="111", nickname="甲", msg_id="1"):
        meta = InboundMeta(text=text, user_id=user_id, nickname=nickname,
                           group_id="999", message_id=msg_id, at_bot=at_bot, is_self=False)
        _ensure_session_ready(session)
        session.memory.add_user(text, nickname, user_id=user_id, message_id=msg_id, at_bot=at_bot)
        await _handle(session, meta)

    from junjun_memory.embedding import get_embedding_client
    print(f"\nembedding 可用: {get_embedding_client().available}（False 则走关键词降级）", flush=True)

    print("\n=== 用例 1: 告知信息（应调 manage_user_profile / save_memory）===", flush=True)
    await send("君君记一下，我下周三过生日，别忘了", msg_id="1")

    print("\n=== 用例 2: 回忆（应调 recall_memory 命中生日）===", flush=True)
    await send("君君你还记得我生日是什么时候吗", msg_id="2")

    print("\n=== 用例 3: 画像验证（查库）===", flush=True)
    from junjun_memory.user_profile import get_profile_store
    points = get_profile_store().get_points("qq", "111")
    print(f"  用户 111 画像: {points}", flush=True)

    from junjun_memory.long_term import get_long_term_memory
    items = await get_long_term_memory().search("生日", top_k=3)
    print(f"  长期记忆命中: {[it.text for it in items]}", flush=True)

    print("\n[done]", flush=True)


asyncio.run(main())
