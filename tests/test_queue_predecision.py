"""严厉审查 P1-7/P0-3 回归：
- 会话队列合并连发消息时，被吞掉的消息必须过决策前段（命令/拦截器不丢）
- bot 自己的消息以 bot 身份（而非 user）进短期记忆
"""

import asyncio

import pytest

from junjun_agent.funnel.session_queue import SessionQueue


@pytest.fixture(autouse=True)
def _memory_db(monkeypatch):
    """junjun_processor 会落库（_store_inbound）——绝不许写生产库
    （2026-08-06 第三次污染事故实锤：「画好了画好了~」写进了 data/junjun.db）。"""
    import junjun_core.database.models as m
    from peewee import SqliteDatabase
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx(m.ALL_TABLES):
        test_db.create_tables(m.ALL_TABLES)
        monkeypatch.setattr(m, "db", test_db)
        import junjun_core.database as pkg
        monkeypatch.setattr(pkg, "db", test_db)
        yield test_db


class TestDrainPreHandler:
    @pytest.mark.asyncio
    async def test_merged_messages_run_pre_handler(self):
        """「/sub add xxx」+「你在吗」连发：命令消息虽被合并，但前段必须执行。"""
        from types import SimpleNamespace
        pre_seen, handled = [], []
        gate = asyncio.Event()

        async def pre_handler(session, meta):
            pre_seen.append(meta.text)

        async def handler(session, meta):
            handled.append(meta.text)
            gate.set()

        session = SimpleNamespace(chat_id="qq:1:group")
        q = SessionQueue("qq:1:group", handler, pre_handler=pre_handler)
        m1 = SimpleNamespace(text="/sub add 123")
        m2 = SimpleNamespace(text="你在吗")
        # 两条消息在 worker 取件前都进队列 -> 触发合并
        q._queue.put_nowait((session, m1, __import__("time").time()))
        q._queue.put_nowait((session, m2, __import__("time").time()))
        q.start()
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()

        assert handled == ["你在吗"]            # 只决策最新一条
        assert "/sub add 123" in pre_seen       # 被合并的命令过了前段，没丢

    @pytest.mark.asyncio
    async def test_no_pre_handler_keeps_old_behavior(self):
        from types import SimpleNamespace
        handled = []
        gate = asyncio.Event()

        async def handler(session, meta):
            handled.append(meta.text)
            gate.set()

        session = SimpleNamespace(chat_id="qq:2:group")
        q = SessionQueue("qq:2:group", handler)
        q.put(session, SimpleNamespace(text="hi"))
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()
        assert handled == ["hi"]


class TestSelfMessageMemory:
    @pytest.mark.asyncio
    async def test_self_message_recorded_as_bot(self, monkeypatch):
        """bot 自己的消息（NapCat 回传）必须以 bot 身份入 STM——以 user 身份
        写入等于把自己说的话伪装成别人说的喂回模型（自我模仿补给线）。"""
        from types import SimpleNamespace
        from junjun_memory.short_term import ShortTermMemory
        import junjun_agent.funnel.session_queue as sq

        class FakeQueues:
            def dispatch(self, s, m, h, **_kw):
                pass
        monkeypatch.setattr(sq, "session_queues", FakeQueues())

        from junjun_agent.processor import junjun_processor
        session = SimpleNamespace(
            chat_id="qq:3:group", memory=ShortTermMemory(), agent=object(),
            is_group=True, platform="qq", group_id="3",
        )
        meta = SimpleNamespace(
            text="画好了画好了~", nickname="君君", user_id="999",
            message_id="m1", at_bot=False, is_self=True,
        )
        await junjun_processor(session, meta)
        assert len(session.memory.entries) == 1
        assert session.memory.entries[0].role == "bot"
