"""WS outbox 测试（候选 C）：断连暂存 / 重连回放 / TTL / 次数上限 / 保序。

DB 用 tmp_path 独立库（bind_ctx + create_tables），绝不触生产 junjun.db。
"""

import json
import time

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import db_writer
from junjun_core.database.models import OutboxMessage
from junjun_core.gateway import outbox

test_db = SqliteDatabase(":memory:")


class _FakeReply:
    """ReplySet 最小替身（enqueue 只用 platform/target/to_message_base）。"""

    def __init__(self, platform="qq", group="", user="u1", text="hi"):
        self.platform = platform
        self.target_group_id = group
        self.target_user_id = user
        self._text = text

    def to_message_base(self, bot_id):
        return {"text": self._text}


class _FakeServer:
    """broadcast_to_platform 桩：mode 控制返回。"""

    def __init__(self):
        self.sent = []
        self.mode = "ok"        # ok / no_conn / boom
        self.platform_connections = {"qq": {"conn-1"}}

    async def broadcast_to_platform(self, platform, payload):
        if self.mode == "boom":
            raise RuntimeError("ws 写炸了")
        if self.mode == "no_conn":
            return False
        self.sent.append((platform, payload))
        return True


@pytest.fixture
def env(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"gateway": {"outbox": True, "outbox_ttl_seconds": 1800,
                         "outbox_max_attempts": 3}})
    outbox._dirty.clear()
    with test_db.bind_ctx([OutboxMessage]):
        test_db.create_tables([OutboxMessage])
        OutboxMessage.delete().execute()
        yield
    outbox._dirty.clear()
    cfg_mod.global_config = old


class TestEnqueue:
    def test_failed_send_enqueues(self, env):
        r = _FakeReply(text="断连时的回复")
        outbox.enqueue(r, r.to_message_base(""))
        rows = list(OutboxMessage.select())
        assert len(rows) == 1
        assert rows[0].platform == "qq"
        assert json.loads(rows[0].payload_json)["text"] == "断连时的回复"
        assert outbox._dirty.get("qq") is True

    def test_disabled_no_enqueue(self, env, monkeypatch):
        monkeypatch.setattr(outbox, "_cfg", lambda: {"outbox": False})
        r = _FakeReply()
        outbox.enqueue(r, r.to_message_base(""))
        assert OutboxMessage.select().count() == 0


class TestFlush:
    def _seed(self, texts, platform="qq"):
        for i, t in enumerate(texts):
            OutboxMessage.create(platform=platform, target_user_id="u1",
                                 payload_json=json.dumps({"text": t}),
                                 created_ts=time.time() + i * 0.001, attempts=0)
        outbox.mark_dirty(platform)

    @pytest.mark.asyncio
    async def test_fifo_replay_and_delete(self, env):
        self._seed(["第一条", "第二条", "第三条"])
        server = _FakeServer()
        n = await outbox.flush(server, "qq")
        assert n == 3
        assert [p["text"] for _, p in server.sent] == ["第一条", "第二条", "第三条"]
        assert OutboxMessage.select().count() == 0, "成功回放后必须删除"

    @pytest.mark.asyncio
    async def test_no_conn_stops_without_penalty(self, env):
        """没有活连接：一条都不发、不扣次数、保留脏标记下轮再来。"""
        self._seed(["等连接"])
        server = _FakeServer()
        server.mode = "no_conn"
        n = await outbox.flush(server, "qq")
        assert n == 0 and not server.sent
        row = OutboxMessage.get()
        assert row.attempts == 0, "没连接不扣消息次数"
        assert outbox._dirty.get("qq") is True

    @pytest.mark.asyncio
    async def test_exception_bumps_attempts_and_keeps_order(self, env):
        """发送异常：扣一次机会、停本轮保序；第二条不受牵连。"""
        self._seed(["会炸的", "排后面的"])
        server = _FakeServer()
        server.mode = "boom"
        n = await outbox.flush(server, "qq")
        assert n == 0
        rows = list(OutboxMessage.select().order_by(OutboxMessage.created_ts))
        assert len(rows) == 2, "失败不删行"
        assert rows[0].attempts == 1 and rows[1].attempts == 0

    @pytest.mark.asyncio
    async def test_ttl_expired_dropped(self, env):
        OutboxMessage.create(platform="qq", target_user_id="u1",
                             payload_json=json.dumps({"text": "陈年旧回复"}),
                             created_ts=time.time() - 7200, attempts=0)
        outbox.mark_dirty("qq")
        server = _FakeServer()
        n = await outbox.flush(server, "qq")
        assert n == 0 and not server.sent, "过期回复不回放"
        assert OutboxMessage.select().count() == 0

    @pytest.mark.asyncio
    async def test_max_attempts_dropped(self, env):
        OutboxMessage.create(platform="qq", target_user_id="u1",
                             payload_json=json.dumps({"text": "倒霉蛋"}),
                             created_ts=time.time(), attempts=3)  # 上限=3
        outbox.mark_dirty("qq")
        server = _FakeServer()
        n = await outbox.flush(server, "qq")
        assert n == 0 and not server.sent
        assert OutboxMessage.select().count() == 0

    @pytest.mark.asyncio
    async def test_not_dirty_skips_db(self, env):
        """无脏标记零开销：不查库直接返回。"""
        server = _FakeServer()
        assert await outbox.flush(server, "qq") == 0

    @pytest.mark.asyncio
    async def test_maybe_flush_only_when_dirty(self, env):
        server = _FakeServer()
        await outbox.maybe_flush(server, "qq")     # 不脏：无操作
        assert not server.sent
        self._seed(["积压"])
        await outbox.maybe_flush(server, "qq")     # 脏：立即回放
        assert len(server.sent) == 1


class TestGatewayWiring:
    @pytest.mark.asyncio
    async def test_send_reply_failure_enqueues(self, env, monkeypatch):
        """router.send_reply 广播失败 → 进 outbox（不断言成功路径不进）。"""
        from junjun_core.contracts import ReplySegment, ReplySet
        from junjun_core.gateway.router import Gateway

        gw = Gateway(bot_user_id="1")
        server = _FakeServer()
        gw.server = server

        reply = ReplySet(platform="qq", target_user_id="u1",
                         segments=[ReplySegment(type="text", data="断连回复")],
                         should_reply=True)
        server.mode = "no_conn"
        await gw.send_reply(reply)
        assert OutboxMessage.select().count() == 1

        server.mode = "ok"
        await gw.send_reply(reply)
        assert OutboxMessage.select().count() == 1, "成功发送不进 outbox"
