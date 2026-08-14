"""屏蔽名单（blocklist）+ 处理器硬闸 测试（0 token，内存库）。

铁律配套误判回归：被屏蔽的只是「不回复」，——正常用户、其他会话、
管理员本人一概不许误伤；消息仍要进记忆/落库（不搭理≠看不见）。
"""

import pytest

from junjun_core.gateway.router import InboundMeta
from junjun_core.gateway.session_manager import ChatSession

from junjun_agent import blocklist as bl


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """屏蔽名单写库——绝不许碰生产库：临时文件库 + 全表绑定（CLAUDE.md 硬约束）。"""
    import peewee
    import junjun_core.database.models as m
    test_db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
    with test_db.bind_ctx(m.ALL_TABLES):
        test_db.create_tables(m.ALL_TABLES)
        bl._reset_for_test()
        yield test_db
    bl._reset_for_test()


@pytest.fixture(autouse=True)
def _admin_env(monkeypatch):
    monkeypatch.setenv("ADMIN_QQ", "999")
    monkeypatch.setenv("JUNJUN_QQ_ACCOUNT", "10000001")


# ---------------------------------------------------------------- CRUD 与护栏

class TestBlockCrud:
    def test_block_and_is_blocked(self):
        ok, _ = bl.block("qq:999:group", "12345")
        assert ok
        assert bl.is_blocked("qq:999:group", "12345")

    def test_per_chat_isolation(self):
        """误判回归：A 群拉黑不影响 B 群/私聊。"""
        bl.block("qq:999:group", "12345")
        assert not bl.is_blocked("qq:888:group", "12345")
        assert not bl.is_blocked("qq:555:private", "12345")

    def test_unblock(self):
        bl.block("qq:999:group", "12345")
        ok, _ = bl.unblock("qq:999:group", "12345")
        assert ok and not bl.is_blocked("qq:999:group", "12345")

    def test_double_block_refused(self):
        bl.block("qq:999:group", "12345")
        ok, msg = bl.block("qq:999:group", "12345")
        assert not ok and "本来就在" in msg

    def test_unblock_not_blocked(self):
        ok, msg = bl.unblock("qq:999:group", "12345")
        assert not ok and "不在" in msg

    def test_persistence_across_cache_reset(self):
        """重启场景：清掉内存缓存后从库里重新加载，名单不丢。"""
        bl.block("qq:999:group", "12345", by="999")
        bl._reset_for_test()
        assert bl.is_blocked("qq:999:group", "12345")

    def test_cannot_block_admin(self):
        ok, msg = bl.block("qq:999:group", "999")
        assert not ok and "管理员" in msg

    def test_cannot_block_self(self):
        ok, msg = bl.block("qq:999:group", "10000001")
        assert not ok and "我自己" in msg

    def test_empty_target(self):
        ok, msg = bl.block("qq:999:group", "")
        assert not ok and "格式" in msg

    def test_list(self):
        bl.block("qq:999:group", "12345")
        bl.block("qq:999:group", "67890")
        assert bl.list_blocked("qq:999:group") == ["12345", "67890"]
        assert bl.list_blocked("qq:888:group") == []


# ---------------------------------------------------------------- 命令层

class _FakeSession:
    def __init__(self, chat_id="qq:999:group"):
        self.chat_id = chat_id


class _FakeMeta:
    user_id = "999"
    nickname = "管理员"


def _ctx(args):
    from junjun_agent.commands import CommandContext
    return CommandContext(session=_FakeSession(), meta=_FakeMeta(), args=args)


class TestCommands:
    @pytest.mark.asyncio
    async def test_block_cmd(self):
        r = await bl.block_cmd(_ctx("12345"))
        assert "不再理 12345" in r
        assert bl.is_blocked("qq:999:group", "12345")

    @pytest.mark.asyncio
    async def test_block_cmd_bad_args(self):
        r = await bl.block_cmd(_ctx("没有号码"))
        assert "格式" in r

    @pytest.mark.asyncio
    async def test_unblock_cmd(self):
        bl.block("qq:999:group", "12345")
        r = await bl.unblock_cmd(_ctx("12345"))
        assert "恢复搭理" in r
        assert not bl.is_blocked("qq:999:group", "12345")

    @pytest.mark.asyncio
    async def test_list_cmd(self):
        assert "没有屏蔽任何人" in await bl.blocklist_cmd(_ctx(""))
        bl.block("qq:999:group", "12345")
        assert "12345" in await bl.blocklist_cmd(_ctx(""))

    def test_commands_registered_admin_only(self):
        from junjun_agent.commands import _commands
        mine = {c.name: c for c in _commands
                if c.name in ("屏蔽", "取消屏蔽", "屏蔽列表")}
        assert len(mine) == 3
        assert all(c.admin_only for c in mine.values())   # 非管理员触发必被拒


class TestAtPrefixMatch:
    """命令匹配：句首「@你 」（@bot 的文本形态）剥掉再匹配——否则群里
    @君君 /命令 永远不命中（admin_only 命令在群里等于残废）。"""

    def test_at_prefix_slash_command(self):
        from junjun_agent.commands import _match
        hit = _match("@你 /屏蔽 12345")
        assert hit is not None and hit[0].name == "屏蔽" and hit[1] == "12345"

    def test_at_prefix_plain_text_no_match(self):
        """误判回归：@bot 说日常话不许误中任何命令。"""
        from junjun_agent.commands import _match
        assert _match("@你 你好呀") is None
        assert _match("@你") is None
        assert _match("@你 屏蔽一下他") is None      # 非斜杠不命中（屏蔽是 / 命令）

    def test_normal_text_no_match(self):
        from junjun_agent.commands import _match
        assert _match("今天天气怎么样") is None


# ---------------------------------------------------------------- 处理器硬闸

class TestProcessorGate:
    def _meta(self, text="你好", user_id="12345"):
        return InboundMeta(text=text, user_id=user_id, nickname="某 bot",
                           group_id="999", message_id="m1", at_bot=True,
                           is_self=False)

    @pytest.mark.asyncio
    async def test_blocked_user_never_queued_but_remembered(self, monkeypatch):
        """被屏蔽者 @bot 也不进决策队列；但消息照常进短期记忆（看得见，不搭理）。"""
        from junjun_agent import processor as proc_mod
        bl.block("qq:999:group", "12345")
        dispatched = []

        class _FakeQueues:
            def dispatch(self, *a, **kw):
                dispatched.append(a)

        monkeypatch.setattr(proc_mod, "_store_inbound", lambda *a, **kw: None)
        import junjun_agent.funnel.session_queue as sq
        monkeypatch.setattr(sq, "session_queues", _FakeQueues())
        session = ChatSession("qq:999:group", "qq", group_id="999")
        from junjun_memory.short_term import ShortTermMemory
        session.memory = ShortTermMemory()   # 绕过真实装配（不碰模型槽）

        class _FakeAgent:
            async def process(self, *a, **kw):
                return ""
        session.agent = _FakeAgent()
        await proc_mod.junjun_processor(session, self._meta())
        assert dispatched == []
        assert session.memory.entries                    # 记忆照记
        assert session.memory.entries[-1].text == "你好"

    @pytest.mark.asyncio
    async def test_normal_user_unaffected(self, monkeypatch):
        """误判回归：同群正常用户（哪怕和上一条同内容）照常进决策队列。"""
        from junjun_agent import processor as proc_mod
        bl.block("qq:999:group", "12345")
        dispatched = []

        class _FakeQueues:
            def dispatch(self, *a, **kw):
                dispatched.append(a)

        monkeypatch.setattr(proc_mod, "_store_inbound", lambda *a, **kw: None)
        import junjun_agent.funnel.session_queue as sq
        monkeypatch.setattr(sq, "session_queues", _FakeQueues())
        session = ChatSession("qq:999:group", "qq", group_id="999")
        from junjun_memory.short_term import ShortTermMemory
        session.memory = ShortTermMemory()

        class _FakeAgent:
            async def process(self, *a, **kw):
                return ""
        session.agent = _FakeAgent()
        await proc_mod.junjun_processor(session, self._meta(user_id="111"))
        assert len(dispatched) == 1


class TestAdminGateHint:
    """2026-08-14 实锤：管理员本人在群里忘 @bot 裸发 /屏蔽 被拒，
    一脸懵「为什么我不是管理员」——上报私聊里点破原因；非管理员
    的上报保持原口径（不向无关人暴露激活机制）。"""

    def _meta(self, user_id):
        return InboundMeta(text="/屏蔽 12345", user_id=user_id, nickname="温某",
                           group_id="999", message_id="m1", at_bot=False,
                           is_self=False)

    async def _dispatch_capture(self, monkeypatch, user_id):
        from junjun_agent import commands as cmd_mod
        from junjun_core import security
        reports = []
        monkeypatch.setattr(security, "report_violation",
                            lambda *a: reports.append(a))

        async def _noop_reply(self, *a, **kw):
            return None
        monkeypatch.setattr(cmd_mod.CommandContext, "reply", _noop_reply)
        # 权限位未激活（群里没 @bot）——由 processor 的 set_caller 语义决定
        security.admin_privileged.set(False)
        session = ChatSession("qq:999:group", "qq", group_id="999")
        handled = await cmd_mod.dispatch(session, self._meta(user_id))
        return handled, reports

    @pytest.mark.asyncio
    async def test_admin_without_at_gets_hint(self, monkeypatch):
        handled, reports = await self._dispatch_capture(monkeypatch, "999")
        assert handled is True
        assert reports and "@我" in reports[0][4]      # detail 带激活提示

    @pytest.mark.asyncio
    async def test_non_admin_report_has_no_hint(self, monkeypatch):
        """误判回归：普通群友的上报不暴露权限激活机制。"""
        handled, reports = await self._dispatch_capture(monkeypatch, "12345")
        assert handled is True
        assert reports and "@我" not in reports[0][4]
