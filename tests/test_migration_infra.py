"""插件迁移基础设施测试：命令总线 / 拦截器总线 / 发送段扩展 / Intimacy 表。"""

import json
from types import SimpleNamespace

import pytest

from junjun_agent import commands, interceptors


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    interceptors.clear_interceptors()
    yield
    commands.clear_commands()
    interceptors.clear_interceptors()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:12345:private")


def _meta(text, user_id="12345", at_bot=False):
    return SimpleNamespace(text=text, user_id=user_id, nickname="甲", at_bot=at_bot,
                           message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


class TestCommandBus:
    @pytest.mark.asyncio
    async def test_slash_command_with_args(self, _fake_gateway):
        @commands.register_command("draw", plugin="ai_draw")
        async def _cmd(ctx):
            return f"画了: {ctx.args}"

        handled = await commands.dispatch(_session(), _meta("/draw 一只猫"))
        assert handled is True
        assert _fake_gateway[0].segments[0].data == "画了: 一只猫"

    @pytest.mark.asyncio
    async def test_alias_and_no_match(self, _fake_gateway):
        @commands.register_command("draw", aliases=["绘图"], plugin="p")
        async def _cmd(ctx):
            return "ok"

        assert await commands.dispatch(_session(), _meta("/绘图")) is True
        assert await commands.dispatch(_session(), _meta("/不存在")) is False
        assert await commands.dispatch(_session(), _meta("普通聊天")) is False

    @pytest.mark.asyncio
    async def test_raw_keyword_command(self, _fake_gateway):
        @commands.register_command("抽老婆", raw=True, plugin="wife")
        async def _cmd(ctx):
            return "你老婆是乙"

        assert await commands.dispatch(_session(), _meta("抽老婆")) is True
        assert _fake_gateway[0].segments[0].data == "你老婆是乙"
        assert await commands.dispatch(_session(), _meta("今天抽老婆了吗")) is False

    @pytest.mark.asyncio
    async def test_admin_only_refused_and_reported(self, _fake_gateway, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "10001")
        reports = []
        monkeypatch.setattr("junjun_core.security.report_violation",
                            lambda *a, **k: reports.append(a))
        from junjun_core.security import set_caller

        @commands.register_command("purge", plugin="p", admin_only=True)
        async def _cmd(ctx):
            return "done"

        set_caller("12345", at_bot=False, is_group=True)
        assert await commands.dispatch(_session(), _meta("/purge", user_id="12345")) is True
        assert "管理员" in _fake_gateway[0].segments[0].data and reports
        # 管理员没 @bot：平时=普通群友，同样被拒
        _fake_gateway.clear()
        set_caller("10001", at_bot=False, is_group=True)
        assert await commands.dispatch(_session(), _meta("/purge", user_id="10001")) is True
        assert "管理员" in _fake_gateway[0].segments[0].data
        # 管理员 @bot：激活权限
        _fake_gateway.clear()
        set_caller("10001", at_bot=True, is_group=True)
        assert await commands.dispatch(_session(), _meta("/purge", user_id="10001")) is True
        assert _fake_gateway[0].segments[0].data == "done"

    @pytest.mark.asyncio
    async def test_disabled_plugin_command_ignored(self, _fake_gateway):
        from junjun_skills import registry
        from langchain_core.tools import tool

        @tool
        async def _dummy_tool(x: str) -> str:
            """dummy"""
            return x

        registry.clear()
        registry.register(_dummy_tool, plugin="wife")

        @commands.register_command("抽老婆", raw=True, plugin="wife")
        async def _cmd(ctx):
            return "ok"

        registry.set_plugin_enabled("wife", False)
        assert await commands.dispatch(_session(), _meta("抽老婆")) is False
        registry.set_plugin_enabled("wife", True)
        assert await commands.dispatch(_session(), _meta("抽老婆")) is True
        registry.clear()

    @pytest.mark.asyncio
    async def test_handler_exception_replies_error(self, _fake_gateway):
        @commands.register_command("boom", plugin="p")
        async def _cmd(ctx):
            raise RuntimeError("炸")

        assert await commands.dispatch(_session(), _meta("/boom")) is True
        assert "出错" in _fake_gateway[0].segments[0].data


class TestInterceptorBus:
    @pytest.mark.asyncio
    async def test_regex_hit_consumes(self, _fake_gateway):
        @interceptors.register_interceptor(r"v\.douyin\.com/\w+", plugin="douyin")
        async def _hit(ctx):
            await ctx.reply(f"解析: {ctx.args}")
            return True

        assert await interceptors.dispatch(_session(), _meta("看这个 https://v.douyin.com/abc123 好好笑")) is True
        assert "v.douyin.com/abc123" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_not_consumed_continues(self):
        @interceptors.register_interceptor(r"bilibili\.com", plugin="bili")
        async def _hit(ctx):
            return False

        assert await interceptors.dispatch(_session(), _meta("bilibili.com 的一个视频")) is False

    @pytest.mark.asyncio
    async def test_group_at_only(self, _fake_gateway):
        @interceptors.register_interceptor(r"pan\.baidu\.com", plugin="netdisk",
                                           group_at_only=True)
        async def _hit(ctx):
            return True

        # 群聊未 @ -> 不触发
        assert await interceptors.dispatch(_session(), _meta("pan.baidu.com/s/xx")) is False
        # 群聊 @ -> 触发
        assert await interceptors.dispatch(_session(), _meta("pan.baidu.com/s/xx", at_bot=True)) is True
        # 私聊不受 group_at_only 限制
        assert await interceptors.dispatch(_session(is_group=False),
                                           _meta("pan.baidu.com/s/xx")) is True

    @pytest.mark.asyncio
    async def test_handler_exception_not_consumed(self):
        @interceptors.register_interceptor(r"xxx", plugin="p")
        async def _hit(ctx):
            raise RuntimeError("boom")

        assert await interceptors.dispatch(_session(), _meta("xxx")) is False


class TestSendSegments:
    def _handler(self):
        from junjun_adapter_napcat.send_handler.main_send_handler import SendHandler
        return SendHandler()

    def test_video_at_mapping(self):
        h = self._handler()
        from maim_message import Seg
        payload = h._process_one(Seg(type="video", data="http://x/v.mp4"), [])
        assert payload == [{"type": "video", "data": {"file": "http://x/v.mp4"}}]
        payload = h._process_one(Seg(type="at", data="12345"), [])
        assert payload == [{"type": "at", "data": {"qq": "12345"}}]

    def test_music_json(self):
        h = self._handler()
        from maim_message import Seg
        payload = h._process_one(Seg(type="music", data=json.dumps(
            {"url": "http://x", "audio": "http://a", "title": "歌"})), [])
        assert payload[0]["type"] == "music"
        assert payload[0]["data"]["title"] == "歌" and payload[0]["data"]["type"] == "custom"

    def test_forward_extraction(self):
        h = self._handler()
        from maim_message import Seg
        nodes = [{"type": "node", "data": {"nickname": "甲", "user_id": "1",
                                           "content": [{"type": "text", "data": {"text": "hi"}}]}}]
        seg, forwards = h._extract_forwards(Seg(type="forward", data=json.dumps(nodes)))
        assert forwards == [nodes]
        assert seg.type == "text"
        # 混合：文本+forward
        mixed = Seg(type="seglist", data=[Seg(type="text", data="看"),
                                          Seg(type="forward", data=json.dumps(nodes))])
        rest, forwards = h._extract_forwards(mixed)
        assert forwards == [nodes] and rest.type == "text"


class TestIntimacyModel:
    def test_create_and_defaults(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            row = m.Intimacy.create(user_id="12345", score=5.0)
            assert m.Intimacy.get_by_id(row.id).score == 5.0
            assert "Intimacy" in [t.__name__ for t in m.ALL_TABLES]


class TestNapcatClient:
    @pytest.mark.asyncio
    async def test_unavailable_without_env(self, monkeypatch):
        monkeypatch.delenv("NAPCAT_HTTP_BASE", raising=False)
        from junjun_core import napcat_client
        assert napcat_client.available() is False
        assert await napcat_client.get_group_members("999") is None
