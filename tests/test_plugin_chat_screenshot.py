"""chat_screenshot 插件测试：渲染 / 命令全链路 / 无记录降级 / 数量解析 / tool 路径。"""

from types import SimpleNamespace

import peewee
import pytest

from junjun_agent import commands
from junjun_core.database import models as m


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:1:private")


def _meta(text):
    return SimpleNamespace(text=text, user_id="12345", nickname="甲", at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text, is_group=True):
    return commands.CommandContext(session=_session(is_group), meta=_meta(text),
                                   args=text.split(" ", 1)[1] if " " in text else "")


def _seed_messages(db):
    """造 3 条消息（2 用户 + 1 bot），返回 chat_id。"""
    db.create_tables([m.Messages])
    m.Messages.create(chat_id="qq:999:group", user_id="12345", user_nickname="甲",
                      time=1.0, message_id="m1", processed_plain_text="今晚吃火锅吗",
                      bot_id="10000001", is_bot=False)
    m.Messages.create(chat_id="qq:999:group", user_id="", user_nickname="君君",
                      time=2.0, message_id="m2", processed_plain_text="好呀好呀",
                      bot_id="10000001", is_bot=True)
    m.Messages.create(chat_id="qq:999:group", user_id="12345", user_nickname="甲",
                      time=3.0, message_id="m3", processed_plain_text="那就这么定了",
                      bot_id="10000001", is_bot=False)
    return "qq:999:group"


class TestRender:
    def test_render_creates_png(self, tmp_path):
        import junjun_skills.plugins.chat_screenshot.tools as cs
        rows = [
            {"nickname": "甲", "timestamp": "2026-07-21 12:00:00",
             "text": "这是一条很长很长很长很长很长很长很长很长很长很长的消息" * 5,
             "is_bot": False},
            {"nickname": "君君", "timestamp": "2026-07-21 12:00:01",
             "text": "收到！\n换行也能渲染", "is_bot": True},
        ]
        out = cs.render_image(rows, tmp_path / "sub" / "shot.png")
        assert out.exists() and out.stat().st_size > 0


class TestParseCount:
    def test_parse_and_clamp(self):
        import junjun_skills.plugins.chat_screenshot.tools as cs
        assert cs._parse_count("") == 20          # 默认
        assert cs._parse_count("abc") == 20       # 非法回退默认
        assert cs._parse_count("30") == 30
        assert cs._parse_count("99") == 50        # 上限
        assert cs._parse_count("0") == 1          # 下限


class TestCommand:
    @pytest.mark.asyncio
    async def test_full_flow_sends_image(self, _fake_gateway, tmp_path, monkeypatch):
        import junjun_skills.plugins.chat_screenshot.tools as cs
        monkeypatch.setattr(cs, "DATA_DIR", tmp_path)
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Messages]):
            _seed_messages(db)
            result = await cs.screenshot_cmd(_ctx("/screenshot 10"))
            assert result is None
            assert len(_fake_gateway) == 1
            segs = _fake_gateway[0].segments
            assert segs[0].type == "image"
            img_path = segs[0].data
            assert img_path.startswith(str(tmp_path)) and img_path.endswith(".png")
            # 图确实落盘且非空
            from pathlib import Path
            assert Path(img_path).stat().st_size > 0
            assert _fake_gateway[0].target_group_id == "999"

    @pytest.mark.asyncio
    async def test_no_records_degrades(self, _fake_gateway, tmp_path, monkeypatch):
        import junjun_skills.plugins.chat_screenshot.tools as cs
        monkeypatch.setattr(cs, "DATA_DIR", tmp_path)
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            result = await cs.screenshot_cmd(_ctx("/screenshot"))
            assert "没有聊天记录可截图" in result
            assert _fake_gateway == []


class TestTool:
    @pytest.mark.asyncio
    async def test_tool_sends_screenshot(self, _fake_gateway, tmp_path, monkeypatch):
        import junjun_skills.plugins.chat_screenshot.tools as cs
        monkeypatch.setattr(cs, "DATA_DIR", tmp_path)
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:999:group")
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Messages]):
            _seed_messages(db)
            out = await cs.chat_screenshot_tool.ainvoke({"message_count": 20})
            assert "3 条消息" in out
            assert len(_fake_gateway) == 1
            rs = _fake_gateway[0]
            assert rs.target_group_id == "999" and rs.target_user_id is None
            assert rs.segments[0].type == "image"
            assert rs.should_reply is True

    @pytest.mark.asyncio
    async def test_tool_no_records(self, _fake_gateway, tmp_path, monkeypatch):
        import junjun_skills.plugins.chat_screenshot.tools as cs
        monkeypatch.setattr(cs, "DATA_DIR", tmp_path)
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:999:group")
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            out = await cs.chat_screenshot_tool.ainvoke({})
            assert "没有聊天记录可截图" in out
            assert _fake_gateway == []
