"""W3 自有项测试：命令总线多 slash 回归 / chat_frequency / cross_scene / intimacy / acpoke 增强。"""

from types import SimpleNamespace

import pytest

from junjun_agent import commands


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:10001:private")


def _meta(text, user_id="12345"):
    return SimpleNamespace(text=text, user_id=user_id, nickname="甲", at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text, user_id="12345", is_group=True):
    return commands.CommandContext(session=_session(is_group), meta=_meta(text, user_id),
                                   args=text.split(" ", 1)[1] if " " in text else "")


class TestMultiSlashRegression:
    """命令总线 _match 曾只检查第一个 slash 命令（emoji_manage 迁移时发现）。"""

    @pytest.mark.asyncio
    async def test_second_slash_command_matches(self, _fake_gateway):
        @commands.register_command("first", plugin="p1")
        async def _c1(ctx):
            return "一"

        @commands.register_command("second", plugin="p2")
        async def _c2(ctx):
            return "二"

        assert await commands.dispatch(_session(), _meta("/first")) is True
        assert _fake_gateway[-1].segments[0].data == "一"
        assert await commands.dispatch(_session(), _meta("/second 参数")) is True
        assert _fake_gateway[-1].segments[0].data == "二"


class TestChatFrequency:
    @pytest.mark.asyncio
    async def test_show(self, _fake_bot_config):
        import junjun_skills.plugins.chat_frequency.tools as cf
        out = await cf.chat_cmd(_ctx("/chat show"))
        assert "生效值" in out and "倍率" in out

    @pytest.mark.asyncio
    async def test_set_admin_and_clamp(self, _fake_bot_config, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "10001")
        import junjun_skills.plugins.chat_frequency.tools as cf
        from junjun_agent.funnel.frequency import frequency_control
        out = await cf.chat_cmd(_ctx("/chat talk_frequency 2.5", user_id="10001"))
        assert "2.50" in out
        assert frequency_control.state("qq:999:group").adjust_factor == 2.5
        out = await cf.chat_cmd(_ctx("/chat t 99", user_id="10001"))
        assert "3.00" in out  # 上限截断

    @pytest.mark.asyncio
    async def test_set_non_admin_refused(self, _fake_bot_config, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "10001")
        reports = []
        monkeypatch.setattr("junjun_core.security.report_violation",
                            lambda *a, **k: reports.append(a))
        import junjun_skills.plugins.chat_frequency.tools as cf
        out = await cf.chat_cmd(_ctx("/chat talk_frequency 2"))
        assert "管理员" in out and reports


class TestCrossScene:
    @pytest.mark.asyncio
    async def test_non_admin_refused(self, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "10001")
        reports = []
        monkeypatch.setattr("junjun_core.security.report_violation",
                            lambda *a, **k: reports.append(a))
        from junjun_core.security import current_user_id
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_skills.plugins.cross_scene.tools import query_cross_scene_chat
        current_user_id.set("12345")
        current_chat_id.set("qq:999:group")
        out = query_cross_scene_chat.invoke({"user_name": "乙"})
        assert "拒绝" in out and reports

    def test_admin_query_excludes_current(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ADMIN_QQ", "10001")
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            m.Messages.create(chat_id="qq:999:group", user_nickname="甲", time=1.0,
                              message_id="a", processed_plain_text="本会话的话", bot_id="1")
            m.Messages.create(chat_id="qq:888:group", user_nickname="乙", time=2.0,
                              message_id="b", processed_plain_text="别群聊火锅", bot_id="1")
            from junjun_core.security import current_user_id
            from junjun_skills.builtin.memory_skills import current_chat_id
            from junjun_skills.plugins.cross_scene.tools import query_cross_scene_chat
            current_user_id.set("10001")
            current_chat_id.set("qq:999:group")
            out = query_cross_scene_chat.invoke({"keyword": "火锅"})
            assert "别群聊火锅" in out and "本会话的话" not in out


class TestIntimacy:
    def test_accumulate_daily_cap_and_levels(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        from junjun_express import intimacy
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            today = "2026-07-21"
            for _ in range(50):  # 50 次普通互动，日上限 3.0
                intimacy._accumulate("u1", intimacy.GAIN_NORMAL, today)
            score, count, level = intimacy.get_intimacy("u1")
            assert score == 3.0 and count == 30  # 3.0/0.1=30 次后不再涨
            assert level == "陌生"
            intimacy._accumulate("u1", intimacy.GAIN_ADDRESSED, "2026-07-22")  # 跨天重置
            assert intimacy.get_intimacy("u1")[0] == pytest.approx(3.3)
            assert intimacy.level_name(95) == "挚友" and intimacy.level_name(55) == "朋友"

    @pytest.mark.asyncio
    async def test_command_describe(self, _fake_gateway, tmp_path):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            m.Intimacy.create(user_id="12345", score=66.0, interaction_count=100)
            import junjun_skills.plugins.intimacy.tools as it
            out = await it.intimacy_cmd(_ctx("查看好感度"))
            assert "66.0" in out and "朋友" in out and "100" in out


class TestPokeEnhance:
    @pytest.fixture(autouse=True)
    def _clear_poke_state(self):
        from junjun_skills.builtin import action_skills
        action_skills._poke_last.clear()

    @pytest.mark.asyncio
    async def test_nickname_resolve_and_send(self, _fake_gateway, monkeypatch, _fake_bot_config):
        from junjun_skills.builtin import action_skills
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:999:group")

        async def _members(gid):
            return [{"user_id": 111, "nickname": "乙", "card": "小乙"}]

        monkeypatch.setattr("junjun_core.napcat_client.get_group_members", _members)
        out = await action_skills.send_poke.ainvoke({"user_id": "小乙"})  # 精确群名片
        assert "已戳" in out
        assert _fake_gateway[0].segments[0].data == "111"
        action_skills._poke_last.clear()
        out = await action_skills.send_poke.ainvoke({"user_id": "乙"})  # 包含匹配
        assert "已戳" in out

    @pytest.mark.asyncio
    async def test_not_found_and_repeat_window(self, _fake_gateway, monkeypatch, _fake_bot_config):
        from junjun_skills.builtin import action_skills
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:999:group")

        async def _members(gid):
            return [{"user_id": 111, "nickname": "乙", "card": ""}]

        monkeypatch.setattr("junjun_core.napcat_client.get_group_members", _members)
        assert "没找到" in await action_skills.send_poke.ainvoke({"user_id": "不存在的人"})
        assert "已戳" in await action_skills.send_poke.ainvoke({"user_id": "111"})
        assert "刚戳过" in await action_skills.send_poke.ainvoke({"user_id": "111"})  # 5 分钟内
