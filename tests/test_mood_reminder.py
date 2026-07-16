"""情绪 + 提醒系统单测（阶段4/5）。"""

import time
from datetime import datetime

import pytest
from peewee import SqliteDatabase

from junjun_express.mood import MoodManager, _REGRESS_AFTER
from junjun_skills.builtin.reminder_skills import parse_remind_time


@pytest.fixture(autouse=True)
def _memory_db(monkeypatch):
    import junjun_core.database.models as m
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx(m.ALL_TABLES):
        test_db.create_tables(m.ALL_TABLES)
        monkeypatch.setattr(m, "db", test_db)
        import junjun_core.database as pkg
        monkeypatch.setattr(pkg, "db", test_db)
        yield test_db


class TestMood:
    def test_default_mood(self):
        mm = MoodManager()
        assert mm.get_mood("c1") == "平静"

    def test_set_and_get(self):
        mm = MoodManager()
        mm.set_mood("c1", "开心")
        assert mm.get_mood("c1") == "开心"

    def test_regress_after_timeout(self):
        mm = MoodManager()
        mm.set_mood("c1", "兴奋")
        mm._moods["c1"].updated_at = time.time() - _REGRESS_AFTER - 1
        assert mm.get_mood("c1") == "平静"

    def test_mood_block_format(self):
        mm = MoodManager()
        mm.set_mood("c1", "有点无语")
        assert "有点无语" in mm.build_mood_block("c1")

    def test_disabled_returns_empty(self, _fake_bot_config):
        _fake_bot_config.raw["mood"] = {"enable_mood": False}
        mm = MoodManager()
        assert mm.get_mood("c1") == ""
        assert mm.build_mood_block("c1") == ""

    @pytest.mark.asyncio
    async def test_evaluate_updates_state(self):
        mm = MoodManager()

        class FakeModel:
            async def ainvoke(self, msgs, config=None):
                class R:
                    content = "被夸了很得意"
                return R()

        await mm.evaluate("c1", "甲: 君君你真棒", model=FakeModel())
        assert mm.get_mood("c1") == "被夸了很得意"

    @pytest.mark.asyncio
    async def test_evaluate_failure_keeps_state(self):
        mm = MoodManager()
        mm.set_mood("c1", "开心")

        class Broken:
            async def ainvoke(self, msgs, config=None):
                raise ConnectionError()

        await mm.evaluate("c1", "x", model=Broken())
        assert mm.get_mood("c1") == "开心"

    def test_eval_cooldown(self):
        mm = MoodManager()
        assert mm.should_evaluate("c1")
        mm._moods["c1"].last_eval = time.time()
        assert not mm.should_evaluate("c1")


class TestParseRemindTime:
    NOW = datetime(2026, 7, 16, 14, 0)

    def test_relative_minutes(self):
        ts = parse_remind_time("10分钟后", now=self.NOW)
        assert ts == self.NOW.timestamp() + 600

    def test_relative_hours(self):
        ts = parse_remind_time("2小时后", now=self.NOW)
        assert ts == self.NOW.timestamp() + 7200

    def test_tomorrow_hour(self):
        ts = parse_remind_time("明天8点", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 17, 8, 0)

    def test_absolute_date(self):
        ts = parse_remind_time("7月20日15:30", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 20, 15, 30)

    def test_past_hour_rolls_to_tomorrow(self):
        ts = parse_remind_time("8点", now=self.NOW)  # 今天 8 点已过
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 17, 8, 0)

    def test_gibberish_returns_none(self):
        assert parse_remind_time("等我有空再说", now=self.NOW) is None


class TestReminderLifecycle:
    def test_create_list_cancel(self):
        from junjun_agent.loop.reminder import create_reminder, list_pending, cancel_reminder
        tid = create_reminder("qq:999:group", "111", "开会", time.time() + 600)
        items = list_pending("qq:999:group")
        assert len(items) == 1 and items[0]["task_id"] == tid
        assert cancel_reminder(tid)
        assert list_pending("qq:999:group") == []
        assert not cancel_reminder(tid)  # 二次取消失败

    @pytest.mark.asyncio
    async def test_due_reminder_fires_and_completes(self, monkeypatch):
        from junjun_agent.loop.reminder import create_reminder, check_due_reminders
        from junjun_core.database import ReminderTasks

        sent = []

        class FakeGateway:
            async def send_reply(self, reply):
                sent.append(reply)

        import junjun_core.gateway.router as router_mod
        monkeypatch.setattr(router_mod, "_gateway", FakeGateway())
        # LLM 文案失败 -> 模板降级
        import junjun_llm
        def _broken(task):
            raise RuntimeError("no model")
        monkeypatch.setattr(junjun_llm, "get_chat_model", _broken)

        tid = create_reminder("qq:999:group", "111", "喝水", time.time() - 5)
        await check_due_reminders()
        assert len(sent) == 1
        assert "喝水" in sent[0].segments[0].data
        assert sent[0].target_group_id == "999"
        assert ReminderTasks.get(ReminderTasks.task_id == tid).is_completed

    @pytest.mark.asyncio
    async def test_daily_repeat_reschedules(self, monkeypatch):
        from junjun_agent.loop.reminder import create_reminder, check_due_reminders
        from junjun_core.database import ReminderTasks

        class FakeGateway:
            async def send_reply(self, reply):
                pass

        import junjun_core.gateway.router as router_mod
        monkeypatch.setattr(router_mod, "_gateway", FakeGateway())
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda t: (_ for _ in ()).throw(RuntimeError()))

        due_at = time.time() - 5
        tid = create_reminder("qq:1:private", "111", "吃药", due_at, repeat_type="daily")
        await check_due_reminders()
        row = ReminderTasks.get(ReminderTasks.task_id == tid)
        assert not row.is_completed
        assert row.remind_time == pytest.approx(due_at + 86400)


class TestReminderSkills:
    def test_set_reminder_skill(self):
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_skills.builtin.reminder_skills import set_reminder, list_reminders
        current_chat_id.set("qq:999:group")
        out = set_reminder.invoke({"content": "开会", "time_spec": "30分钟后", "user_id": "111"})
        assert "已设好" in out
        assert "开会" in list_reminders.invoke({})

    def test_set_reminder_bad_time(self):
        from junjun_skills.builtin.reminder_skills import set_reminder
        out = set_reminder.invoke({"content": "x", "time_spec": "随便什么时候", "user_id": "111"})
        assert "没听懂" in out

    def test_manage_mood_skill(self):
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_skills.builtin.reminder_skills import manage_mood
        current_chat_id.set("qq:999:group")
        manage_mood.invoke({"action": "set", "state": "开心"})
        out = manage_mood.invoke({"action": "get"})
        assert "开心" in out
