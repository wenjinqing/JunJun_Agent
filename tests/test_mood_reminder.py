"""情绪 + 提醒系统单测（阶段4/5）。"""

import time
from datetime import datetime

import pytest
from peewee import SqliteDatabase

from junjun_express.mood import MoodManager, _REGRESS_AFTER
from junjun_skills.builtin.reminder_skills import (
    parse_remind_time, parse_repeat_type, parse_weekly_time)


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

    # ---- 回归：2026-07-21 线上 bug（日组吞十位 → 11点变1点、10点45变0点）----
    def test_two_digit_hour(self):
        ts = parse_remind_time("明天11点", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 17, 11, 0)

    def test_two_digit_hour_today(self):
        ts = parse_remind_time("16点", now=self.NOW)  # 今天 16 点未过
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 16, 16, 0)

    def test_hour_with_minutes(self):
        ts = parse_remind_time("明天10点45分", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 17, 10, 45)

    def test_morning_modifier(self):
        ts = parse_remind_time("上午11点", now=datetime(2026, 7, 21, 2, 50))
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 21, 11, 0)

    def test_afternoon_modifier(self):
        ts = parse_remind_time("下午3点", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 16, 15, 0)

    def test_evening_modifier(self):
        ts = parse_remind_time("晚上8点", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 16, 20, 0)

    def test_noon_modifier(self):
        ts = parse_remind_time("中午12点半", now=datetime(2026, 7, 16, 11, 0))
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 16, 12, 30)

    def test_dot_format(self):
        ts = parse_remind_time("明天10.05", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 17, 10, 5)

    def test_invalid_hour_returns_none(self):
        assert parse_remind_time("25点", now=self.NOW) is None


class TestParseRepeatType:
    """周期表达识别（2026-08-06：eval 抓出「每天推送」能力缺口后补齐）。"""

    def test_daily_meitian(self):
        assert parse_repeat_type("每天早上8点") == ("daily", "早上8点")

    def test_daily_tiantian(self):
        assert parse_repeat_type("天天晚上9点提醒我") == ("daily", "晚上9点提醒我")

    def test_tiantian_guard_mingtian(self):
        """「明天天气」不算周期（router 同款子串坑）。"""
        assert parse_repeat_type("明天8点")[0] == ""
        assert parse_repeat_type("今天天气提醒我带伞")[0] == ""

    def test_weekly(self):
        assert parse_repeat_type("每周五晚上8点") == ("weekly", "周五晚上8点")

    def test_weekly_xingqi(self):
        assert parse_repeat_type("每星期日中午12点") == ("weekly", "周日中午12点")

    def test_none(self):
        assert parse_repeat_type("明天8点") == ("", "明天8点")


class TestParseWeeklyTime:
    NOW = datetime(2026, 7, 16, 14, 0)  # 周四

    def test_friday_evening(self):
        ts = parse_weekly_time("周五晚上8点", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 17, 20, 0)

    def test_same_day_passed_rolls_week(self):
        ts = parse_weekly_time("周四8点", now=self.NOW)  # 今天周四 8 点已过
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 23, 8, 0)

    def test_same_day_future_stays_today(self):
        ts = parse_weekly_time("周四16点", now=self.NOW)  # 今天 16 点未过
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 16, 16, 0)

    def test_half_and_noon(self):
        ts = parse_weekly_time("周日中午12点半", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 19, 12, 30)

    def test_digit_weekday(self):
        ts = parse_weekly_time("星期5晚上8点", now=self.NOW)
        assert datetime.fromtimestamp(ts) == datetime(2026, 7, 17, 20, 0)

    def test_gibberish_returns_none(self):
        assert parse_weekly_time("随便什么时候", now=self.NOW) is None


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

    def test_set_reminder_daily_repeat(self):
        """周期提醒落库 repeat_type=daily（eval daily-tech-news case 对应能力）。"""
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_skills.builtin.reminder_skills import set_reminder
        from junjun_core.database import ReminderTasks
        current_chat_id.set("qq:999:group")
        out = set_reminder.invoke({"content": "推科技新闻", "time_spec": "每天早上8点",
                                   "user_id": "111"})
        assert "已设好" in out and "每天" in out
        row = ReminderTasks.get(ReminderTasks.content == "推科技新闻")
        assert row.repeat_type == "daily"
        assert row.remind_time > time.time()

    def test_set_reminder_weekly_repeat(self):
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_skills.builtin.reminder_skills import set_reminder
        from junjun_core.database import ReminderTasks
        current_chat_id.set("qq:999:group")
        out = set_reminder.invoke({"content": "交周报", "time_spec": "每周五晚上8点",
                                   "user_id": "111"})
        assert "已设好" in out and "每周" in out
        row = ReminderTasks.get(ReminderTasks.content == "交周报")
        assert row.repeat_type == "weekly"
        assert datetime.fromtimestamp(row.remind_time).weekday() == 4  # 周五

    def test_manage_mood_skill(self):
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_skills.builtin.reminder_skills import manage_mood
        current_chat_id.set("qq:999:group")
        manage_mood.invoke({"action": "set", "state": "开心"})
        out = manage_mood.invoke({"action": "get"})
        assert "开心" in out
