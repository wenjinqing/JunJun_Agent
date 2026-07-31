"""事件雷达测试：预过滤 / 抽取解析 / 时间校验 / 去重 / 容量上限。

LLM 与 reminder 落库均 monkeypatch；DB 用内存库隔离。
"""

import time
from datetime import datetime, timedelta

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m

test_db = SqliteDatabase(":memory:")


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


@pytest.fixture
def radar():
    old = cfg_mod.global_config
    _set_config({"event_radar": {"enable": True, "max_pending": 2, "lead_minutes": 15}})
    with test_db.bind_ctx([m.ReminderTasks]):
        test_db.create_tables([m.ReminderTasks])
        m.ReminderTasks.delete().execute()
        from junjun_agent.loop import event_radar
        yield event_radar
    cfg_mod.global_config = old


class TestPrefilter:
    def test_time_hints(self, radar):
        assert radar.should_scan("周六晚八点开黑")
        assert radar.should_scan("明天下午考试，救命")
        assert radar.should_scan("下周三 ddl 前交")
        assert radar.should_scan("8点半食堂拼单")

    def test_no_hint_rejected(self, radar):
        assert not radar.should_scan("哈哈哈哈笑死")
        assert not radar.should_scan("好的")
        assert not radar.should_scan("这个" * 100)  # 超长
        assert not radar.should_scan("")


class TestParse:
    def _future_iso(self, hours=24):
        return (datetime.now() + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M")

    def test_event_parsed(self, radar):
        raw = f'{{"is_event": true, "content": "开黑", "time": "{self._future_iso()}"}}'
        ev = radar.parse_extraction(raw)
        assert ev and ev["content"] == "开黑" and ev["ts"] > time.time()

    def test_not_event(self, radar):
        assert radar.parse_extraction('{"is_event": false}') is None
        assert radar.parse_extraction("随便什么文本") is None
        assert radar.parse_extraction('{"is_event": true, "content": "", "time": "x"}') is None

    def test_past_time_rejected(self, radar):
        past = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
        assert radar.parse_extraction(
            f'{{"is_event": true, "content": "开黑", "time": "{past}"}}') is None

    def test_too_far_rejected(self, radar):
        far = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M")
        assert radar.parse_extraction(
            f'{{"is_event": true, "content": "开黑", "time": "{far}"}}') is None


class TestRegister:
    def test_register_and_dedupe(self, radar):
        ts = time.time() + 3600
        assert radar.register_event("qq:1:group", "111", "甲", "开黑", ts)
        # 同事项 ±30 分钟内重复 -> 不建
        assert not radar.register_event("qq:1:group", "222", "乙", "开黑", ts + 600)
        rows = list(m.ReminderTasks.select())
        assert len(rows) == 1
        assert rows[0].content.startswith("群事件：开黑")
        # 提前 15 分钟提醒
        assert rows[0].remind_time < ts

    def test_capacity_cap(self, radar):
        base = time.time() + 3600
        assert radar.register_event("qq:1:group", "111", "甲", "开黑", base)
        assert radar.register_event("qq:1:group", "111", "甲", "聚餐", base + 7200)
        # max_pending=2，第三个被拒
        assert not radar.register_event("qq:1:group", "111", "甲", "考试", base + 10800)


class TestScan:
    @pytest.mark.asyncio
    async def test_full_chain(self, radar):
        """scan：LLM 抽取 -> 校验 -> 落库全链路（假模型）。"""
        future = (datetime.now() + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")

        class _Resp:
            content = f'{{"is_event": true, "content": "开黑", "time": "{future}"}}'

        class _Model:
            async def ainvoke(self, msgs, config=None):
                return _Resp()

        assert await radar.scan("qq:1:group", "111", "甲", "周六晚八点开黑", model=_Model())
        assert m.ReminderTasks.select().count() == 1

    @pytest.mark.asyncio
    async def test_llm_says_no(self, radar):
        class _Resp:
            content = '{"is_event": false}'

        class _Model:
            async def ainvoke(self, msgs, config=None):
                return _Resp()

        assert not await radar.scan("qq:1:group", "111", "甲", "明天见", model=_Model())
        assert m.ReminderTasks.select().count() == 0
