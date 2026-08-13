"""alerting 0-token 测试（2026-08-13 审查 P1）：号池空防抖 / token 日累计超阈 /
每日用量汇总推送。notify_admin 全部 monkeypatch 捕获，不触网关。

DB 用 tmp 文件库 + bind_ctx（CLAUDE.md 硬约束：测试绝不写生产库）。
"""

import time

import pytest

from junjun_core import alerting


@pytest.fixture(autouse=True)
def _reset():
    alerting._reset_for_test()
    yield
    alerting._reset_for_test()


@pytest.fixture()
def captured(monkeypatch):
    sent = []

    async def _fake(text):
        sent.append(text)
        return True

    monkeypatch.setattr("junjun_core.security.notify_admin", _fake)
    return sent


class TestPoolEmpty:
    @pytest.mark.asyncio
    async def test_fires_once_within_debounce(self, captured):
        alerting.note_pool_empty()
        alerting.note_pool_empty()   # 4h 内第二次必须被防抖吞掉
        await asyncio_sleep()
        assert len(captured) == 1 and "号池空" in captured[0]

    @pytest.mark.asyncio
    async def test_fires_again_after_debounce(self, captured, monkeypatch):
        base = time.time()
        monkeypatch.setattr(alerting.time, "time", lambda: base)
        alerting.note_pool_empty()
        monkeypatch.setattr(alerting.time, "time",
                            lambda: base + alerting._POOL_DEBOUNCE + 1)
        alerting.note_pool_empty()
        await asyncio_sleep()
        assert len(captured) == 2

    def test_no_event_loop_degrades_to_log(self, capsys):
        """同步无循环上下文（启动早期）不炸，降级日志。"""
        alerting.note_pool_empty()
        assert "告警" in capsys.readouterr().out


class TestTokenThreshold:
    @pytest.mark.asyncio
    async def test_crossing_fires_once_a_day(self, captured, monkeypatch):
        monkeypatch.setattr(alerting, "_cfg",
                            lambda: {"daily_token_threshold": 1000})
        alerting.note_usage(600)
        alerting.note_usage(600)     # 越阈点
        alerting.note_usage(5000)    # 同日再越不再报
        await asyncio_sleep()
        assert len(captured) == 1 and "阈值" in captured[0]

    @pytest.mark.asyncio
    async def test_threshold_zero_off(self, captured, monkeypatch):
        monkeypatch.setattr(alerting, "_cfg", lambda: {"daily_token_threshold": 0})
        alerting.note_usage(10**9)
        await asyncio_sleep()
        assert captured == []

    @pytest.mark.asyncio
    async def test_new_day_rearms(self, captured, monkeypatch):
        monkeypatch.setattr(alerting, "_cfg",
                            lambda: {"daily_token_threshold": 100})
        alerting.note_usage(200)
        await asyncio_sleep()
        assert len(captured) == 1
        # 跨天：计数清零，告警重新武装
        monkeypatch.setattr(alerting.time, "strftime",
                            lambda fmt: "2099-01-01")
        alerting.note_usage(200)
        await asyncio_sleep()
        assert len(captured) == 2


class TestDailyReport:
    @pytest.mark.asyncio
    async def test_report_with_rows(self, captured, tmp_path):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx(m.ALL_TABLES):
            db.create_tables([m.LLMUsage])
            now = time.time()
            m.LLMUsage.create(time=now, model_name="m", request_type="agent",
                              prompt_tokens=1000, completion_tokens=200, chat_id="c")
            m.LLMUsage.create(time=now, model_name="m", request_type="agent",
                              prompt_tokens=500, completion_tokens=100, chat_id="c")
            m.LLMUsage.create(time=now, model_name="m", request_type="vlm",
                              prompt_tokens=300, completion_tokens=50, chat_id="c")
            m.LLMUsage.create(time=now - 90000, model_name="m", request_type="old",
                              prompt_tokens=9999, completion_tokens=0, chat_id="c")  # 超窗
            await alerting.daily_usage_report()
        assert len(captured) == 1
        text = captured[0]
        assert "agent：2 次" in text and "vlm：1 次" in text
        assert "old" not in text           # 超窗不计
        assert f"{1000+200+500+100+300+50:,}" in text  # 总数

    @pytest.mark.asyncio
    async def test_report_zero_usage_is_signal(self, captured, tmp_path):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx(m.ALL_TABLES):
            db.create_tables([m.LLMUsage])
            await alerting.daily_usage_report()
        assert len(captured) == 1 and "零 LLM 调用" in captured[0]


async def asyncio_sleep():
    import asyncio
    await asyncio.sleep(0)
    await asyncio.sleep(0)
