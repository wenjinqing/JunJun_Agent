"""频率控制单测：时段规则解析 + 动态调节因子。"""

from datetime import datetime

import pytest

from junjun_agent.funnel.frequency import (
    FrequencyControl, resolve_talk_value, _in_range, _parse_hhmm,
)


class TestTimeRange:
    def test_normal_range(self):
        assert _in_range(_parse_hhmm("10:30"), "09:00-22:59")
        assert not _in_range(_parse_hhmm("23:30"), "09:00-22:59")

    def test_overnight_range(self):
        assert _in_range(_parse_hhmm("23:30"), "23:00-02:00")
        assert _in_range(_parse_hhmm("01:00"), "23:00-02:00")
        assert not _in_range(_parse_hhmm("12:00"), "23:00-02:00")

    def test_bad_format_returns_false(self):
        assert not _in_range(600, "not-a-range")
        assert not _in_range(600, "")


class TestResolveTalkValue:
    @pytest.fixture(autouse=True)
    def _rules(self, _fake_bot_config):
        _fake_bot_config.raw["chat"]["enable_talk_value_rules"] = True
        _fake_bot_config.raw["chat"]["talk_value_rules"] = [
            {"target": "", "time": "00:00-08:59", "value": 0.5},
            {"target": "", "time": "09:00-22:59", "value": 0.92},
            {"target": "qq:777:group", "time": "09:00-22:59", "value": 0.3},
        ]

    def test_global_rule_by_time(self):
        v = resolve_talk_value("qq:999:group", now=datetime(2026, 7, 16, 6, 0))
        assert v == 0.5

    def test_specific_chat_overrides_global(self):
        v = resolve_talk_value("qq:777:group", now=datetime(2026, 7, 16, 12, 0))
        assert v == 0.3

    def test_no_match_falls_back_to_base(self, _fake_bot_config):
        _fake_bot_config.raw["chat"]["talk_value_rules"] = [
            {"target": "", "time": "03:00-04:00", "value": 0.1},
        ]
        v = resolve_talk_value("qq:999:group", now=datetime(2026, 7, 16, 12, 0))
        assert v == 0.9  # 基础 talk_value

    def test_rules_disabled(self, _fake_bot_config):
        _fake_bot_config.raw["chat"]["enable_talk_value_rules"] = False
        v = resolve_talk_value("qq:999:group", now=datetime(2026, 7, 16, 6, 0))
        assert v == 0.9


class TestFrequencyAdjust:
    def test_too_frequent_lowers(self):
        fc = FrequencyControl()
        fc.apply_evaluation("c1", "过于频繁", now_ts=1000)
        assert fc.state("c1").adjust_factor == pytest.approx(0.8)

    def test_too_few_raises(self):
        fc = FrequencyControl()
        fc.apply_evaluation("c1", "过少", now_ts=1000)
        assert fc.state("c1").adjust_factor == pytest.approx(1.2)

    def test_clamped_to_bounds(self):
        fc = FrequencyControl()
        for _ in range(20):
            fc.apply_evaluation("c1", "过于频繁", now_ts=1000)
        assert fc.state("c1").adjust_factor == pytest.approx(0.1)
        for _ in range(30):
            fc.apply_evaluation("c1", "过少", now_ts=1000)
        assert fc.state("c1").adjust_factor == pytest.approx(1.5)

    def test_cooldown_and_min_messages(self):
        fc = FrequencyControl()
        fc.state("c1").last_adjust_time = 1000
        for _ in range(25):
            fc.note_message("c1")
        assert not fc.should_evaluate("c1", now_ts=1100)   # 冷却未到
        assert fc.should_evaluate("c1", now_ts=1200)        # 160s 后可评
        fc.apply_evaluation("c1", "正常", now_ts=1200)
        assert not fc.should_evaluate("c1", now_ts=1400)    # 消息计数已清零

    def test_effective_combines_rule_and_factor(self, _fake_bot_config):
        _fake_bot_config.raw["chat"]["enable_talk_value_rules"] = False
        fc = FrequencyControl()
        fc.apply_evaluation("c1", "过于频繁", now_ts=1000)  # factor 0.8
        assert fc.effective_talk_value("c1") == pytest.approx(0.9 * 0.8)
