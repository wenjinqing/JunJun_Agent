"""HonestyGuard 单元测试（Phase 3）。"""

from types import SimpleNamespace

import pytest

from junjun_agent.honesty_guard import (
    record_tool_call, start_decision, verify, _CLAIM_RULES,
)


@pytest.fixture
def session():
    return SimpleNamespace(chat_id="qq:1:group")


def test_no_claim_no_issue(session):
    start_decision(session)
    ok, text, issues = verify(session, "今天天气不错")
    assert ok
    assert text == "今天天气不错"
    assert issues == []


def test_block_draw_claim_without_tool(session):
    start_decision(session)
    ok, text, issues = verify(session, "画好了，等下发给你")
    assert not ok
    assert "ai_draw" in " ".join(issues)
    assert "系统拦住我了" in text


def test_allow_draw_claim_with_tool(session):
    start_decision(session)
    record_tool_call(session, "ai_draw", result="图片任务已接受")
    ok, text, issues = verify(session, "画好了，等下发给你")
    assert ok
    assert text == "画好了，等下发给你"


def test_block_feed_claim(session):
    start_decision(session)
    ok, text, issues = verify(session, "说说已经发好了")
    assert not ok
    assert any("send_feed" in i for i in issues)


def test_allow_feed_claim_with_tool(session):
    start_decision(session)
    record_tool_call(session, "send_feed", result="说说已发布")
    ok, text, issues = verify(session, "说说已经发好了")
    assert ok


def test_only_recent_decision_tools_count(session):
    start_decision(session)
    record_tool_call(session, "ai_draw", result="old")
    # 模拟新一轮决策
    start_decision(session)
    ok, text, issues = verify(session, "画好了")
    assert not ok


def test_old_tool_calls_pruned_but_not_current(session):
    start_decision(session)
    record_tool_call(session, "ai_draw", result="new")
    ok, _, _ = verify(session, "画好了")
    assert ok
