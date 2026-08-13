"""主动出站单点收口测试：日预算闸（2026-08-13 审查 P1）。

背景：订阅推送/提醒/任务汇报/深研成品全走 send_proactive，此前只有
主动搭话 loop 有自己的 per-chat 日限额，直发口没有总闸——循环 bug
或订阅风暴 = 无限刷屏。预算超限丢弃 + 当日首超上报管理员（一次）。
"""

import pytest

from junjun_agent import outbound
from junjun_core.contracts import ReplySegment


class _FakeGateway:
    def __init__(self):
        self.sent = []

    async def send_reply(self, rs):
        self.sent.append(rs)


@pytest.fixture()
def harness(monkeypatch):
    gw = _FakeGateway()
    from junjun_core.gateway import router
    monkeypatch.setattr(router, "get_gateway", lambda: gw)
    alerts = []

    async def _fake_notify(text):
        alerts.append(text)
        return True

    from junjun_core import security
    monkeypatch.setattr(security, "notify_admin", _fake_notify)
    outbound._reset_budget_for_test()
    yield {"gw": gw, "alerts": alerts}
    outbound._reset_budget_for_test()


def _set_budget(monkeypatch, global_n, chat_n):
    from junjun_core.config import get_global_config
    monkeypatch.setitem(get_global_config().raw, "outbound",
                        {"daily_global_budget": global_n,
                         "daily_chat_budget": chat_n})


def _seg(text="招呼"):
    return [ReplySegment(type="text", data=text)]


class TestProactiveBudget:
    @pytest.mark.asyncio
    async def test_under_budget_sends(self, harness, monkeypatch):
        _set_budget(monkeypatch, 3, 2)
        ok = await outbound.send_proactive("qq:1:private", _seg(), remember=False)
        assert ok and len(harness["gw"].sent) == 1

    @pytest.mark.asyncio
    async def test_global_budget_blocks_and_alerts_once(self, harness, monkeypatch):
        _set_budget(monkeypatch, 2, 99)
        assert await outbound.send_proactive("qq:1:private", _seg(), remember=False)
        assert await outbound.send_proactive("qq:2:private", _seg(), remember=False)
        assert not await outbound.send_proactive("qq:3:private", _seg(), remember=False)
        assert not await outbound.send_proactive("qq:4:private", _seg(), remember=False)
        assert len(harness["gw"].sent) == 2, "超限的一律不进网关"
        assert len(harness["alerts"]) == 1, "当日首超上报一次，别每丢一条骚扰一次"

    @pytest.mark.asyncio
    async def test_per_chat_budget_isolated(self, harness, monkeypatch):
        _set_budget(monkeypatch, 99, 1)
        assert await outbound.send_proactive("qq:a:group", _seg(), remember=False)
        assert not await outbound.send_proactive("qq:a:group", _seg(), remember=False)
        assert await outbound.send_proactive("qq:b:group", _seg(), remember=False)

    @pytest.mark.asyncio
    async def test_new_day_rearms(self, harness, monkeypatch):
        _set_budget(monkeypatch, 1, 99)
        assert await outbound.send_proactive("qq:1:private", _seg(), remember=False)
        assert not await outbound.send_proactive("qq:1:private", _seg(), remember=False)
        outbound._budget_day = "2000-01-01"  # 模拟跨天
        assert await outbound.send_proactive("qq:1:private", _seg(), remember=False)

    @pytest.mark.asyncio
    async def test_zero_disables(self, harness, monkeypatch):
        _set_budget(monkeypatch, 0, 0)
        for _ in range(5):
            assert await outbound.send_proactive("qq:1:private", _seg(), remember=False)
        assert len(harness["gw"].sent) == 5

    def test_default_budget_exists(self, harness):
        """默认配置下预算必须生效（不设=无限是事故敞口）。"""
        gb, cb = outbound._budget_cfg()
        assert gb > 0 and cb > 0
