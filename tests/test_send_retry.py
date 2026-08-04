"""发送重试测试（2026-08-04 用户反馈：消息一次失败就丢）。

分层策略：
- no connection -> 直接重发
- 超时/retcode + 有文本 -> 查历史：已送达免重发（EventChecker 误报），未送达补发
- 无文本（纯媒体）-> 盲补一次
"""

import pytest

import junjun_adapter_napcat.send_handler.send_retry as sr


def _ok():
    return {"status": "ok", "retcode": 0, "data": {"message_id": 123}}


def _failed():
    return {"status": "failed", "retcode": 1200, "message": "Connect Timeout"}


def _text_params():
    return {"group_id": 999, "message": [{"type": "text", "data": {"text": "晚安大家"}}]}


def _history_with(texts, now=1000.0):
    return {"status": "ok", "data": {"messages": [
        {"time": now - 5, "raw_message": t} for t in texts]}}


@pytest.fixture
def _fake_sender(monkeypatch):
    calls = []

    class _Fake:
        def __init__(self):
            self.script = []      # [(action预期不需要), resp] 按顺序返回
        async def send_message_to_napcat(self, action, params):
            calls.append(action)
            return self.script.pop(0) if self.script else _ok()

    fake = _Fake()
    monkeypatch.setattr(sr.nc_message_sender, "send_message_to_napcat",
                        fake.send_message_to_napcat)
    # 不等真实秒数
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(sr.asyncio, "sleep", _no_sleep)
    return fake, calls


class TestSendWithRetry:
    @pytest.mark.asyncio
    async def test_ok_first_try_no_retry(self, _fake_sender, monkeypatch):
        fake, calls = _fake_sender
        fake.script = [_ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert resp["status"] == "ok"
        assert calls == ["send_group_msg"]          # 不重发不查历史

    @pytest.mark.asyncio
    async def test_no_connection_direct_resend(self, _fake_sender, monkeypatch):
        fake, calls = _fake_sender
        fake.script = [{"status": "error", "message": "no connection"}, _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert resp["status"] == "ok"
        assert calls == ["send_group_msg", "send_group_msg"]   # 直接重发，不查历史

    @pytest.mark.asyncio
    async def test_failed_but_delivered_no_resend(self, _fake_sender, monkeypatch):
        """EventChecker 误报：历史里有同文本 -> 不重发（防刷屏双发）。"""
        fake, calls = _fake_sender
        fake.script = [_failed(), _history_with(["晚安大家"])]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert resp["status"] == "ok" and resp.get("_verified") is True
        assert calls == ["send_group_msg", "get_group_msg_history"]

    @pytest.mark.asyncio
    async def test_failed_and_missing_resend(self, _fake_sender, monkeypatch):
        """历史确认真没送达 -> 补发一次。"""
        fake, calls = _fake_sender
        fake.script = [_failed(), _history_with(["别的话"]), _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert resp["status"] == "ok"
        assert calls == ["send_group_msg", "get_group_msg_history", "send_group_msg"]

    @pytest.mark.asyncio
    async def test_history_query_failure_resends(self, _fake_sender, monkeypatch):
        """历史查询本身失败：按未送达处理，补发。"""
        fake, calls = _fake_sender
        fake.script = [_failed(), {"status": "error", "message": "timeout"}, _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert resp["status"] == "ok"
        assert calls[-1] == "send_group_msg"

    @pytest.mark.asyncio
    async def test_stale_history_not_matched(self, _fake_sender, monkeypatch):
        """窗口外的同文本不算送达（120 秒前发过的晚安不算数）。"""
        fake, calls = _fake_sender
        fake.script = [_failed(), _history_with(["晚安大家"], now=600), _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 800.0)  # since=800, 历史 time=595 在 120s 窗口外
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert calls[-1] == "send_group_msg"

    @pytest.mark.asyncio
    async def test_media_only_blind_resend(self, _fake_sender, monkeypatch):
        """纯图消息无文本指纹：直接补发一次。"""
        fake, calls = _fake_sender
        fake.script = [_failed(), _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        params = {"group_id": 999,
                  "message": [{"type": "image", "data": {"file": "base64://xx"}}]}
        resp = await sr.send_with_retry("send_group_msg", params)
        assert resp["status"] == "ok"
        assert calls == ["send_group_msg", "send_group_msg"]   # 不查历史

    @pytest.mark.asyncio
    async def test_private_history_action(self, _fake_sender, monkeypatch):
        """私聊走 get_friend_msg_history。"""
        fake, calls = _fake_sender
        fake.script = [_failed(), _history_with(["晚安"])]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        params = {"user_id": 123, "message": [{"type": "text", "data": {"text": "晚安"}}]}
        resp = await sr.send_with_retry("send_private_msg", params)
        assert "get_friend_msg_history" in calls

    @pytest.mark.asyncio
    async def test_retry_also_fails_returns_failure(self, _fake_sender, monkeypatch):
        fake, calls = _fake_sender
        fake.script = [_failed(), _history_with([]), _failed()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert resp["status"] == "failed"
        assert len(calls) == 3            # 只补一次，不无限重试
