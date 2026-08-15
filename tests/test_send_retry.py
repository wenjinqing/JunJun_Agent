"""发送重试测试（2026-08-04 用户反馈：消息一次失败就丢）。

分层策略：
- no connection -> 直接重发
- 确定性失败签名（rich media transfer failed 等）-> 跳过历史确认直接补发
  （2026-08-15 实锤：上传失败后本地历史留发送方残影，确认机制被骗吞补发）
- 超时/retcode + 有文本 -> 查历史：已送达免重发（EventChecker 误报），未送达补发
  · 只认 bot 自己发的（群友同文本不算数）
  · 媒体消息要求历史记录带同款 CQ 段（同文本纯文本残影不算数）
- 无文本（纯媒体）-> 盲补一次
"""

import pytest

import junjun_adapter_napcat.send_handler.send_retry as sr

_BOT = "2477702109"


def _ok():
    return {"status": "ok", "retcode": 0, "data": {"message_id": 123}}


def _failed():
    return {"status": "failed", "retcode": 1200, "message": "Connect Timeout"}


def _rich_media_failed():
    """2026-08-15 生产实锤：视频上传失败的 NapCat 原始报错。"""
    return {"status": "failed", "retcode": -1,
            "message": "EventChecker Failed: NodeIKernelMsgService/sendMsg "
                       "rich media transfer failed"}


def _text_params():
    return {"group_id": 999, "message": [{"type": "text", "data": {"text": "晚安大家"}}]}


def _video_params():
    return {"group_id": 999, "message": [
        {"type": "text", "data": {"text": "📺 小猫游戏时乱叫该如何应对？"}},
        {"type": "video", "data": {"file": "file:///F:/tmp/v.mp4"}}]}


def _history_with(texts, now=1000.0, sender=_BOT):
    """历史记录：默认是 bot 自己发的（self_id == sender）。"""
    return {"status": "ok", "data": {"messages": [
        {"time": now - 5, "raw_message": t, "self_id": _BOT,
         "user_id": sender, "sender": {"user_id": sender}} for t in texts]}}


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


class TestDeliveredInHistoryHardening20260815:
    """2026-08-15 生产实锤：B站视频 rich media transfer failed，本地历史留
    发送方残影，历史确认误判「已送达」吞掉补发，群友实际没收到。"""

    @pytest.mark.asyncio
    async def test_rich_media_failure_skips_history(self, _fake_sender, monkeypatch):
        """确定性失败签名：不查历史直接补发（残影会骗过确认）。"""
        fake, calls = _fake_sender
        fake.script = [_rich_media_failed(), _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _video_params())
        assert resp["status"] == "ok"
        assert calls == ["send_group_msg", "send_group_msg"]   # 无 get_history

    @pytest.mark.asyncio
    async def test_groupmate_same_text_not_confirmed(self, _fake_sender, monkeypatch):
        """群友消息里的同文本不算送达证据——只有 bot 自己发的才算。"""
        fake, calls = _fake_sender
        fake.script = [_failed(),
                       _history_with(["晚安大家"], sender="2664174279"), _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _text_params())
        assert resp["status"] == "ok"
        assert calls == ["send_group_msg", "get_group_msg_history", "send_group_msg"]

    @pytest.mark.asyncio
    async def test_media_send_text_only_history_not_confirmed(self, _fake_sender,
                                                              monkeypatch):
        """发的是图文混合：历史里同文本但没带视频段的记录不算媒体送达。"""
        fake, calls = _fake_sender
        fake.script = [_failed(),
                       _history_with(["📺小猫游戏时乱叫该如何应对？"]), _ok()]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _video_params())
        assert resp["status"] == "ok"
        assert calls[-1] == "send_group_msg"     # 不被纯文本残影骗过

    @pytest.mark.asyncio
    async def test_media_send_history_with_video_confirmed(self, _fake_sender,
                                                           monkeypatch):
        """历史里 bot 自己的同文本+视频段 -> EventChecker 误报，确认免重发。"""
        fake, calls = _fake_sender
        fake.script = [_failed(), _history_with(
            ["📺小猫游戏时乱叫该如何应对？[CQ:video,file=xxx.mp4]"])]
        monkeypatch.setattr(sr.time, "time", lambda: 1000.0)
        resp = await sr.send_with_retry("send_group_msg", _video_params())
        assert resp.get("_verified") is True
        assert calls == ["send_group_msg", "get_group_msg_history"]
