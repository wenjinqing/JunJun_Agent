"""心跳看门狗测试（2026-08-09「突然收不到消息」定位仪器）。

读法对应三条支路：
- 心跳停 → NapCat↔Adapter 或 NapCat 进程（check_heartbeat 告警）
- 心跳在但 online/good=false → NapCat↔腾讯 协议层（handler 内告警）
- 心跳正常且在线 → 腾讯侧吞消息
"""

import time

import pytest

import junjun_adapter_napcat.recv_handler.meta_event_handler as meh


@pytest.fixture(autouse=True)
def _reset_state():
    meh._last_heartbeat_ts = 0.0
    meh._last_event_ts = 0.0
    meh._last_online = True
    meh._alarm_active = False
    yield
    meh._last_heartbeat_ts = 0.0
    meh._last_event_ts = 0.0
    meh._last_online = True
    meh._alarm_active = False


class TestCheckHeartbeat:
    def test_no_heartbeat_seen_no_alarm(self):
        """启动初期 NapCat 还没连入：不告警（不是故障）。"""
        assert meh.check_heartbeat() is None

    def test_fresh_heartbeat_no_alarm(self):
        meh._last_heartbeat_ts = time.time()
        assert meh.check_heartbeat() is None

    def test_stale_heartbeat_alarms_once(self):
        """超时进入告警态只报一次，持续超时不刷屏。"""
        meh._last_heartbeat_ts = time.time() - 200
        first = meh.check_heartbeat()
        assert first and "心跳超时" in first
        assert meh.check_heartbeat() is None  # 告警态中不重复报

    def test_recovery_reported_once(self):
        meh._last_heartbeat_ts = time.time() - 200
        assert meh.check_heartbeat() is not None      # 进告警态
        meh._last_heartbeat_ts = time.time()          # 恢复
        msg = meh.check_heartbeat()
        assert msg and "恢复" in msg
        assert meh.check_heartbeat() is None          # 恢复也只报一次

    def test_flap_realarms(self):
        """恢复后再断：状态机回位，能再次告警。"""
        meh._last_heartbeat_ts = time.time() - 200
        meh.check_heartbeat()                          # 告警
        meh._last_heartbeat_ts = time.time()
        meh.check_heartbeat()                          # 恢复
        meh._last_heartbeat_ts = time.time() - 200
        assert meh.check_heartbeat() is not None       # 再次告警


class TestHeartbeatStatus:
    @pytest.mark.asyncio
    async def test_offline_status_logged_and_tracked(self):
        """心跳体 status.online=false -> napcat_online 翻 False（协议层断的证据）。"""
        await meh.meta_event_handler.handle_meta_event({
            "meta_event_type": "heartbeat",
            "status": {"online": False, "good": False},
        })
        assert meh.heartbeat_status()["napcat_online"] is False
        assert meh._last_heartbeat_ts > 0

    @pytest.mark.asyncio
    async def test_online_recovers(self):
        await meh.meta_event_handler.handle_meta_event({
            "meta_event_type": "heartbeat",
            "status": {"online": False, "good": True},
        })
        await meh.meta_event_handler.handle_meta_event({
            "meta_event_type": "heartbeat",
            "status": {"online": True, "good": True},
        })
        assert meh.heartbeat_status()["napcat_online"] is True

    @pytest.mark.asyncio
    async def test_missing_status_treated_online(self):
        """误判回归：心跳不带 status 字段（老 NapCat）不能误报离线。"""
        await meh.meta_event_handler.handle_meta_event({"meta_event_type": "heartbeat"})
        assert meh.heartbeat_status()["napcat_online"] is True


class TestActivity:
    def test_note_activity_updates_ts(self):
        meh.note_activity()
        assert meh.heartbeat_status()["last_event_age"] is not None
