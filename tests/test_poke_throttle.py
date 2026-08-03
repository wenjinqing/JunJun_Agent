"""戳一戳防刷屏（2026-08-03 用户反馈：群友连戳 -> bot 每条都回，太烦人）。

三层抑制：同人窗口（60s）+ 会话地板（20s）+ 连戳升级吐槽（5 次后放一条，
再冷却 600s）。像真人：先不理，烦了就哼一声，而不是每下都回。
"""

from types import SimpleNamespace

import pytest

from junjun_adapter_napcat.recv_handler import notice_handler as nh


class _Clock:
    def __init__(self):
        self.t = 1_000_000.0


@pytest.fixture
def env(monkeypatch):
    sent = []
    clock = _Clock()

    async def _fake_send(msg_base):
        sent.append(msg_base)

    async def _allow(u, g):
        return True

    monkeypatch.setattr(
        "junjun_adapter_napcat.message_sending.message_send_instance.message_send",
        _fake_send)
    monkeypatch.setattr(nh, "message_handler_allow", _allow)
    monkeypatch.setattr(nh, "time", SimpleNamespace(time=lambda: clock.t))

    import junjun_adapter_napcat.config as cfg_mod
    # chat 配置走默认值（60/20/5/600）
    cfg = SimpleNamespace(junjun_server=SimpleNamespace(platform_name="qq"))
    monkeypatch.setattr(cfg_mod, "get_config", lambda: cfg)

    nh._reset_for_test()
    yield SimpleNamespace(sent=sent, clock=clock)
    nh._reset_for_test()


def _poke(user=12345, group=999):
    return {"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "self_id": 10000001, "target_id": 10000001,
            "user_id": user, "group_id": group}


class TestPokeThrottle:
    @pytest.mark.asyncio
    async def test_first_poke_passes(self, env):
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.sent) == 1
        assert env.sent[0].message_segment.data == "（戳了戳你）"

    @pytest.mark.asyncio
    async def test_rapid_pokes_suppressed_then_passes_after_window(self, env):
        await nh.notice_handler.handle_notice(_poke())
        await nh.notice_handler.handle_notice(_poke())  # +0s：抑制
        await nh.notice_handler.handle_notice(_poke())  # +0s：抑制
        assert len(env.sent) == 1
        env.clock.t += 61  # 过同人窗口（60s）
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.sent) == 2

    @pytest.mark.asyncio
    async def test_spam_escalates_once_then_long_cooldown(self, env):
        await nh.notice_handler.handle_notice(_poke())      # 放行
        for _ in range(4):                                   # 抑制 ×4（计数 1..4）
            await nh.notice_handler.handle_notice(_poke())
        assert len(env.sent) == 1
        await nh.notice_handler.handle_notice(_poke())      # 计数 5 -> 升级吐槽
        assert len(env.sent) == 2
        assert env.sent[1].message_segment.data == "（连续戳了你好几下）"
        # 升级后进入 600s 冷却：过 61s 仍不放行
        env.clock.t += 61
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.sent) == 2
        # 过冷却才恢复普通回应，且升级计数已清零（不反复吐槽）
        env.clock.t += 600
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.sent) == 3
        assert env.sent[2].message_segment.data == "（戳了戳你）"

    @pytest.mark.asyncio
    async def test_chat_floor_caps_multi_user_spam(self, env):
        await nh.notice_handler.handle_notice(_poke(user=111))
        assert len(env.sent) == 1
        # 另一个人 10s 内戳：撞会话地板（20s），不放行
        env.clock.t += 10
        await nh.notice_handler.handle_notice(_poke(user=222))
        assert len(env.sent) == 1
        # 地板命中不算该用户连戳（不喂升级计数）
        assert nh._suppressed.get(("g:999", "222"), 0) == 0
        # 过地板后另一人正常放行
        env.clock.t += 11
        await nh.notice_handler.handle_notice(_poke(user=222))
        assert len(env.sent) == 2

    @pytest.mark.asyncio
    async def test_private_chat_throttled_independently(self, env):
        await nh.notice_handler.handle_notice(_poke(user=111, group=None))
        await nh.notice_handler.handle_notice(_poke(user=111, group=None))
        assert len(env.sent) == 1
        # 私聊与群聊互相不影响
        await nh.notice_handler.handle_notice(_poke(user=111, group=999))
        # 群聊第一条受会话地板影响吗？不同 chat_key，不受影响——但同一时刻
        # 私聊刚放行过，群聊是另一个 chat_key，放行
        assert len(env.sent) == 2
