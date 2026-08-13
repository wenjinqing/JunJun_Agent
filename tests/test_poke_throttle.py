"""戳一戳防刷屏 + 群戳新政（0 token 廉价回敬）测试。

三层抑制：同人窗口（60s）+ 会话地板（20s）+ 连戳升级冷却（5 次后放一条，
再冷却 600s）——时机语义不变。
2026-08-13 用户裁决：群聊戳一戳不进决策链（token 消耗巨大 + 防 bot 互戳
滚雪球），放行的是 adapter 本地廉价回敬（反戳/表情，日额度 3 次）；
私聊维持合成文本进决策。本文件群路径断言全部改到 nc 发送记录上。
"""

import time as _real_time
from types import SimpleNamespace

import pytest

from junjun_adapter_napcat.recv_handler import notice_handler as nh


class _Clock:
    def __init__(self):
        self.t = 1_000_000.0


@pytest.fixture
def env(monkeypatch):
    sent = []
    nc_calls = []
    clock = _Clock()

    async def _fake_send(msg_base):
        sent.append(msg_base)

    async def _allow(u, g):
        return True

    class _FakeNC:
        async def send_message_to_napcat(self, action, params):
            nc_calls.append((action, params))
            return {"status": "ok"}

    monkeypatch.setattr(
        "junjun_adapter_napcat.message_sending.message_send_instance.message_send",
        _fake_send)
    monkeypatch.setattr(
        "junjun_adapter_napcat.send_handler.nc_sending.nc_message_sender", _FakeNC())
    monkeypatch.setattr(nh, "message_handler_allow", _allow)
    # 假钟只接管 time()；strftime/localtime 用真实现（日额度按天计数要用）
    monkeypatch.setattr(nh, "time", SimpleNamespace(
        time=lambda: clock.t, strftime=_real_time.strftime,
        localtime=_real_time.localtime))

    import junjun_adapter_napcat.config as cfg_mod
    # chat 配置走默认值（60/20/5/600/3）
    cfg = SimpleNamespace(junjun_server=SimpleNamespace(platform_name="qq"))
    monkeypatch.setattr(cfg_mod, "get_config", lambda: cfg)

    nh._reset_for_test()
    yield SimpleNamespace(sent=sent, nc_calls=nc_calls, clock=clock)
    nh._reset_for_test()


def _poke(user=12345, group=999):
    return {"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "self_id": 10000001, "target_id": 10000001,
            "user_id": user, "group_id": group}


class TestPokeThrottle:
    @pytest.mark.asyncio
    async def test_first_poke_passes(self, env):
        """群戳第一条：不进决策，本地廉价回敬一次。"""
        await nh.notice_handler.handle_notice(_poke())
        assert env.sent == []
        assert len(env.nc_calls) == 1

    @pytest.mark.asyncio
    async def test_rapid_pokes_suppressed_then_passes_after_window(self, env):
        await nh.notice_handler.handle_notice(_poke())
        await nh.notice_handler.handle_notice(_poke())  # +0s：抑制
        await nh.notice_handler.handle_notice(_poke())  # +0s：抑制
        assert len(env.nc_calls) == 1
        env.clock.t += 61  # 过同人窗口（60s）
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.nc_calls) == 2

    @pytest.mark.asyncio
    async def test_spam_escalates_once_then_long_cooldown(self, env):
        await nh.notice_handler.handle_notice(_poke())      # 放行（额度 1）
        for _ in range(4):                                   # 抑制 ×4（计数 1..4）
            await nh.notice_handler.handle_notice(_poke())
        assert len(env.nc_calls) == 1
        await nh.notice_handler.handle_notice(_poke())      # 计数 5 -> 升级放一条（额度 2）
        assert len(env.nc_calls) == 2
        # 升级后进入 600s 冷却：过 61s 仍不放行
        env.clock.t += 61
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.nc_calls) == 2
        # 过冷却才恢复普通回应，且升级计数已清零（不反复放）
        env.clock.t += 600
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.nc_calls) == 3

    @pytest.mark.asyncio
    async def test_chat_floor_caps_multi_user_spam(self, env):
        await nh.notice_handler.handle_notice(_poke(user=111))
        assert len(env.nc_calls) == 1
        # 另一个人 10s 内戳：撞会话地板（20s），不放行
        env.clock.t += 10
        await nh.notice_handler.handle_notice(_poke(user=222))
        assert len(env.nc_calls) == 1
        # 地板命中不算该用户连戳（不喂升级计数）
        assert nh._suppressed.get(("g:999", "222"), 0) == 0
        # 过地板后另一人正常放行
        env.clock.t += 11
        await nh.notice_handler.handle_notice(_poke(user=222))
        assert len(env.nc_calls) == 2

    @pytest.mark.asyncio
    async def test_private_chat_throttled_independently(self, env):
        """私聊维持进决策（合成文本）；与群聊额度/地板互不影响。"""
        await nh.notice_handler.handle_notice(_poke(user=111, group=None))
        await nh.notice_handler.handle_notice(_poke(user=111, group=None))
        assert len(env.sent) == 1                       # 私聊第二条被同人窗口抑制
        assert env.sent[0].message_segment.data == "（戳了戳你）"
        # 群聊是另一个 chat_key，不受私聊地板影响——走本地回敬（不进 sent）
        await nh.notice_handler.handle_notice(_poke(user=111, group=999))
        assert len(env.sent) == 1
        assert len(env.nc_calls) == 1

    @pytest.mark.asyncio
    async def test_group_daily_budget_caps_replies(self, env):
        """日额度 3：同群同人第 4 次起直接无视（token 止损命门）。"""
        for i in range(5):
            env.clock.t += 61                            # 每次都过同人窗口+地板
            await nh.notice_handler.handle_notice(_poke())
        assert env.sent == []
        assert len(env.nc_calls) == 3


class TestPokeStickerReply:
    """2026-08-13 用户裁决：群戳回敬发库存表情包，不再发内置小黄豆 emoji。"""

    @pytest.mark.asyncio
    async def test_sticker_branch_sends_image_not_face(self, env, tmp_path,
                                                       monkeypatch):
        sticker = tmp_path / "abc.jpg"
        sticker.write_bytes(b"\xff\xd8\xff")  # 内容不验，存在即可（不真发）
        monkeypatch.setattr(nh, "_pick_sticker", lambda: sticker)
        monkeypatch.setattr(nh.random, "random", lambda: 0.9)  # 表情包优先
        await nh.notice_handler.handle_notice(_poke())
        assert len(env.nc_calls) == 1
        action, params = env.nc_calls[0]
        assert action == "send_group_msg"
        seg = params["message"][0]
        assert seg["type"] == "image"
        assert seg["data"]["file"].startswith("file:///")  # NapCat 同机直读
        assert "face" not in str(env.nc_calls)

    @pytest.mark.asyncio
    async def test_empty_library_falls_back_to_poke(self, env, monkeypatch):
        """表情包库空：兜底反戳，不许退回发小黄豆。"""
        monkeypatch.setattr(nh, "_pick_sticker", lambda: None)
        monkeypatch.setattr(nh.random, "random", lambda: 0.9)  # 表情包优先但库空
        await nh.notice_handler.handle_notice(_poke())
        assert env.nc_calls[0][0] in ("send_poke", "send_group_poke")
        assert "face" not in str(env.nc_calls)

    @pytest.mark.asyncio
    async def test_poke_failure_falls_back_to_sticker(self, env, tmp_path,
                                                      monkeypatch):
        """反戳失败（NapCat 拒绝）兜底到表情包——原 face 兜底位的替身。"""
        sticker = tmp_path / "abc.png"
        sticker.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setattr(nh, "_pick_sticker", lambda: sticker)
        monkeypatch.setattr(nh.random, "random", lambda: 0.1)  # 反戳优先

        class _PokeFailNC:
            async def send_message_to_napcat(self, action, params):
                env.nc_calls.append((action, params))
                if "poke" in action:
                    return {"status": "failed", "wording": "戳不了"}
                return {"status": "ok"}

        monkeypatch.setattr(
            "junjun_adapter_napcat.send_handler.nc_sending.nc_message_sender",
            _PokeFailNC())
        await nh.notice_handler.handle_notice(_poke())
        actions = [a for a, _ in env.nc_calls]
        assert actions == ["send_poke", "send_group_poke", "send_group_msg"]
        assert env.nc_calls[-1][1]["message"][0]["type"] == "image"
        assert "face" not in str(env.nc_calls)

    def test_pick_sticker_filters_exts(self, tmp_path, monkeypatch):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.txt").write_text("not image")
        monkeypatch.setattr(nh, "_EMOJI_REG_DIR", tmp_path)
        for _ in range(20):
            p = nh._pick_sticker()
            assert p is not None and p.suffix == ".jpg"

    def test_pick_sticker_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nh, "_EMOJI_REG_DIR", tmp_path / "nonexistent")
        assert nh._pick_sticker() is None
