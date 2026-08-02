"""感知后续推送测试：去重/完成补发/占位跳过/超时放弃/决策块「还在看」提示。"""

import asyncio
from types import SimpleNamespace

import pytest

import junjun_core.config.config as cfg_mod
from junjun_agent.loop import perception_followup as pf


@pytest.fixture
def env(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw={})
    pf._WATCHED.clear()
    sent = []

    class _FakeGW:
        async def send_reply(self, rs):
            sent.append(rs)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    session = SimpleNamespace(
        chat_id="qq:999:group",
        memory=SimpleNamespace(add_bot=lambda t: None))
    yield session, sent, monkeypatch
    pf._WATCHED.clear()
    cfg_mod.global_config = old


def _done_task(value):
    t = asyncio.get_event_loop().create_future()
    t.set_result(value)
    return t


def _stub_compose(mp, text="看完啦，是只橘猫"):
    async def _compose(session, results):
        return text
    mp.setattr(pf, "_compose", _compose)


class TestSchedule:
    @pytest.mark.asyncio
    async def test_followup_pushes(self, env):
        session, sent, mp = env
        _stub_compose(mp)
        t = asyncio.ensure_future(asyncio.sleep(0.01, result="一只橘猫"))
        pf.schedule(session, [{"kind": "image", "task": t}])
        await asyncio.sleep(0.1)
        assert len(sent) == 1
        assert sent[0].segments[0].data == "看完啦，是只橘猫"
        assert sent[0].target_group_id == "999"

    @pytest.mark.asyncio
    async def test_dedup_same_task(self, env):
        """同一在途任务注册两次只补一次（用户连发两条相关消息的场景）。"""
        session, sent, mp = env
        _stub_compose(mp)
        t = asyncio.ensure_future(asyncio.sleep(0.01, result="一只橘猫"))
        pf.schedule(session, [{"kind": "image", "task": t}])
        pf.schedule(session, [{"kind": "image", "task": t}])  # 重复注册
        await asyncio.sleep(0.1)
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_placeholder_skipped(self, env):
        """感知失败（占位结果）不补发。"""
        session, sent, mp = env
        _stub_compose(mp)
        t = asyncio.ensure_future(asyncio.sleep(0.01, result="[图片]"))
        pf.schedule(session, [{"kind": "image", "task": t}])
        await asyncio.sleep(0.1)
        assert not sent

    @pytest.mark.asyncio
    async def test_timeout_drops(self, env, monkeypatch):
        """超过 _MAX_WAIT 还没看完 -> 放弃补发（不堵后台）。"""
        session, sent, mp = env
        _stub_compose(mp)
        monkeypatch.setattr(pf, "_MAX_WAIT", 0.05)
        t = asyncio.ensure_future(asyncio.sleep(10, result="永远等不到"))
        pf.schedule(session, [{"kind": "video", "task": t}])
        await asyncio.sleep(0.15)
        assert not sent
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass


class TestCompose:
    @pytest.mark.asyncio
    async def test_llm_compose_and_fallback(self, env):
        session, _, mp = env

        class _FakeModel:
            async def ainvoke(self, messages, config=None):
                return type("R", (), {"content": "哎呀看完了，是只大橘"})()
        import junjun_llm
        mp.setattr(junjun_llm, "get_chat_model", lambda slot: _FakeModel())
        mp.setattr(junjun_llm, "get_callbacks", lambda: [])
        out = await pf._compose(session, [("图片", "一只橘猫")])
        assert "大橘" in out

        # LLM 炸了 -> 模板兜底
        class _Boom:
            async def ainvoke(self, *a, **kw):
                raise RuntimeError("api down")
        mp.setattr(junjun_llm, "get_chat_model", lambda slot: _Boom())
        out = await pf._compose(session, [("图片", "一只橘猫")])
        assert "一只橘猫" in out


class TestMemoryBlockPending:
    @pytest.mark.asyncio
    async def test_pending_hint_injected(self, monkeypatch):
        """3s 内没看完 -> 记忆块带「还在看」提示 + 返回在途条目。"""
        old = cfg_mod.global_config
        cfg_mod.global_config = cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
            raw={"perception": {"ready_wait_seconds": 0.01}})
        import junjun_memory.vision as vision_mod
        vision_mod._PENDING.clear()

        async def _slow_describe(data, *, model, prompt):
            await asyncio.sleep(0.2)
            return "一只橘猫"
        monkeypatch.setattr(vision_mod, "_describe", _slow_describe)

        async def _dl(url):
            return b"img"
        monkeypatch.setattr(vision_mod, "_download", _dl)
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot: object())

        from junjun_agent.processor import _build_memory_block
        session = SimpleNamespace(chat_id="qq:999:group", memory=None)
        meta = SimpleNamespace(image_urls=["http://x/1.jpg"], sticker_urls=None,
                               voice_records=None, video_urls=None, text="看图",
                               user_id="1", nickname="甲")
        block, pending = await _build_memory_block(session, meta)
        assert "还在看" in block and "绝不要说" in block
        assert len(pending) == 1 and pending[0]["kind"] == "image"
        vision_mod._PENDING.clear()
        cfg_mod.global_config = old
