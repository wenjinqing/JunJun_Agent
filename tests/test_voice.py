"""语音消息理解测试：音频获取/转写共享/有界等待/渲染/适配器语音段/网关提取。"""

import asyncio

import pytest

import junjun_core.config.config as cfg_mod
from junjun_memory import voice


@pytest.fixture
def env(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw={"voice": {"enable": True}, "perception": {"ready_wait_seconds": 3.0}})
    voice._PENDING.clear()
    yield monkeypatch
    voice._PENDING.clear()
    cfg_mod.global_config = old


class TestFetchAudio:
    @pytest.mark.asyncio
    async def test_http_download(self, env, monkeypatch):
        class _Resp:
            content = b"mp3bytes"

            def raise_for_status(self):
                pass

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url):
                return _Resp()
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Client())
        assert await voice._fetch_audio("http://x/v.mp3") == b"mp3bytes"

    @pytest.mark.asyncio
    async def test_get_record_base64(self, env, monkeypatch):
        import base64

        class _Sender:
            async def send_message_to_napcat(self, action, params):
                assert action == "get_record" and params["out_format"] == "mp3"
                return {"data": {"base64": base64.b64encode(b"mp3").decode()}}
        import junjun_adapter_napcat.send_handler.nc_sending as nc
        monkeypatch.setattr(nc, "nc_message_sender", _Sender())
        assert await voice._fetch_audio("abc.silk") == b"mp3"

    @pytest.mark.asyncio
    async def test_fetch_failure_none(self, env, monkeypatch):
        class _Sender:
            async def send_message_to_napcat(self, action, params):
                raise RuntimeError("napcat down")
        import junjun_adapter_napcat.send_handler.nc_sending as nc
        monkeypatch.setattr(nc, "nc_message_sender", _Sender())
        assert await voice._fetch_audio("abc.silk") is None


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_ok(self, env, monkeypatch):
        monkeypatch.setattr(voice, "_fetch_audio", lambda ref: _async(b"mp3"))
        import junjun_llm.asr as asr_mod

        async def _fake(data, **kw):
            return "今晚吃啥"
        monkeypatch.setattr(asr_mod, "transcribe_bytes", _fake)
        out = await voice.transcribe_voices(["ref1"])
        assert out == ["今晚吃啥"]
        assert voice.render_voice_block(out) == "对方发来一条语音，说的是：「今晚吃啥」"

    @pytest.mark.asyncio
    async def test_fetch_fail_placeholder(self, env, monkeypatch):
        async def _none(ref):
            return None
        monkeypatch.setattr(voice, "_fetch_audio", _none)
        out = await voice.transcribe_voices(["ref1"])
        assert out == ["[语音]"]
        assert voice.render_voice_block(out) == ""  # 全占位不渲染

    @pytest.mark.asyncio
    async def test_inflight_shared(self, env, monkeypatch):
        calls = []

        async def _slow(ref):
            calls.append(ref)
            await asyncio.sleep(0.05)
            return b"mp3"
        monkeypatch.setattr(voice, "_fetch_audio", _slow)
        import junjun_llm.asr as asr_mod

        async def _fake(data, **kw):
            return "转写"
        monkeypatch.setattr(asr_mod, "transcribe_bytes", _fake)
        a, b = await asyncio.gather(voice.transcribe_voices(["r"], wait=0),
                                    voice.transcribe_voices(["r"], wait=0))
        assert a == b == ["转写"] and len(calls) == 1

    @pytest.mark.asyncio
    async def test_bounded_wait_placeholder_but_task_lives(self, env, monkeypatch):
        """3s 内没转完 -> 本轮占位；在途任务不取消，完成后同语音直接命中。"""
        async def _slow(ref):
            await asyncio.sleep(0.2)
            return b"mp3"
        monkeypatch.setattr(voice, "_fetch_audio", _slow)
        import junjun_llm.asr as asr_mod

        async def _fake(data, **kw):
            return "迟到的转写"
        monkeypatch.setattr(asr_mod, "transcribe_bytes", _fake)
        out = await voice.transcribe_voices(["r"], wait=0.01)
        assert out == ["[语音]"]
        await asyncio.sleep(0.3)  # 等在途任务跑完
        out2 = await voice.transcribe_voices(["r"], wait=1.0)
        assert out2 == ["迟到的转写"]

    @pytest.mark.asyncio
    async def test_disabled(self, env, monkeypatch):
        cfg_mod.global_config.raw["voice"]["enable"] = False
        assert await voice.transcribe_voices(["r"]) == []


async def _async(v):
    return v


class TestAdapterAndGateway:
    def test_record_seg_emits_voice_ref(self):
        """适配器：record 段 -> [语音] 占位 + voice 引用段。"""
        from junjun_adapter_napcat.recv_handler.message_handler import MessageHandler
        h = MessageHandler.__new__(MessageHandler)  # 绕过 __init__ 依赖

        async def _run():
            return await h._parse_message_segments([
                {"type": "record", "data": {"file": "abc.silk", "url": "http://x/v"}},
            ], self_id="1", group_id="2")
        segs, at_bot = asyncio.run(_run())
        types = [s.type for s in segs]
        assert "text" in types and "voice" in types
        voice_seg = next(s for s in segs if s.type == "voice")
        assert voice_seg.data == "http://x/v"  # url 优先

    def test_gateway_extract_voices(self):
        from junjun_core.contracts import Seg
        from junjun_core.gateway.router import _extract_voices
        seg = Seg(type="seglist", data=[
            Seg(type="text", data="hi"),
            Seg(type="voice", data="abc.silk"),
        ])
        assert _extract_voices(seg) == ["abc.silk"]
