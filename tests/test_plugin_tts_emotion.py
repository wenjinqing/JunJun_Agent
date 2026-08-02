"""TTS 情绪语气测试：心情->情绪码映射、LLM 显式情绪、参数透传、降级映射、开关。"""

import pytest

import junjun_core.config.config as cfg_mod
from junjun_skills.plugins.tts import emotion as emo
from junjun_skills.plugins.tts import tools as tts


@pytest.fixture(autouse=True)
def cfg(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"tts": {"enable_emotion": True, "emotion_scale": 4}})
    yield
    cfg_mod.global_config = old


class TestMoodMapping:
    @pytest.mark.parametrize("mood,code", [
        ("开心", "happy"), ("被夸了很得意", "excited"), ("有点兴奋", "excited"),
        ("生气", "angry"), ("气死了", "angry"), ("有点无语", "coldness"),
        ("难过", "sad"), ("委屈巴巴", "sad"), ("emo了", "sad"),
        ("震惊", "surprise"),
    ])
    def test_mood_to_code(self, mood, code):
        assert emo.mood_to_doubao(mood) == code

    @pytest.mark.parametrize("mood", ["平静", "", "在想你", "有点饿"])
    def test_no_match(self, mood):
        assert emo.mood_to_doubao(mood) == ""

    def test_gsv2p_mapping(self):
        assert emo.doubao_to_gsv2p("angry") == "生气"
        assert emo.doubao_to_gsv2p("") == "默认"
        assert emo.doubao_to_gsv2p("unknown") == "默认"


class TestLlmEmotion:
    def test_basic(self):
        assert emo.parse_llm_emotion("生气") == ("angry", 0)
        assert emo.parse_llm_emotion("温柔一点") == ("tender", 0)

    def test_crazy_boosts_scale(self):
        """「发疯/气炸」这类拉满强度。"""
        code, boost = emo.parse_llm_emotion("发疯")
        assert code == "angry" and boost == 1

    def test_unknown(self):
        assert emo.parse_llm_emotion("随便") is None
        assert emo.parse_llm_emotion("") is None


class TestResolve:
    def test_llm_explicit_wins(self, monkeypatch):
        """LLM 显式指定优先于心情。"""
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "开心")
        code, scale = emo.resolve_emotion("c1", "生气")
        assert code == "angry" and scale == 4

    def test_crazy_clamps_to_5(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "平静")
        code, scale = emo.resolve_emotion("c1", "发疯")
        assert code == "angry" and scale == 5  # 4+1

    def test_auto_from_chat_mood(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "难过")
        code, scale = emo.resolve_emotion("c1", "")
        assert code == "sad" and scale == 4

    def test_falls_back_to_self_mood(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "平静")
        monkeypatch.setattr(mood_manager, "get_self_mood", lambda: "开心")
        code, _ = emo.resolve_emotion("c1", "")
        assert code == "happy"

    def test_calm_means_no_param(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "平静")
        monkeypatch.setattr(mood_manager, "get_self_mood", lambda: "平静")
        assert emo.resolve_emotion("c1", "") == ("", 0)

    def test_disabled(self, monkeypatch):
        cfg_mod.global_config = cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"tts": {"enable_emotion": False}})
        assert emo.resolve_emotion("c1", "发疯") == ("", 0)

    def test_mood_system_crash_is_safe(self, monkeypatch):
        """心情系统炸了 -> 不带情绪，不炸合成。"""
        import junjun_express.mood as mood_mod
        monkeypatch.setattr(mood_mod.mood_manager, "get_mood",
                            lambda cid: (_ for _ in ()).throw(RuntimeError("db down")))
        assert emo.resolve_emotion("c1", "") == ("", 0)


class TestParamPassthrough:
    @pytest.mark.asyncio
    async def test_doubao_receives_emotion(self, monkeypatch):
        """豆包后端收到 emotion/scale 并透传给 ja_tts。"""
        monkeypatch.setenv("DOUBAO_TTS_API_KEY", "k")
        captured = {}

        async def _fake_synth(text, speaker, *, emotion="", emotion_scale=0):
            captured.update(text=text, speaker=speaker,
                            emotion=emotion, emotion_scale=emotion_scale)
            return b"audio"

        import junjun_skills.plugins.ja_tts.tools as ja
        monkeypatch.setattr(ja, "synthesize", _fake_synth)
        out = await tts.synthesize_doubao("你好呀", emotion="happy", emotion_scale=5)
        assert out == b"audio"
        assert captured["emotion"] == "happy" and captured["emotion_scale"] == 5

    @pytest.mark.asyncio
    async def test_gsv2p_payload_emotion(self, monkeypatch):
        """GSV2P payload 的 other_params.emotion 用中文情绪词。"""
        monkeypatch.setenv("TTS_GSV2P_TOKEN", "t")
        seen = {}

        class _Resp:
            status_code = 200
            content = b"x" * 200
        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, json=None, headers=None):
                seen.update(json["other_params"])
                return _Resp()
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        out = await tts.synthesize_gsv2p("你好", emotion_zh="生气")
        assert out == b"x" * 200
        assert seen["emotion"] == "生气"

    @pytest.mark.asyncio
    async def test_fallback_maps_emotion_to_gsv2p(self, monkeypatch):
        """豆包失败降级 gsv2p 时，emotion 码映射成中文词。"""
        async def _doubao_fail(text, *, emotion="", emotion_scale=0):
            return None
        async def _gsv2p_ok(text, *, emotion_zh="默认"):
            _gsv2p_ok.seen = emotion_zh
            return b"audio"
        monkeypatch.setattr(tts, "synthesize_doubao", _doubao_fail)
        monkeypatch.setattr(tts, "synthesize_gsv2p", _gsv2p_ok)
        monkeypatch.setattr(tts, "_backend_configured", lambda b: b == "gsv2p")
        used, audio = await tts._synthesize_with_fallback(
            "你好", "doubao", emotion="sad", emotion_scale=4)
        assert used == "gsv2p" and audio == b"audio"
        assert _gsv2p_ok.seen == "难过"

    @pytest.mark.asyncio
    async def test_synthesize_to_file_resolves_emotion(self, monkeypatch, tmp_path):
        """端到端（无网络）：心情「生气」-> 豆包收到 angry。"""
        monkeypatch.setattr(tts, "OUTPUT_DIR", tmp_path)
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "生气")
        captured = {}

        async def _doubao(text, *, emotion="", emotion_scale=0):
            captured.update(emotion=emotion, scale=emotion_scale)
            return b"audio"
        monkeypatch.setattr(tts, "synthesize_doubao", _doubao)
        monkeypatch.setattr(tts, "_backend_configured", lambda b: b == "doubao")

        path = await tts._synthesize_to_file("哼，不理你了", "doubao", chat_id="c1")
        assert path is not None and path.exists()
        assert captured["emotion"] == "angry" and captured["scale"] == 4


def _server_frame(ja, event, *, sid=b"", cid=b"", payload=b"{}"):
    """按协议布局手工拼一帧下行帧（_marshal 不写 connect_id，直接复用会被误读）。"""
    import struct
    buf = bytearray([0x11, (ja._MsgType.FULL_SERVER_RESPONSE << 4) | ja._FLAG_WITH_EVENT,
                     0x10, 0x00])
    buf += struct.pack(">i", int(event))
    if event not in ja._READ_NO_SESSION:
        buf += struct.pack(">I", len(sid)) + sid
    if event in ja._READ_CONNECT_ID:
        buf += struct.pack(">I", len(cid)) + cid
    buf += struct.pack(">I", len(payload)) + payload
    return bytes(buf)


class _FakeWS:
    """按握手顺序回放两帧：CONNECTION_STARTED -> SESSION_STARTED，之后断流。"""
    def __init__(self, ja):
        self.sent = []
        self._queue = [
            _server_frame(ja, ja._Event.CONNECTION_STARTED, cid=b"x"),
            _server_frame(ja, ja._Event.SESSION_STARTED, sid=b"s"),
        ]
    async def send(self, frame): self.sent.append(frame)
    async def recv(self):
        if self._queue:
            return self._queue.pop(0)
        raise RuntimeError("handshake done, stop")


class _Conn:
    def __init__(self, ws): self._ws = ws
    async def __aenter__(self): return self._ws
    async def __aexit__(self, *a): return False


def _capture_start_session(monkeypatch, **kw):
    """跑 _synthesize_ws 到握手完成，返回 START_SESSION 帧的 req_params。"""
    import json as _json
    import junjun_skills.plugins.ja_tts.tools as ja
    import websockets
    ws = _FakeWS(ja)
    monkeypatch.setattr(websockets, "connect", lambda *a, **k: _Conn(ws))
    import asyncio
    async def _run():
        with pytest.raises(Exception):
            await ja._synthesize_ws("你好", "k", "spk", **kw)
    asyncio.run(_run())
    assert len(ws.sent) >= 2, "握手没走到 START_SESSION"
    _, event, payload = ja._unmarshal(ws.sent[1])
    assert event == ja._Event.START_SESSION
    return _json.loads(payload.decode("utf-8"))["req_params"]


class TestJaTtsWire:
    def test_audio_params_include_emotion(self, monkeypatch):
        """emotion 写进 audio_params，scale 钳到 [1,5]。"""
        req = _capture_start_session(monkeypatch, emotion="angry", emotion_scale=9)
        ap = req["audio_params"]
        assert ap["emotion"] == "angry"
        assert ap["emotion_scale"] == 5

    def test_no_emotion_no_param(self, monkeypatch):
        """不带情绪时 audio_params 无 emotion 键（保持旧行为）。"""
        req = _capture_start_session(monkeypatch)
        assert "emotion" not in req["audio_params"]
