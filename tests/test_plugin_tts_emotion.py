"""TTS 情绪风格测试：心情->风格映射、LLM 显式情绪、参数组合透传、降级映射、开关。

设计背景（2026-08-02 实测）：vv 音色 emotion 参数只调音高色彩、不按类别区分，
可听情绪 = emotion + speech_rate + loudness_rate 参数组合（见 emotion.py 头注）。
"""

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
    @pytest.mark.parametrize("mood,style", [
        ("开心", "happy"), ("被夸了很得意", "excited"), ("有点兴奋", "excited"),
        ("生气", "angry"), ("气死了", "angry"), ("气炸了", "crazy"),
        ("有点无语", "coldness"), ("难过", "sad"), ("委屈巴巴", "sad"),
        ("emo了", "sad"), ("震惊", "surprised"), ("好温柔", "tender"),
    ])
    def test_mood_to_style(self, mood, style):
        assert emo.mood_to_style(mood) == style

    @pytest.mark.parametrize("mood", ["平静", "", "在想你", "有点饿"])
    def test_no_match(self, mood):
        assert emo.mood_to_style(mood) == ""

    def test_gsv2p_mapping(self):
        assert emo.style_to_gsv2p("angry") == "生气"
        assert emo.style_to_gsv2p("crazy") == "生气"
        assert emo.style_to_gsv2p("") == "默认"
        assert emo.style_to_gsv2p("unknown") == "默认"


class TestStylePresets:
    def test_angry_is_fast_loud(self):
        s = emo._style_dict("angry")
        assert s["emotion"] == "angry"
        assert s["speech_rate"] > 0 and s["loudness_rate"] > 0

    def test_sad_is_slow_quiet(self):
        s = emo._style_dict("sad")
        assert s["emotion"] == "sad"
        assert s["speech_rate"] < 0 and s["loudness_rate"] < 0

    def test_crazy_maxes_scale(self):
        s = emo._style_dict("crazy")
        assert s["emotion_scale"] == 5
        assert s["speech_rate"] > emo._style_dict("angry")["speech_rate"]

    def test_opposite_directions(self):
        """可听性核心：生气和难过在速率/音量上方向相反（categorical 区分）。"""
        a, s = emo._style_dict("angry"), emo._style_dict("sad")
        assert a["speech_rate"] * s["speech_rate"] < 0
        assert a["loudness_rate"] * s["loudness_rate"] < 0


class TestLlmEmotion:
    def test_basic(self):
        assert emo.parse_llm_emotion("生气") == "angry"
        assert emo.parse_llm_emotion("温柔一点") == "tender"

    def test_crazy(self):
        assert emo.parse_llm_emotion("发疯") == "crazy"

    def test_unknown(self):
        assert emo.parse_llm_emotion("随便") is None
        assert emo.parse_llm_emotion("") is None


class TestResolve:
    def test_llm_explicit_wins(self, monkeypatch):
        """LLM 显式指定优先于心情。"""
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "开心")
        s = emo.resolve_emotion("c1", "生气")
        assert s["style"] == "angry" and s["emotion_scale"] == 4

    def test_crazy_scale_5(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "平静")
        s = emo.resolve_emotion("c1", "发疯")
        assert s["style"] == "crazy" and s["emotion_scale"] == 5

    def test_auto_from_chat_mood(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "难过")
        s = emo.resolve_emotion("c1", "")
        assert s["style"] == "sad"

    def test_falls_back_to_self_mood(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "平静")
        monkeypatch.setattr(mood_manager, "get_self_mood", lambda: "开心")
        s = emo.resolve_emotion("c1", "")
        assert s["style"] == "happy"

    def test_calm_means_none(self, monkeypatch):
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "平静")
        monkeypatch.setattr(mood_manager, "get_self_mood", lambda: "平静")
        assert emo.resolve_emotion("c1", "") is None

    def test_disabled(self, monkeypatch):
        cfg_mod.global_config = cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"tts": {"enable_emotion": False}})
        assert emo.resolve_emotion("c1", "发疯") is None

    def test_mood_system_crash_is_safe(self, monkeypatch):
        """心情系统炸了 -> 不带情绪，不炸合成。"""
        import junjun_express.mood as mood_mod
        monkeypatch.setattr(mood_mod.mood_manager, "get_mood",
                            lambda cid: (_ for _ in ()).throw(RuntimeError("db down")))
        assert emo.resolve_emotion("c1", "") is None


class TestParamPassthrough:
    @pytest.mark.asyncio
    async def test_doubao_receives_style(self, monkeypatch):
        """豆包后端把 style 的四个参数全部透传给 ja_tts。"""
        monkeypatch.setenv("DOUBAO_TTS_API_KEY", "k")
        captured = {}

        async def _fake_synth(text, speaker, **kw):
            captured.update(text=text, speaker=speaker, **kw)
            return b"audio"

        import junjun_skills.plugins.ja_tts.tools as ja
        monkeypatch.setattr(ja, "synthesize", _fake_synth)
        style = emo._style_dict("angry")
        out = await tts.synthesize_doubao("你好呀", style=style)
        assert out == b"audio"
        assert captured["emotion"] == "angry"
        assert captured["speech_rate"] == 20 and captured["loudness_rate"] == 10

    @pytest.mark.asyncio
    async def test_doubao_no_style_defaults(self, monkeypatch):
        """无情绪时不传任何情绪参数（保持旧行为）。"""
        monkeypatch.setenv("DOUBAO_TTS_API_KEY", "k")
        captured = {}

        async def _fake_synth(text, speaker, **kw):
            captured.update(kw)
            return b"audio"

        import junjun_skills.plugins.ja_tts.tools as ja
        monkeypatch.setattr(ja, "synthesize", _fake_synth)
        await tts.synthesize_doubao("你好", style=None)
        assert captured == {}

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
    async def test_fallback_maps_style_to_gsv2p(self, monkeypatch):
        """豆包失败降级 gsv2p 时，风格映射成中文词。"""
        async def _doubao_fail(text, *, style=None):
            return None
        async def _gsv2p_ok(text, *, emotion_zh="默认"):
            _gsv2p_ok.seen = emotion_zh
            return b"audio"
        monkeypatch.setattr(tts, "synthesize_doubao", _doubao_fail)
        monkeypatch.setattr(tts, "synthesize_gsv2p", _gsv2p_ok)
        monkeypatch.setattr(tts, "_backend_configured", lambda b: b == "gsv2p")
        used, audio = await tts._synthesize_with_fallback(
            "你好", "doubao", style=emo._style_dict("sad"))
        assert used == "gsv2p" and audio == b"audio"
        assert _gsv2p_ok.seen == "难过"

    @pytest.mark.asyncio
    async def test_synthesize_to_file_resolves_style(self, monkeypatch, tmp_path):
        """端到端（无网络）：心情「生气」-> 豆包收到 angry 风格。"""
        monkeypatch.setattr(tts, "OUTPUT_DIR", tmp_path)
        from junjun_express.mood import mood_manager
        monkeypatch.setattr(mood_manager, "get_mood", lambda cid: "生气")
        captured = {}

        async def _doubao(text, *, style=None):
            captured["style"] = style
            return b"audio"
        monkeypatch.setattr(tts, "synthesize_doubao", _doubao)
        monkeypatch.setattr(tts, "_backend_configured", lambda b: b == "doubao")

        path = await tts._synthesize_to_file("哼，不理你了", "doubao", chat_id="c1")
        assert path is not None and path.exists()
        assert captured["style"]["style"] == "angry"
        assert captured["style"]["speech_rate"] > 0


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
    def test_full_style_params(self, monkeypatch):
        """emotion/scale/speech/loudness 全进 audio_params，超界钳制。"""
        req = _capture_start_session(
            monkeypatch, emotion="angry", emotion_scale=9,
            speech_rate=200, loudness_rate=-99)
        ap = req["audio_params"]
        assert ap["emotion"] == "angry"
        assert ap["emotion_scale"] == 5      # 9 钳到 5
        assert ap["speech_rate"] == 100      # 200 钳到 100
        assert ap["loudness_rate"] == -50    # -99 钳到 -50

    def test_no_emotion_no_param(self, monkeypatch):
        """不带情绪时 audio_params 无 emotion 键，速率为 0（保持旧行为）。"""
        req = _capture_start_session(monkeypatch)
        ap = req["audio_params"]
        assert "emotion" not in ap
        assert ap["speech_rate"] == 0 and ap["loudness_rate"] == 0
