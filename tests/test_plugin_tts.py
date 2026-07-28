"""tts 插件测试：命令 / 工具 / 多后端降级 / 截断 / 限流（不连真实网络）。"""

import asyncio
from types import SimpleNamespace

import pytest

from junjun_agent import commands
from junjun_agent.tasks import task_manager


async def _drain():
    """等待全部后台任务完成。"""
    tasks = list(task_manager._running.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:1:private")


def _meta(text):
    return SimpleNamespace(text=text, user_id="12345", nickname="甲", at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text, is_group=True):
    return commands.CommandContext(session=_session(is_group), meta=_meta(text),
                                   args=text.split(" ", 1)[1] if " " in text else "")


_FAKE_MP3 = b"\xff\xfb" + b"\x00" * 1024  # 假 mp3 字节


@pytest.fixture
def _plugin(monkeypatch, tmp_path):
    """导入插件并替换输出目录 / 限流表；四个后端全部配上假凭据。"""
    import junjun_skills.plugins.tts.tools as tts
    monkeypatch.setattr(tts, "OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("DOUBAO_TTS_API_KEY", "fake-doubao")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-sf")
    monkeypatch.setenv("TTS_GSV2P_TOKEN", "fake-gsv2p")
    monkeypatch.setenv("TTS_SOVITS_REF_AUDIO", r"E:\ref.wav")
    monkeypatch.setenv("TTS_SOVITS_PROMPT_TEXT", "参考文本")
    monkeypatch.delenv("TTS_DEFAULT_BACKEND", raising=False)
    tts._last_use.clear()
    return tts


def _patch_ok(monkeypatch, tts, backend, captured=None):
    """把指定后端的 synthesize helper 替换为成功假实现。"""

    async def _fake(text):
        if captured is not None:
            captured.append(text)
        return _FAKE_MP3

    monkeypatch.setattr(tts, f"synthesize_{backend}", _fake)


def _patch_none(monkeypatch, tts, backend):
    async def _none(text):
        return None

    monkeypatch.setattr(tts, f"synthesize_{backend}", _none)


class TestTTSCommand:
    @pytest.mark.asyncio
    async def test_success_sends_voice_and_file(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        _patch_ok(monkeypatch, tts, "doubao")
        result = await tts.tts_cmd(_ctx("/tts 你好世界"))
        assert "在合成" in result        # 提交即返回 ack
        await _drain()                    # 后台合成完直发
        assert len(_fake_gateway) == 1
        segs = _fake_gateway[0].segments
        assert segs[0].type == "voice"
        # 文件落盘且 voice 段 data 是该本地路径
        from pathlib import Path
        path = Path(segs[0].data)
        assert path.exists() and path.parent == tts.OUTPUT_DIR
        assert path.name.startswith("tts_doubao_") and path.suffix == ".mp3"
        assert path.read_bytes() == _FAKE_MP3

    @pytest.mark.asyncio
    async def test_backend_arg_parsed(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        _patch_ok(monkeypatch, tts, "siliconflow")
        await tts.tts_cmd(_ctx("/tts 你好 siliconflow"))
        await _drain()
        from pathlib import Path
        path = Path(_fake_gateway[0].segments[0].data)
        assert path.name.startswith("tts_siliconflow_")

    @pytest.mark.asyncio
    async def test_default_backend_from_env(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        monkeypatch.setenv("TTS_DEFAULT_BACKEND", "gsv2p")
        _patch_ok(monkeypatch, tts, "gsv2p")
        await tts.tts_cmd(_ctx("/tts 你好世界"))
        await _drain()
        from pathlib import Path
        path = Path(_fake_gateway[0].segments[0].data)
        assert path.name.startswith("tts_gsv2p_")

    @pytest.mark.asyncio
    async def test_fallback_to_next_backend(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        _patch_none(monkeypatch, tts, "doubao")   # 默认后端失败
        _patch_ok(monkeypatch, tts, "siliconflow")  # 降级到硅基流动
        result = await tts.tts_cmd(_ctx("/tts 你好世界"))
        assert "在合成" in result
        await _drain()
        from pathlib import Path
        path = Path(_fake_gateway[0].segments[0].data)
        assert path.name.startswith("tts_siliconflow_")

    @pytest.mark.asyncio
    async def test_all_backends_fail_friendly_text(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        for b in tts.BACKENDS:
            _patch_none(monkeypatch, tts, b)
        result = await tts.tts_cmd(_ctx("/tts 你好世界"))
        assert "在合成" in result          # 先回 ack
        await _drain()                      # 后台失败发降级文案
        assert "失败" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_no_backend_configured_degrades(self, _plugin, monkeypatch):
        tts = _plugin
        for key in ("DOUBAO_TTS_API_KEY", "SILICONFLOW_API_KEY", "TTS_GSV2P_TOKEN",
                    "TTS_SOVITS_REF_AUDIO", "TTS_SOVITS_PROMPT_TEXT"):
            monkeypatch.delenv(key, raising=False)
        result = await tts.tts_cmd(_ctx("/tts 你好世界"))
        assert "没配置" in result

    @pytest.mark.asyncio
    async def test_long_text_truncated(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        captured = []
        _patch_ok(monkeypatch, tts, "doubao", captured)
        await tts.tts_cmd(_ctx(f"/tts {'啊' * 400}"))
        await _drain()
        assert len(captured[0]) == 300

    @pytest.mark.asyncio
    async def test_rate_limit(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        _patch_ok(monkeypatch, tts, "doubao")
        await tts.tts_cmd(_ctx("/tts 你好"))
        await _drain()
        result = await tts.tts_cmd(_ctx("/tts 你好"))
        assert "秒" in result
        await _drain()
        assert len(_fake_gateway) == 1  # 第二次没发语音

    @pytest.mark.asyncio
    async def test_empty_args_usage(self, _plugin):
        tts = _plugin
        result = await tts.tts_cmd(_ctx("/tts"))
        assert "用法" in result

    def test_voice_alias_registered(self, _plugin):
        # 模块在先前测试已 import 过，需 reload 触发重新注册到命令总线
        import importlib
        importlib.reload(_plugin)
        names = {c["name"] for c in commands.list_commands()}
        assert "tts" in names
        matched = commands._match("/voice 你好")
        assert matched is not None and matched[0].name == "tts"
        assert matched[0].plugin == "tts"


class TestUnifiedTTSTool:
    @pytest.mark.asyncio
    async def test_tool_sends_voice(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        _patch_ok(monkeypatch, tts, "doubao")
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:999:group")
        try:
            out = await tts.unified_tts.ainvoke({"text": "你好世界"})
        finally:
            current_chat_id.reset(token)
        assert "在合成" in out       # 立即返回 ack
        await _drain()                # 后台合成完直发
        rs = _fake_gateway[0]
        assert rs.target_group_id == "999"
        assert rs.segments[0].type == "voice"

    @pytest.mark.asyncio
    async def test_tool_backend_arg(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        _patch_ok(monkeypatch, tts, "sovits")
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:1:private")
        try:
            out = await tts.unified_tts.ainvoke({"text": "你好", "backend": "sovits"})
        finally:
            current_chat_id.reset(token)
        assert "在合成" in out
        await _drain()
        from pathlib import Path
        path = Path(_fake_gateway[0].segments[0].data)
        assert path.name.startswith("tts_sovits_") and path.suffix == ".wav"
        assert _fake_gateway[0].target_user_id == "1"

    @pytest.mark.asyncio
    async def test_tool_no_backend_configured(self, _plugin, monkeypatch):
        tts = _plugin
        for key in ("DOUBAO_TTS_API_KEY", "SILICONFLOW_API_KEY", "TTS_GSV2P_TOKEN",
                    "TTS_SOVITS_REF_AUDIO", "TTS_SOVITS_PROMPT_TEXT"):
            monkeypatch.delenv(key, raising=False)
        out = await tts.unified_tts.ainvoke({"text": "test"})
        assert "未配置" in out

    @pytest.mark.asyncio
    async def test_tool_all_fail_degrades(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        for b in tts.BACKENDS:
            _patch_none(monkeypatch, tts, b)
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:999:group")
        try:
            out = await tts.unified_tts.ainvoke({"text": "test"})
        finally:
            current_chat_id.reset(token)
        assert "在合成" in out           # 先回 ack
        await _drain()                    # 后台失败发降级文案
        assert "失败" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_tool_rate_limit(self, _fake_gateway, _plugin, monkeypatch):
        tts = _plugin
        _patch_ok(monkeypatch, tts, "doubao")
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:999:group")
        try:
            await tts.unified_tts.ainvoke({"text": "第一句"})
            await _drain()
            out = await tts.unified_tts.ainvoke({"text": "第二句"})
        finally:
            current_chat_id.reset(token)
        assert "秒" in out
        await _drain()
        assert len(_fake_gateway) == 1

    def test_tool_name_registered(self, _plugin):
        tts = _plugin
        assert tts.TOOLS == [tts.unified_tts]
        assert tts.unified_tts.name == "unified_tts"


class TestDoubaoSpeakerRouting:
    """豆包音色按语言路由（2026-07-22 用户指定）：日语 ja_*，中文 zh_female_vv_*。"""

    def test_japanese_text_uses_ja_voice(self, monkeypatch):
        monkeypatch.delenv("TTS_DOUBAO_SPEAKER", raising=False)
        from junjun_skills.plugins.tts import tools as tts
        assert tts._doubao_speaker_for("こんにちは、元気ですか") == "ja_female_bv521_uranus_bigtts"
        assert tts._doubao_speaker_for("カタカナテスト") == "ja_female_bv521_uranus_bigtts"

    def test_chinese_text_uses_zh_voice(self, monkeypatch):
        monkeypatch.delenv("TTS_DOUBAO_SPEAKER", raising=False)
        from junjun_skills.plugins.tts import tools as tts
        assert tts._doubao_speaker_for("今天天气真好") == "zh_female_vv_uranus_bigtts"
        assert tts._doubao_speaker_for("hello world") == "zh_female_vv_uranus_bigtts"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("TTS_DOUBAO_SPEAKER", "custom_voice")
        from junjun_skills.plugins.tts import tools as tts
        assert tts._doubao_speaker_for("こんにちは") == "custom_voice"
