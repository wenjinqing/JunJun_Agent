"""ja_tts 插件测试：命令 / 工具 / 降级 / 截断 / 限流（不连真实 WS）。"""

from types import SimpleNamespace

import pytest

from junjun_agent import commands


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
    """导入插件并替换输出目录 / 限流表 / API key。"""
    import junjun_skills.plugins.ja_tts.tools as ja
    monkeypatch.setattr(ja, "OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("DOUBAO_TTS_API_KEY", "fake-key")
    ja._last_use.clear()
    return ja


def _patch_synth(monkeypatch, ja, captured=None):
    async def _fake(text, speaker=""):
        if captured is not None:
            captured.append((text, speaker))
        return _FAKE_MP3

    monkeypatch.setattr(ja, "synthesize", _fake)


class TestJaTTSCommand:
    @pytest.mark.asyncio
    async def test_synth_success_sends_voice_and_file(self, _fake_gateway, _plugin, monkeypatch):
        ja = _plugin
        _patch_synth(monkeypatch, ja)
        result = await ja.ja_tts_cmd(_ctx("/ja_tts こんにちは"))
        assert result is None
        assert len(_fake_gateway) == 1
        segs = _fake_gateway[0].segments
        assert segs[0].type == "voice"
        # 文件落盘且 voice 段 data 是该本地路径
        from pathlib import Path
        path = Path(segs[0].data)
        assert path.exists() and path.parent == ja.OUTPUT_DIR
        assert path.name.startswith("ja_") and path.suffix == ".mp3"
        assert path.read_bytes() == _FAKE_MP3

    @pytest.mark.asyncio
    async def test_voice_arg_parsed(self, _fake_gateway, _plugin, monkeypatch):
        ja = _plugin
        captured = []
        _patch_synth(monkeypatch, ja, captured)
        await ja.ja_tts_cmd(_ctx("/ja_tts おはよう vv"))
        assert captured[0][1] == ja.VOICE_PRESETS["vv"]

    @pytest.mark.asyncio
    async def test_no_api_key_degrades(self, _plugin, monkeypatch):
        ja = _plugin
        monkeypatch.delenv("DOUBAO_TTS_API_KEY", raising=False)
        result = await ja.ja_tts_cmd(_ctx("/ja_tts こんにちは"))
        assert "DOUBAO_TTS_API_KEY" in result

    @pytest.mark.asyncio
    async def test_long_text_truncated(self, _fake_gateway, _plugin, monkeypatch):
        ja = _plugin
        captured = []
        _patch_synth(monkeypatch, ja, captured)
        long_text = "あ" * 400
        await ja.ja_tts_cmd(_ctx(f"/ja_tts {long_text}"))
        # 送进合成的不超过 300 字（罗马字转换只对汉字生效，假名原文保留）
        assert len(captured[0][0]) <= 300

    @pytest.mark.asyncio
    async def test_synth_failure_degrades(self, _plugin, monkeypatch):
        ja = _plugin

        async def _none(text, speaker=""):
            return None

        monkeypatch.setattr(ja, "synthesize", _none)
        result = await ja.ja_tts_cmd(_ctx("/ja_tts こんにちは"))
        assert "失败" in result

    @pytest.mark.asyncio
    async def test_rate_limit(self, _fake_gateway, _plugin, monkeypatch):
        ja = _plugin
        _patch_synth(monkeypatch, ja)
        await ja.ja_tts_cmd(_ctx("/ja_tts こんにちは"))
        result = await ja.ja_tts_cmd(_ctx("/ja_tts こんにちは"))
        assert "秒" in result
        assert len(_fake_gateway) == 1  # 第二次没发语音

    @pytest.mark.asyncio
    async def test_empty_args_usage(self, _plugin):
        ja = _plugin
        result = await ja.ja_tts_cmd(_ctx("/ja_tts"))
        assert "用法" in result


class TestJaTTSTool:
    @pytest.mark.asyncio
    async def test_tool_sends_voice(self, _fake_gateway, _plugin, monkeypatch):
        ja = _plugin
        _patch_synth(monkeypatch, ja)
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:999:group")
        try:
            out = await ja.ja_tts_tool.ainvoke({"text": "こんにちは"})
        finally:
            current_chat_id.reset(token)
        assert "已发送" in out
        rs = _fake_gateway[0]
        assert rs.target_group_id == "999"
        assert rs.segments[0].type == "voice"

    @pytest.mark.asyncio
    async def test_tool_no_key_degrades(self, _plugin, monkeypatch):
        ja = _plugin
        monkeypatch.delenv("DOUBAO_TTS_API_KEY", raising=False)
        out = await ja.ja_tts_tool.ainvoke({"text": "test"})
        assert "未配置" in out

    @pytest.mark.asyncio
    async def test_tool_name_registered(self, _plugin):
        ja = _plugin
        assert ja.TOOLS == [ja.ja_tts_tool]
        assert ja.ja_tts_tool.name == "ja_tts"
