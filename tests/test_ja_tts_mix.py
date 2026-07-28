"""ja_tts 中日混读处理测试：翻译判定 / ensure_japanese / mix 开关 / 豆包路由。"""

import pytest

from junjun_skills.plugins.ja_tts import tools as ja


class TestNeedsTranslation:
    def test_pure_japanese_no_translate(self):
        assert ja.needs_translation("こんにちは、元気ですか") is False

    def test_pure_chinese_translates(self):
        assert ja.needs_translation("今天天气真好") is True

    def test_mixed_with_simplified_marker_translates(self):
        # 有假名但夹简体特征字（这/们/说）-> 混合，需要翻译
        assert ja.needs_translation("こんにちは，这是测试") is True
        assert ja.needs_translation("我们要不要一起去，楽しみ") is True

    def test_japanese_kanji_no_translate(self):
        # 日语汉字（学生/先生）不含简体特征字 -> 不翻
        assert ja.needs_translation("私は学生です") is False

    def test_empty_and_ascii(self):
        assert ja.needs_translation("") is False
        assert ja.needs_translation("hello") is False


class TestEnsureJapanese:
    @pytest.mark.asyncio
    async def test_chinese_translated(self, monkeypatch):
        captured = {}

        class _Model:
            async def ainvoke(self, msgs):
                captured["prompt"] = msgs[0].content
                class R: content = "今日はいい天気ですね"
                return R()

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: _Model())
        out = await ja.ensure_japanese("今天天气真好")
        assert out == "今日はいい天気ですね"
        assert "今天天气真好" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_japanese_passthrough_no_llm(self, monkeypatch):
        def _boom(slot):
            raise AssertionError("纯日语不应调 LLM")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        assert await ja.ensure_japanese("こんにちは") == "こんにちは"

    @pytest.mark.asyncio
    async def test_translate_failure_keeps_original(self, monkeypatch):
        def _boom(slot):
            raise RuntimeError("模型槽未配置")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        assert await ja.ensure_japanese("今天天气真好") == "今天天气真好"


class TestMixSwitch:
    @pytest.mark.asyncio
    async def test_mix_skips_translation(self, monkeypatch, tmp_path):
        """mix=True 时中文直接送合成，不翻译。"""
        def _boom(slot):
            raise AssertionError("mix 模式不应调翻译")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        monkeypatch.setattr(ja, "OUTPUT_DIR", tmp_path)
        captured = {}

        async def _fake_synth(text, speaker=""):
            captured["text"] = text
            return b"\xff\xfb" + b"\x00" * 64

        monkeypatch.setattr(ja, "synthesize", _fake_synth)
        path = await ja._synthesize_to_file("你好，こんにちは", ja.VOICE_PRESETS["ja"], mix=True)
        assert path is not None
        assert "你好" in captured["text"]  # 原文直接合成

    @pytest.mark.asyncio
    async def test_default_translates_for_ja_voice(self, monkeypatch, tmp_path):
        """默认（mix=False）日语音色先翻译再合成。"""
        class _Model:
            async def ainvoke(self, msgs):
                class R: content = "こんにちは、元気？"
                return R()

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: _Model())
        monkeypatch.setattr(ja, "OUTPUT_DIR", tmp_path)
        captured = {}

        async def _fake_synth(text, speaker=""):
            captured["text"] = text
            return b"\xff\xfb" + b"\x00" * 64

        monkeypatch.setattr(ja, "synthesize", _fake_synth)
        path = await ja._synthesize_to_file("你好", ja.VOICE_PRESETS["ja"], mix=False)
        assert path is not None
        assert captured["text"] == "こんにちは、元気？"

    @pytest.mark.asyncio
    async def test_zh_voice_no_translation(self, monkeypatch, tmp_path):
        """中文音色不翻译。"""
        def _boom(slot):
            raise AssertionError("中文音色不应调翻译")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        monkeypatch.setattr(ja, "OUTPUT_DIR", tmp_path)
        captured = {}

        async def _fake_synth(text, speaker=""):
            captured["text"] = text
            return b"\xff\xfb" + b"\x00" * 64

        monkeypatch.setattr(ja, "synthesize", _fake_synth)
        path = await ja._synthesize_to_file("你好呀", ja.VOICE_PRESETS["vv"], mix=False)
        assert path is not None
        assert captured["text"] == "你好呀"


class TestCmdMixPrefix:
    def test_parse_mix_prefix(self):
        """命令层「混合」前缀解析（与 _parse_args 组合）。"""
        text, speaker = ja._parse_args("混合 你好世界 ja")
        mix = False
        for prefix in ("混合 ", "mix ", "混读 "):
            if text.startswith(prefix):
                mix = True
                text = text[len(prefix):].strip()
                break
        assert mix is True and text == "你好世界"
        assert speaker == ja.VOICE_PRESETS["ja"]
