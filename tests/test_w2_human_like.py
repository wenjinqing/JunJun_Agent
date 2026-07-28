"""W2 拟人行为测试：情绪行为指令 + 阅读延迟 + 记忆检索查询清洗。"""

import random

import pytest

from junjun_express.mood import MoodManager


class TestMoodBehaviorBlock:
    def _mgr(self, state):
        mgr = MoodManager()
        mgr._moods["qq:1:group"] = mgr._moods.setdefault("qq:1:group")
        from junjun_express.mood import ChatMood
        mgr._moods["qq:1:group"] = ChatMood(state=state)
        return mgr

    def test_negative_mood_short_reply_directive(self):
        block = self._mgr("有点无语").build_mood_block("qq:1:group")
        assert "有点无语" in block
        assert "回复尽量短" in block
        assert "不要主动发表情包" in block

    def test_positive_mood_active_directive(self):
        block = self._mgr("被夸了很得意").build_mood_block("qq:1:group")
        assert "活泼" in block
        assert "主动发表情包/语音" in block

    def test_neutral_mood_tone_only(self):
        block = self._mgr("平静").build_mood_block("qq:1:group")
        assert "语气自然反映" in block
        assert "回复尽量短" not in block

    def test_empty_when_disabled(self, monkeypatch):
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"mood": {"enable_mood": False}}))
        assert self._mgr("开心").build_mood_block("qq:1:group") == ""


class TestReadingDelay:
    def test_first_piece_has_reading_delay(self):
        from junjun_agent.postprocess import process_response
        out = process_response("好的呀", incoming="你觉得明天天气怎么样要不要出去玩",
                               rand=random.Random(42))
        assert out and out[0].delay > 0.3  # 有阅读延迟

    def test_disabled_when_config_off(self, monkeypatch):
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"response_timing": {"enable": False}}))
        from junjun_agent.postprocess import process_response
        out = process_response("好的呀", incoming="很长很长的消息" * 50, rand=random.Random(42))
        assert out and out[0].delay == 0.0

    def test_delay_capped(self):
        from junjun_agent.postprocess import process_response
        out = process_response("好", incoming="长" * 10000, rand=random.Random(42))
        assert out[0].delay <= 4.0 * 1.3 + 0.01  # cap * 抖动上限


class TestMemoryQueryCleaning:
    @pytest.mark.asyncio
    async def test_query_stripped_of_placeholders(self, monkeypatch):
        """[回复...]/@前缀 不进检索查询。"""
        captured = {}

        class _FakeMem:
            async def search(self, query, *, top_k, chat_id):
                captured["query"] = query
                return []

        monkeypatch.setattr("junjun_memory.long_term.get_long_term_memory", lambda: _FakeMem())
        from types import SimpleNamespace
        from junjun_agent.processor import _build_memory_block
        session = SimpleNamespace(chat_id="qq:1:group")
        meta = SimpleNamespace(text="@你 [回复 鹤: 今晚开黑吗] 不来", image_urls=None)
        await _build_memory_block(session, meta)
        assert "[回复" not in captured["query"]
        assert "@你" not in captured["query"]
        assert "不来" in captured["query"]
