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

    def test_negative_mood_soft_tone_directive(self):
        """负面情绪只调语气，不压人格（2026-08-04 情绪卡死「无语」事件：
        旧版「不想说话/不想折腾工具」把温柔和主动行为全压没，还抑制工具意愿）。"""
        block = self._mgr("有点无语").build_mood_block("qq:1:group")
        assert "有点无语" in block
        assert "话少一点" in block
        assert "依然会认真回应" in block   # 别人需要时依然在
        assert "不想折腾工具" not in block  # 旧版压工具意愿的表述已移除

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


class TestMoodEvalPrompt:
    """情绪重评 prompt 防锚定（2026-08-04 实锤：旧 prompt 示例里写着「有点无语」，
    模型直接抄示例，全局心境卡死在无语）。"""

    def test_no_anchor_examples(self):
        from junjun_express.mood import _EVAL_PROMPT
        assert "有点无语" not in _EVAL_PROMPT
        assert "如：" not in _EVAL_PROMPT

    def test_eval_target_is_bot_not_group_vibe(self):
        from junjun_express.mood import _EVAL_PROMPT
        assert "不是群聊氛围" in _EVAL_PROMPT   # 群聊吵闹 ≠ 我无语
        assert "平静" in _EVAL_PROMPT           # 无明确信号往平静回落


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, msgs, config=None):
        return _FakeResp(self._content)


class TestSelfMoodProtection:
    """注意：set_self_mood 会写 SelfMood 表——必须用内存库 bind_ctx，
    绝不落生产库（2026-08-04 教训：测试直写 prod selfmood 连带污染
    resolve_emotion，test_ja_tts_mix 的桩被 style_kw 打爆）。"""

    def _mem_db(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        return db, m

    @pytest.mark.asyncio
    async def test_calm_not_promoted_to_self_mood(self, tmp_path):
        """别群评出的「平静」不许冲掉这边真实沉淀的全局心境。"""
        db, m = self._mem_db(tmp_path)
        with db.bind_ctx([m.SelfMood]):
            db.create_tables([m.SelfMood])
            mgr = MoodManager()
            mgr.set_self_mood("被夸了很得意", reason="c1")
            await mgr.evaluate("c2", "普通水群对话", model=_FakeModel("平静"))
            assert mgr.get_mood("c2") == "平静"
            assert mgr.get_self_mood() == "被夸了很得意"   # 没冲掉

    @pytest.mark.asyncio
    async def test_real_emotion_promoted(self, tmp_path):
        """非平静的真实情绪照常塑造全局心境。"""
        db, m = self._mem_db(tmp_path)
        with db.bind_ctx([m.SelfMood]):
            db.create_tables([m.SelfMood])
            mgr = MoodManager()
            await mgr.evaluate("c1", "大家都在夸君君", model=_FakeModel("被夸了很得意"))
            assert mgr.get_self_mood() == "被夸了很得意"


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
        meta = SimpleNamespace(text="@你 [回复「鹤」: 今晚开黑吗] 不来", image_urls=None)
        await _build_memory_block(session, meta)
        assert "[回复" not in captured["query"]
        assert "@你" not in captured["query"]
        assert "不来" in captured["query"]
