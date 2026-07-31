"""自我心境持久化 + 私人日记测试。

DB 用内存库 bind_ctx 隔离；LLM / 长期记忆 / 素材收集全部 monkeypatch。
"""

import time
from types import SimpleNamespace

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m

# 隔离用内存库：bind_ctx 直接绑真实 db 会把测试数据写进 data/junjun.db，
# 假日记会顶掉今晚的真日记（生产 diary_tick 发现当天已有条目会跳过）
test_db = SqliteDatabase(":memory:")


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


@pytest.fixture
def _cfg():
    old = cfg_mod.global_config
    yield
    cfg_mod.global_config = old


class TestSelfMood:
    def test_persist_and_read(self, _cfg):
        """set_self_mood 落库，重载 manager 后能读回（重启不丢）。"""
        _set_config({"mood": {"enable_mood": True, "self_mood_hours": 12}})
        with test_db.bind_ctx([m.SelfMood]):
            test_db.create_tables([m.SelfMood])
            from junjun_express.mood import MoodManager
            mm = MoodManager()
            mm.set_self_mood("有点小开心", reason="qq:1:group")
            # 换一个全新 manager（模拟重启），从 DB 读回
            mm2 = MoodManager()
            assert mm2.get_self_mood() == "有点小开心"

    def test_stale_self_mood_regresses(self, _cfg):
        """超过 self_mood_hours 视为回到平静。"""
        _set_config({"mood": {"enable_mood": True, "self_mood_hours": 1}})
        with test_db.bind_ctx([m.SelfMood]):
            test_db.create_tables([m.SelfMood])
            from junjun_express.mood import MoodManager
            mm = MoodManager()
            mm.set_self_mood("郁闷")
            mm._self_mood.updated_at = time.time() - 2 * 3600
            assert mm.get_self_mood() == "平静"

    def test_mood_block_appends_self_mood(self, _cfg):
        """会话情绪与全局心境不同时，mood block 带整体心境行。"""
        _set_config({"mood": {"enable_mood": True, "self_mood_hours": 12}})
        with test_db.bind_ctx([m.SelfMood]):
            test_db.create_tables([m.SelfMood])
            from junjun_express.mood import MoodManager
            mm = MoodManager()
            mm.set_mood("c1", "开心")
            mm.set_self_mood("有点想某人")
            block = mm.build_mood_block("c1")
            assert "你当前的情绪：开心" in block
            assert "整体心境：有点想某人" in block


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, msgs, config=None):
        return _FakeResp(self._content)


class TestDiary:
    @pytest.mark.asyncio
    async def test_write_diary_full_chain(self, _cfg, monkeypatch):
        """写日记：LLM 产出解析 -> 落库 -> 进长期记忆 -> 心情沉淀为自我心境。"""
        _set_config({"diary": {"enable": True}, "mood": {"enable_mood": True}})
        with test_db.bind_ctx([m.DiaryEntry, m.SelfMood]):
            test_db.create_tables([m.DiaryEntry, m.SelfMood])
            from junjun_express import diary as d
            from junjun_express.mood import MoodManager, mood_manager

            monkeypatch.setattr(d, "_gather_material", lambda day: "我今天说过的话：大家好")
            indexed = []

            async def _fake_index(day, content):
                indexed.append((day, content))

            monkeypatch.setattr(d, "_index_to_memory", _fake_index)

            # 隔离 mood_manager 的 self 状态（测试间互不污染）
            old_self = mood_manager._self_mood
            mood_manager._self_mood = None
            try:
                model = _FakeModel("今天群里好热闹，白菜兔又来讲冷笑话，笑得我。\n心情：开心")
                content = await d.write_diary(model=model, force=True)
                assert content and "白菜兔" in content
                row = d._get_entry(d._today())
                assert row is not None and row.mood == "开心"
                assert indexed and "白菜兔" in indexed[0][1]
                assert mood_manager.get_self_mood() == "开心"
            finally:
                mood_manager._self_mood = old_self

    @pytest.mark.asyncio
    async def test_skip_when_exists(self, _cfg, monkeypatch):
        """今天已写过且非 force -> 跳过。"""
        _set_config({"diary": {"enable": True}})
        with test_db.bind_ctx([m.DiaryEntry]):
            test_db.create_tables([m.DiaryEntry])
            from junjun_express import diary as d
            d._save_entry(d._today(), "旧日记", "平静")
            called = []
            monkeypatch.setattr(d, "_gather_material",
                                lambda day: called.append(1) or "素材")
            assert await d.write_diary(model=_FakeModel("x")) is None
            assert not called

    @pytest.mark.asyncio
    async def test_disabled_skips(self, _cfg):
        _set_config({"diary": {"enable": False}})
        with test_db.bind_ctx([m.DiaryEntry]):
            test_db.create_tables([m.DiaryEntry])
            from junjun_express import diary as d
            assert await d.write_diary(model=_FakeModel("x")) is None

    def test_parse_output(self):
        from junjun_express import diary as d
        content, mood = d._parse_diary_output("正文第一行\n正文第二行\n心情：有点累")
        assert content == "正文第一行\n正文第二行"
        assert mood == "有点累"
        content2, mood2 = d._parse_diary_output("只有正文")
        assert content2 == "只有正文" and mood2 == ""

    @pytest.mark.asyncio
    async def test_tick_time_gate(self, _cfg, monkeypatch):
        """diary_tick：未到点不写；到点且没写过才写。"""
        _set_config({"diary": {"enable": True, "time": "23:59"}})
        with test_db.bind_ctx([m.DiaryEntry]):
            test_db.create_tables([m.DiaryEntry])
            m.DiaryEntry.delete().execute()  # 内存库跨用例存活，清空今天可能存在的条目
            from junjun_express import diary as d
            writes = []

            async def _fake_write(**kw):
                writes.append(1)

            monkeypatch.setattr(d, "write_diary", _fake_write)
            await d.diary_tick()
            assert not writes  # 23:59 未到（除非恰好在 23:59 跑测试）
            _set_config({"diary": {"enable": True, "time": "00:00"}})
            await d.diary_tick()
            assert writes  # 00:00 必已过
