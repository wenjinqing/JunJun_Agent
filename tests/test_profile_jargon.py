"""用户画像 + 黑话单测（内存 SQLite）。"""


import pytest
from peewee import SqliteDatabase

from junjun_memory.user_profile import UserProfileStore, make_person_id


@pytest.fixture(autouse=True)
def _memory_db(monkeypatch):
    """peewee 表绑定到内存库，测试隔离。"""
    import junjun_core.database.models as m
    test_db = SqliteDatabase(":memory:")
    tables = m.ALL_TABLES
    with test_db.bind_ctx(tables):
        test_db.create_tables(tables)
        monkeypatch.setattr(m, "db", test_db)
        import junjun_core.database as pkg
        monkeypatch.setattr(pkg, "db", test_db)
        yield test_db


class TestUserProfile:
    def test_person_id_deterministic(self):
        assert make_person_id("qq", "111") == make_person_id("qq", "111")
        assert make_person_id("qq", "111") != make_person_id("qq", "222")

    def test_add_point_and_get(self):
        store = UserProfileStore()
        store.add_point("qq", "111", "喜好", "爱吃火锅", 0.8, nickname="甲")
        points = store.get_points("qq", "111")
        assert len(points) == 1
        assert points[0]["category"] == "喜好"
        assert points[0]["content"] == "爱吃火锅"

    def test_duplicate_point_reinforces_not_duplicates(self):
        store = UserProfileStore()
        store.add_point("qq", "111", "喜好", "爱吃火锅", 0.8)
        store.add_point("qq", "111", "喜好", "爱吃火锅", 0.8)
        points = store.get_points("qq", "111")
        assert len(points) == 1
        assert points[0]["weight"] > 0.8  # 重复提及强化

    def test_field_level_merge_keeps_other_points(self):
        store = UserProfileStore()
        store.add_point("qq", "111", "喜好", "爱吃火锅", 0.8)
        store.add_point("qq", "111", "身份", "程序员", 0.9)
        points = store.get_points("qq", "111")
        assert len(points) == 2

    def test_max_points_evicts_lowest_weight(self):
        from junjun_memory.user_profile import MAX_POINTS
        store = UserProfileStore()
        for i in range(MAX_POINTS + 5):
            store.add_point("qq", "111", "杂项", f"内容{i}", 0.5 + (i % 10) * 0.01)
        points = store.get_points("qq", "111", top_k=100)
        assert len(points) == MAX_POINTS

    def test_relation_block_format(self):
        store = UserProfileStore()
        store.add_point("qq", "111", "喜好", "爱吃火锅", 0.9, nickname="甲")
        store.set_name("qq", "111", "老甲")
        block = store.build_relation_block("qq", "111")
        assert "老甲" in block
        assert "爱吃火锅" in block

    def test_unknown_user_empty_block(self):
        store = UserProfileStore()
        assert store.build_relation_block("qq", "999999") == ""


class TestJargon:
    def test_record_and_lookup(self):
        from junjun_express.jargon import record_jargon, lookup_jargon
        record_jargon("awsl", "啊我死了，表示可爱到极点")
        assert "啊我死了" in lookup_jargon("awsl")

    def test_count_accumulates(self):
        from junjun_express.jargon import record_jargon
        from junjun_core.database import Jargon
        record_jargon("yyds", "永远的神")
        record_jargon("yyds", "")
        row = Jargon.get(Jargon.term == "yyds")
        assert row.count == 2
        assert row.explanation == "永远的神"  # 空解释不覆盖

    def test_match_requires_min_count(self):
        from junjun_express.jargon import record_jargon, match_jargon_from_text
        record_jargon("xswl", "笑死我了")
        assert match_jargon_from_text("今天xswl") == []  # count=1 不可信
        record_jargon("xswl", "笑死我了")
        hits = match_jargon_from_text("今天xswl")
        assert hits and hits[0]["term"] == "xswl"

    def test_jargon_block_empty_when_no_hit(self):
        from junjun_express.jargon import build_jargon_block
        assert build_jargon_block("普通的一句话") == ""


class TestMemorySkills:
    @pytest.mark.asyncio
    async def test_manage_user_profile_skill(self):
        from junjun_skills.builtin.memory_skills import manage_user_profile, current_platform
        current_platform.set("qq")
        out = manage_user_profile.invoke({"user_id": "333", "category": "称呼", "content": "叫他老王"})
        assert "已更新" in out
        from junjun_memory.user_profile import get_profile_store
        points = get_profile_store().get_points("qq", "333")
        assert points and points[0]["content"] == "叫他老王"

    @pytest.mark.asyncio
    async def test_query_jargon_skill(self):
        from junjun_express.jargon import record_jargon
        from junjun_skills.builtin.memory_skills import query_jargon
        record_jargon("绝绝子", "太绝了")
        out = query_jargon.invoke({"term": "绝绝子"})
        assert "太绝了" in out
        out2 = query_jargon.invoke({"term": "不存在的词xyz"})
        assert "没有" in out2

    @pytest.mark.asyncio
    async def test_recall_memory_skill_no_data(self, monkeypatch, tmp_path):
        import junjun_memory.long_term as lt_mod
        from junjun_memory.long_term import LongTermMemory
        monkeypatch.setattr(lt_mod, "_ltm", LongTermMemory(data_dir=tmp_path))
        from junjun_skills.builtin.memory_skills import recall_memory
        out = await recall_memory.ainvoke({"query": "从没聊过的事"})
        assert "没有找到" in out
