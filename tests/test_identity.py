"""Identity Core 自我模型（P6-3）：蒸馏/加权衰减/归档/注入块/重置。"""

import time

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m
from junjun_express import identity as idm


@pytest.fixture
def env(monkeypatch, tmp_path):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"identity": {"enable": True, "interval_days": 7,
                          "min_diaries": 3, "max_entries": 8}})
    monkeypatch.setattr(idm, "_STATE_PATH", tmp_path / "identity_state.json")
    db = SqliteDatabase(":memory:")
    with db.bind_ctx([m.SelfIdentity, m.DiaryEntry]):
        db.create_tables([m.SelfIdentity, m.DiaryEntry])
        yield monkeypatch
    cfg_mod.global_config = old


def _seed_diaries(n=3, days_ago=1):
    now = time.time()
    for i in range(n):
        m.DiaryEntry.create(date=f"2026-08-0{i + 1}",
                            content=f"今天又被他们逗得好开心，聊了火锅和游戏 {i}",
                            mood="开心", created_at=now - days_ago * 86400)


class _FakeModel:
    def __init__(self, text):
        self._text = text

    async def ainvoke(self, msgs, config=None):
        return type("R", (), {"content": self._text})()


_GOOD_JSON = """[
  {"category": "我喜欢", "content": "被他们逗开心的时候"},
  {"category": "最近在乎", "content": "群里每个人的事"},
  {"category": "无效分类", "content": "会被过滤"},
  {"category": "我们的梗"}
]"""


class TestParse:
    def test_parse_good(self):
        entries = idm._parse_entries(_GOOD_JSON)
        assert len(entries) == 2  # 无效分类和缺 content 的被过滤
        assert entries[0]["category"] == "我喜欢"

    def test_parse_garbage(self):
        assert idm._parse_entries("我不会输出 JSON") == []
        assert idm._parse_entries('{"not": "array"}') == []


class TestUpsert:
    def test_add_confirm_decay_archive(self, env):
        now = time.time()
        # 已有两条：一条会被确认，一条不被确认且权重低 -> 归档
        old1 = m.SelfIdentity.create(category="我喜欢", content="被他们逗开心的时候",
                                     weight=1.0, seen_count=2, created_at=now, updated_at=now)
        old2 = m.SelfIdentity.create(category="我看不惯", content="冷场",
                                     weight=0.55, seen_count=1, created_at=now, updated_at=now)
        added, confirmed, archived = idm._upsert_entries([
            {"category": "我喜欢", "content": "被他们逗开心的时候"},  # 确认 old1
            {"category": "最近在乎", "content": "群里每个人的事"},     # 新增
        ])
        assert (added, confirmed, archived) == (1, 1, 1)
        old1 = m.SelfIdentity.get_by_id(old1.id)
        assert old1.seen_count == 3 and old1.weight == pytest.approx(1.2)
        old2 = m.SelfIdentity.get_by_id(old2.id)
        assert old2.archived is True  # 0.55*0.85=0.4675 < 0.5
        active = idm.get_entries(limit=10)
        assert len(active) == 2  # old1 + 新增


class TestDistill:
    @pytest.mark.asyncio
    async def test_distill_creates_entries(self, env):
        _seed_diaries(3)
        n = await idm.distill(model=_FakeModel(_GOOD_JSON), force=True)
        assert n == 2
        rows = idm.get_entries()
        assert any("被他们逗开心" in r.content for r in rows)
        # 状态已落盘（last_distill 推进）
        assert idm._load_state()["last_distill"] > 0

    @pytest.mark.asyncio
    async def test_bad_output_keeps_old(self, env):
        _seed_diaries(3)
        m.SelfIdentity.create(category="我喜欢", content="旧条目",
                              weight=1.0, seen_count=1)
        n = await idm.distill(model=_FakeModel("胡说八道"), force=True)
        assert n == 0
        assert [r.content for r in idm.get_entries()] == ["旧条目"]

    @pytest.mark.asyncio
    async def test_not_enough_diaries(self, env):
        _seed_diaries(2)  # min_diaries=3

        async def _boom(*a, **kw):
            raise AssertionError("不该调用模型")
        n = await idm.distill(model=_FakeModel(_GOOD_JSON))
        assert n == 0

    @pytest.mark.asyncio
    async def test_tick_cadence(self, env):
        _seed_diaries(3)
        calls = []

        async def _distill(**kw):
            calls.append(1)
            return 0
        env.setattr(idm, "distill", _distill)
        await idm.identity_tick()
        assert calls  # 首次（无状态）会蒸
        # 状态推进后间隔未到不再蒸
        idm._save_state({"last_distill": time.time()})
        await idm.identity_tick()
        assert len(calls) == 1


class TestBlockAndReset:
    def test_block_with_constitution_framing(self, env):
        m.SelfIdentity.create(category="我喜欢", content="热闹", weight=1.0, seen_count=3)
        block = idm.build_identity_block()
        assert "热闹" in block and "宪法" in block

    def test_block_empty(self, env):
        assert idm.build_identity_block() == ""

    def test_reset_archives_all(self, env):
        for i in range(3):
            m.SelfIdentity.create(category="我喜欢", content=f"条目{i}",
                                  weight=1.0, seen_count=1)
        assert idm.reset_identity() == 3
        assert idm.get_entries() == []
        assert idm.build_identity_block() == ""
