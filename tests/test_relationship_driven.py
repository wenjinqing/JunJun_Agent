"""P2-19 关系驱动行为测试：好感度 -> 语气行为 / 主动搭话选人权重。

- behavior_hint：六档行为映射
- processor._build_relation_block：好感度档位 + 行为指导注入；开关可关
- proactive._intimacy_weight / scan：私聊按好感度排序，高亲密度优先
"""

import asyncio
from types import SimpleNamespace

import pytest

from junjun_express import intimacy


class TestBehaviorHint:
    def test_all_levels_have_hint(self):
        for level in ("挚友", "好朋友", "朋友", "熟人", "认识", "陌生"):
            assert intimacy.behavior_hint(level)

    def test_intimacy_gradient(self):
        """档位越高行为越亲近（文本语义抽查）。"""
        assert "无话不谈" in intimacy.behavior_hint("挚友")
        assert "距离" in intimacy.behavior_hint("陌生")

    def test_unknown_level_falls_back(self):
        assert intimacy.behavior_hint("不存在") == intimacy.behavior_hint("陌生")


class TestRelationBlockInjection:
    def _session(self):
        return SimpleNamespace(platform="qq", chat_id="qq:12345:private")

    def _meta(self, user_id="12345", nickname="小明"):
        return SimpleNamespace(user_id=user_id, nickname=nickname)

    def test_intimacy_line_injected(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        from junjun_agent import processor
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            m.Intimacy.create(user_id="12345", score=75.0, interaction_count=200)
            block = processor._build_relation_block(self._session(), self._meta())
        assert "好朋友" in block and "75/100" in block and "200" in block
        assert "亲近自然" in block  # 行为指导跟着档位走

    def test_no_record_shows_stranger(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        from junjun_agent import processor
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            block = processor._build_relation_block(self._session(), self._meta())
        assert "陌生" in block

    def test_disable_switch(self, tmp_path):
        import peewee
        from junjun_core.config import get_global_config
        from junjun_core.database import models as m
        from junjun_agent import processor
        get_global_config().raw["relationship"] = {"enable": False}
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            m.Intimacy.create(user_id="12345", score=95.0, interaction_count=999)
            block = processor._build_relation_block(self._session(), self._meta())
        assert "好感度" not in block

    def test_no_user_id_empty(self):
        from junjun_agent import processor
        assert processor._build_relation_block(
            self._session(), self._meta(user_id="")) == ""


class TestProactiveWeight:
    def _session(self, chat_id, user_id="", is_group=False):
        return SimpleNamespace(chat_id=chat_id, user_id=user_id, is_group=is_group)

    def test_group_weight_zero(self):
        from junjun_agent.loop.proactive import _intimacy_weight
        assert _intimacy_weight(self._session("qq:1:group", is_group=True)) == 0.0

    def test_private_weight_is_score(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        from junjun_agent.loop.proactive import _intimacy_weight
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            m.Intimacy.create(user_id="111", score=80.0, interaction_count=10)
            w = _intimacy_weight(self._session("qq:111:private", user_id="111"))
        assert w == 80.0

    @pytest.mark.asyncio
    async def test_scan_orders_by_intimacy(self, tmp_path, monkeypatch):
        """高好感度的私聊排在前面先被主动搭话。"""
        import peewee
        from junjun_core.database import models as m
        from junjun_agent.loop import proactive
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        sessions = [
            self._session("qq:low:private", user_id="low"),
            self._session("qq:high:private", user_id="high"),
            self._session("qq:g:group", is_group=True),
        ]
        tried = []
        monkeypatch.setattr(proactive.proactive_manager, "eligible", lambda s: True)

        async def _try(session, **kw):
            tried.append(session.chat_id)
            return True
        monkeypatch.setattr(proactive.proactive_manager, "try_proactive", _try)
        monkeypatch.setattr(
            "junjun_core.gateway.session_manager.get_session_manager",
            lambda: SimpleNamespace(all_sessions=lambda: {s.chat_id: s for s in sessions}))

        with db.bind_ctx([m.Intimacy]):
            db.create_tables([m.Intimacy])
            m.Intimacy.create(user_id="low", score=20.0, interaction_count=5)
            m.Intimacy.create(user_id="high", score=90.0, interaction_count=50)
            await proactive.proactive_manager.scan()
        assert tried[0] == "qq:high:private"
        assert tried.index("qq:high:private") < tried.index("qq:low:private")

    @pytest.mark.asyncio
    async def test_scan_weight_switch_off(self, monkeypatch):
        """proactive_weight=False 时保持原顺序（先 eligible 先尝试）。"""
        from junjun_core.config import get_global_config
        from junjun_agent.loop import proactive
        get_global_config().raw["relationship"] = {"proactive_weight": False}
        sessions = [
            self._session("qq:a:private", user_id="a"),
            self._session("qq:b:private", user_id="b"),
        ]
        tried = []
        monkeypatch.setattr(proactive.proactive_manager, "eligible", lambda s: True)

        async def _try(session, **kw):
            tried.append(session.chat_id)
            return True
        monkeypatch.setattr(proactive.proactive_manager, "try_proactive", _try)
        monkeypatch.setattr(
            "junjun_core.gateway.session_manager.get_session_manager",
            lambda: SimpleNamespace(all_sessions=lambda: {s.chat_id: s for s in sessions}))
        # 打爆权重函数：若被调用会抛错，证明排序没发生
        monkeypatch.setattr(
            proactive, "_intimacy_weight",
            lambda s: (_ for _ in ()).throw(AssertionError("不应排序")))
        await proactive.proactive_manager.scan()
        assert tried == ["qq:a:private", "qq:b:private"]
