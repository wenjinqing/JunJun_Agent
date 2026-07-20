"""阶段 5 单测：复读参与 / 主动私聊资格 / 表情包选择 / 表达学习。"""

import json
import time

import pytest
from peewee import SqliteDatabase


@pytest.fixture(autouse=True)
def _memory_db(monkeypatch):
    import junjun_core.database.models as m
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx(m.ALL_TABLES):
        test_db.create_tables(m.ALL_TABLES)
        monkeypatch.setattr(m, "db", test_db)
        import junjun_core.database as pkg
        monkeypatch.setattr(pkg, "db", test_db)
        yield test_db


@pytest.fixture(autouse=True)
def _stage5_config(_fake_bot_config):
    _fake_bot_config.raw.update({
        "repeat": {"enable": True, "threshold": 4, "min_interval_seconds": 60,
                   "min_message_length": 1, "max_message_length": 50},
        "proactive_chat": {"enable": True, "min_idle_minutes": 120,
                           "max_daily_proactive": 2, "silent_hours": "23:00-09:00",
                           "enable_in_groups": False, "enable_in_private": True},
        "emoji": {"steal_emoji": True, "max_reg_num": 2000, "do_replace": True},
        "expression": {"learning_list": []},
    })
    return _fake_bot_config


class TestRepeat:
    def _detector(self):
        from junjun_agent.loop.repeat import RepeatDetector
        return RepeatDetector()

    def test_threshold_triggers_with_distinct_users(self):
        d = self._detector()
        assert d.note("g1", "u1", "哈哈哈") is None
        assert d.note("g1", "u2", "哈哈哈") is None
        assert d.note("g1", "u3", "哈哈哈") is None
        assert d.note("g1", "u4", "哈哈哈") == "哈哈哈"  # 第4条触发

    def test_single_user_spam_no_trigger(self):
        d = self._detector()
        for _ in range(6):
            result = d.note("g1", "u1", "刷屏")
        assert result is None  # 同一人刷不算复读

    def test_different_text_breaks_chain(self):
        d = self._detector()
        d.note("g1", "u1", "哈哈哈")
        d.note("g1", "u2", "哈哈哈")
        d.note("g1", "u3", "换话题了")
        assert d.note("g1", "u4", "哈哈哈") is None  # 链断重新计

    def test_self_message_breaks_chain(self):
        d = self._detector()
        d.note("g1", "u1", "666")
        d.note("g1", "u2", "666")
        d.note("g1", "bot", "666", is_self=True)
        assert d.note("g1", "u3", "666") is None

    def test_cooldown_blocks_second_repeat(self):
        d = self._detector()
        now = time.time()
        for _i, u in enumerate(["u1", "u2", "u3", "u4"]):
            r = d.note("g1", u, "第一波", now=now)
        assert r == "第一波"
        for u in ["u1", "u2", "u3", "u4"]:
            r = d.note("g1", u, "第二波", now=now + 10)  # 冷却内
        assert r is None
        for u in ["u1", "u2", "u3", "u4"]:
            r = d.note("g1", u, "第三波", now=now + 70)  # 冷却过
        assert r == "第三波"

    def test_same_content_never_repeated_twice(self):
        d = self._detector()
        now = time.time()
        for u in ["u1", "u2", "u3", "u4"]:
            d.note("g1", u, "梗", now=now)
        for u in ["u1", "u2", "u3", "u4"]:
            r = d.note("g1", u, "梗", now=now + 100)
        assert r is None  # 同内容不二跟

    def test_too_long_message_ignored(self):
        d = self._detector()
        long = "长" * 60
        for u in ["u1", "u2", "u3", "u4"]:
            assert d.note("g1", u, long) is None


class TestProactiveEligibility:
    def _session(self, *, group=False, idle_min=200):
        from junjun_core.gateway.session_manager import ChatSession
        from junjun_memory.short_term import ShortTermMemory
        s = ChatSession("qq:1:group" if group else "qq:1:private", "qq",
                        group_id="1" if group else None,
                        user_id=None if group else "1")
        s.memory = ShortTermMemory()
        s.memory.add_user("之前聊过", "甲")
        s.last_active_ts = time.time() - idle_min * 60
        return s

    def _manager(self):
        from junjun_agent.loop.proactive import ProactiveChatManager
        return ProactiveChatManager()

    def test_idle_private_eligible(self, monkeypatch):
        import junjun_agent.loop.proactive as p
        monkeypatch.setattr(p, "_in_silent_hours", lambda now=None: False)
        assert self._manager().eligible(self._session(idle_min=200))

    def test_not_idle_enough(self, monkeypatch):
        import junjun_agent.loop.proactive as p
        monkeypatch.setattr(p, "_in_silent_hours", lambda now=None: False)
        assert not self._manager().eligible(self._session(idle_min=30))

    def test_group_disabled_by_default(self, monkeypatch):
        import junjun_agent.loop.proactive as p
        monkeypatch.setattr(p, "_in_silent_hours", lambda now=None: False)
        assert not self._manager().eligible(self._session(group=True, idle_min=200))

    def test_silent_hours_blocks(self, monkeypatch):
        import junjun_agent.loop.proactive as p
        monkeypatch.setattr(p, "_in_silent_hours", lambda now=None: True)
        assert not self._manager().eligible(self._session(idle_min=200))

    def test_daily_limit(self, monkeypatch):
        import junjun_agent.loop.proactive as p
        monkeypatch.setattr(p, "_in_silent_hours", lambda now=None: False)
        m = self._manager()
        s = self._session(idle_min=200)
        m._reset_daily()
        m._daily_count[s.chat_id] = 2  # 已达上限
        assert not m.eligible(s)

    def test_never_chatted_not_eligible(self, monkeypatch):
        import junjun_agent.loop.proactive as p
        monkeypatch.setattr(p, "_in_silent_hours", lambda now=None: False)
        s = self._session(idle_min=200)
        s.memory.entries = []
        assert not self._manager().eligible(s)


class TestEmojiPick:
    def _seed(self, tmp_path, n=3):
        from junjun_core.database import Emoji
        for i in range(n):
            p = tmp_path / f"e{i}.img"
            p.write_bytes(b"fake")
            Emoji.create(full_path=str(p), emoji_hash=f"h{i}",
                         description=f"表情{i}", emotion=json.dumps(["开心" if i == 0 else "无语"]))

    def test_pick_by_emotion(self, tmp_path):
        from junjun_express.emoji import EmojiManager
        self._seed(tmp_path)
        m = EmojiManager()
        picked = m.pick("开心", "c1")
        assert picked is not None

    def test_cooldown(self, tmp_path):
        from junjun_express.emoji import EmojiManager
        self._seed(tmp_path)
        m = EmojiManager()
        assert m.pick("开心", "c1") is not None
        assert m.pick("开心", "c1") is None  # 冷却中
        assert m.pick("开心", "c2") is not None  # 不同会话不受影响

    def test_empty_library(self):
        from junjun_express.emoji import EmojiManager
        assert EmojiManager().pick("开心", "c1") is None

    def test_usage_count_increments(self, tmp_path):
        from junjun_core.database import Emoji
        from junjun_express.emoji import EmojiManager
        self._seed(tmp_path, n=1)
        EmojiManager().pick("开心", "c1")
        assert Emoji.select().first().usage_count == 1


class TestExpression:
    @pytest.mark.asyncio
    async def test_learn_and_select(self):
        from junjun_express.expression import ExpressionLearner, select_expressions

        learner = ExpressionLearner()
        for i in range(15):
            learner.note("g1", "甲", f"我直接一个爆炸，太离谱了吧{i}")

        class FakeModel:
            async def ainvoke(self, msgs, config=None):
                class R:
                    content = '[{"situation": "表示震惊", "style": "我直接一个爆炸"}]'
                return R()

        learned = await learner.learn("g1", model=FakeModel())
        assert learned == 1
        exprs = select_expressions("g1", "表示震惊的语境")
        assert exprs and exprs[0]["style"] == "我直接一个爆炸"

    @pytest.mark.asyncio
    async def test_duplicate_style_reinforces(self):
        from junjun_core.database import Expression
        from junjun_express.expression import ExpressionLearner

        class FakeModel:
            async def ainvoke(self, msgs, config=None):
                class R:
                    content = '[{"situation": "开心", "style": "太好耶"}]'
                return R()

        learner = ExpressionLearner()
        for round_ in range(2):
            for i in range(15):
                learner.note("g1", "甲", f"msg{round_}-{i}")
            await learner.learn("g1", model=FakeModel())
        rows = list(Expression.select())
        assert len(rows) == 1
        assert rows[0].count == 2

    @pytest.mark.asyncio
    async def test_llm_garbage_no_crash(self):
        from junjun_express.expression import ExpressionLearner

        class Garbage:
            async def ainvoke(self, msgs, config=None):
                class R:
                    content = "这不是JSON"
                return R()

        learner = ExpressionLearner()
        for i in range(15):
            learner.note("g1", "甲", f"消息内容{i}")
        assert await learner.learn("g1", model=Garbage()) == 0

    def test_expression_block_empty_without_data(self):
        from junjun_express.expression import build_expression_block
        assert build_expression_block("g1", "随便") == ""
