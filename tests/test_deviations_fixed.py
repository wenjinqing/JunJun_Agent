"""偏离修复回归测试：LPMM 知识库 / 表达反思 / hippo 话题缓存。"""

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
def _no_ltm_write(monkeypatch, tmp_path):
    """长期库写重定向到临时目录（不污染 data/）。"""
    import junjun_memory.long_term as lt
    monkeypatch.setattr(lt, "_ltm", lt.LongTermMemory(data_dir=tmp_path / "ltm"))


class _FakeOpenIE:
    """openIE 抽取 fake：识别「X是Y」句式。"""
    async def ainvoke(self, msgs, config=None):
        text = msgs[0].content

        class R:
            pass
        r = R()
        if "五条悟" in text and "从这个问题" not in text:
            r.content = json.dumps({
                "entities": ["五条悟", "咒术回战", "六眼"],
                "triples": [["五条悟", "出自", "咒术回战"], ["五条悟", "拥有", "六眼"]],
            }, ensure_ascii=False)
        elif "从这个问题" in text or "关键实体" in text:
            r.content = '["五条悟"]'
        else:
            r.content = '{"entities": [], "triples": []}'
        return r


class TestKnowledgeBase:
    @pytest.mark.asyncio
    async def test_import_and_ppr_search(self, tmp_path):
        from junjun_memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(data_dir=tmp_path)
        n = await kb.import_text("五条悟是《咒术回战》里的最强咒术师，拥有六眼和无下限咒术。",
                                 model=_FakeOpenIE())
        assert n == 1
        paras = await kb.search("五条悟是谁", model=_FakeOpenIE())
        assert paras and "五条悟" in paras[0]

    @pytest.mark.asyncio
    async def test_duplicate_import_skipped(self, tmp_path):
        from junjun_memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(data_dir=tmp_path)
        text = "五条悟是《咒术回战》里的最强咒术师。"
        assert await kb.import_text(text, model=_FakeOpenIE()) == 1
        assert await kb.import_text(text, model=_FakeOpenIE()) == 0

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path):
        from junjun_memory.knowledge import KnowledgeBase
        kb1 = KnowledgeBase(data_dir=tmp_path)
        await kb1.import_text("五条悟拥有六眼，是特级咒术师。", model=_FakeOpenIE())
        kb2 = KnowledgeBase(data_dir=tmp_path)
        paras = await kb2.search("五条悟", model=_FakeOpenIE())
        assert paras

    @pytest.mark.asyncio
    async def test_openie_failure_still_imports(self, tmp_path):
        from junjun_memory.knowledge import KnowledgeBase

        class Broken:
            async def ainvoke(self, msgs, config=None):
                raise ConnectionError()
        kb = KnowledgeBase(data_dir=tmp_path)
        assert await kb.import_text("一段没抽出实体的资料内容，够长了吧。", model=Broken()) == 1
        # 关键词降级仍可检索
        paras = await kb.search("资料内容", model=Broken())
        assert paras

    @pytest.mark.asyncio
    async def test_quick_algo_missing_degrades(self, tmp_path, monkeypatch):
        from junjun_memory import knowledge as kmod
        monkeypatch.setattr(kmod.KnowledgeBase, "_probe_quick_algo", staticmethod(lambda: False))
        kb = kmod.KnowledgeBase(data_dir=tmp_path)
        await kb.import_text("五条悟是最强的咒术师，六眼持有者。", model=_FakeOpenIE())
        paras = await kb.search("五条悟", model=_FakeOpenIE())
        assert paras  # 无图也能关键词命中

    @pytest.mark.asyncio
    async def test_skills(self, tmp_path, monkeypatch):
        from junjun_memory import knowledge as kmod
        monkeypatch.setattr(kmod, "_kb", kmod.KnowledgeBase(data_dir=tmp_path))
        monkeypatch.setattr(kmod.KnowledgeBase, "_probe_quick_algo", staticmethod(lambda: False))

        # import_knowledge / search_knowledge 走 utils 模型——注入 fake
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda t: _FakeOpenIE())

        from junjun_skills.builtin.knowledge_skills import import_knowledge, search_knowledge
        out = await import_knowledge.ainvoke({"text": "五条悟是特级咒术师，拥有六眼。"})
        assert "已导入" in out
        out = await search_knowledge.ainvoke({"question": "五条悟是谁"})
        assert "五条悟" in out


class TestReflector:
    @pytest.fixture
    def _cfg(self, _fake_bot_config):
        _fake_bot_config.raw["expression"] = {
            "reflect": True, "reflect_operator_id": "qq:10086:private",
        }
        return _fake_bot_config

    @pytest.mark.asyncio
    async def test_ask_sends_to_operator(self, _cfg, monkeypatch):
        from junjun_core.database import Expression
        Expression.create(chat_id="qq:1:group", situation="震惊", style="我直接爆炸",
                          count=2, last_active_time=time.time())
        sent = []

        class FakeGW:
            async def send_reply(self, reply):
                sent.append(reply)
        import junjun_core.gateway.router as rm
        monkeypatch.setattr(rm, "_gateway", FakeGW())

        from junjun_express.reflector import ExpressionReflector
        r = ExpressionReflector()
        r.last_ask_time = 0  # 强制到期
        monkeypatch.setattr(r, "_interval", lambda: 0)
        assert await r.check_and_ask()
        assert sent and sent[0].target_user_id == "10086"
        assert "我直接爆炸" in sent[0].segments[0].data

    @pytest.mark.asyncio
    async def test_operator_delete_reply(self, _cfg, monkeypatch):
        from junjun_core.database import Expression
        row = Expression.create(chat_id="qq:1:group", situation="s", style="bad",
                                count=1, last_active_time=time.time())

        class FakeGW:
            async def send_reply(self, reply):
                pass
        import junjun_core.gateway.router as rm
        monkeypatch.setattr(rm, "_gateway", FakeGW())

        from junjun_express.reflector import ExpressionReflector
        r = ExpressionReflector()
        r.last_ask_time = 0
        monkeypatch.setattr(r, "_interval", lambda: 0)
        await r.check_and_ask()
        receipt = r.handle_operator_reply("qq:10086:private", "删除 1")
        assert receipt and "已删" in receipt
        assert Expression.select().count() == 0

    def test_non_operator_ignored(self, _cfg):
        from junjun_express.reflector import ExpressionReflector
        r = ExpressionReflector()
        r._pending = {1: 99}
        assert r.handle_operator_reply("qq:99999:group", "删除 1") is None

    def test_disabled_no_ask(self, _fake_bot_config):
        _fake_bot_config.raw["expression"] = {"reflect": False}
        from junjun_express.reflector import ExpressionReflector
        assert not ExpressionReflector().enabled()


class _FakeSummaryModel:
    def __init__(self, lines):
        self.lines = lines

    async def ainvoke(self, msgs, config=None):
        content = msgs[0].content

        class R:
            pass
        r = R()
        if "合并压缩" in content:
            r.content = "合并后的内容"
        else:
            r.content = "\n".join(self.lines)
        return r


class TestTopicCache:
    @pytest.mark.asyncio
    async def test_same_topic_accumulates(self, tmp_path):
        from junjun_memory.summarizer import ChatSummarizer
        s = ChatSummarizer(data_dir=tmp_path)
        # 两轮不同内容（相同内容会走「已包含」短路不调合并 LLM——正确行为）
        for round_, line in enumerate(["学习: 甲开始学做菜（甲）", "学习: 甲今天学会了红烧肉（甲）"]):
            model = _FakeSummaryModel([line])
            for i in range(10):
                s.note("c1", "甲", f"做菜进展{round_}-{i}")
            await s.summarize("c1", model=model)
        topics = s._topics["c1"]
        assert "学习" in topics
        assert topics["学习"].update_count == 2
        assert topics["学习"].content == "合并后的内容"

    @pytest.mark.asyncio
    async def test_mature_topic_archived_to_ltm(self, tmp_path):
        from junjun_memory.summarizer import ChatSummarizer, FLUSH_UPDATES
        from junjun_memory.long_term import get_long_term_memory
        s = ChatSummarizer(data_dir=tmp_path)
        model = _FakeSummaryModel(["约定: 周五聚餐（全员）"])
        for round_ in range(FLUSH_UPDATES):
            for i in range(10):
                s.note("c1", "甲", f"聚餐讨论{round_}-{i}")
            await s.summarize("c1", model=model)
        assert "约定" not in s._topics["c1"]  # 成熟已归档移除
        hits = await get_long_term_memory().search("聚餐", top_k=3)
        assert hits and "约定" in hits[0].text

    @pytest.mark.asyncio
    async def test_topic_cache_persists(self, tmp_path):
        from junjun_memory.summarizer import ChatSummarizer
        s1 = ChatSummarizer(data_dir=tmp_path)
        model = _FakeSummaryModel(["爱好: 乙喜欢钓鱼（乙）"])
        for i in range(10):
            s1.note("c1", "乙", f"钓鱼话题{i}")
        await s1.summarize("c1", model=model)
        s2 = ChatSummarizer(data_dir=tmp_path)
        topics = s2._load_topics("c1")
        assert "爱好" in topics
