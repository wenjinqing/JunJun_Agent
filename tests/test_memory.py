"""阶段 4 记忆系统单测：长期记忆 faiss 存取 / 摘要批次 / 画像 merge / 黑话。"""

import asyncio
import time

import pytest

from junjun_memory.long_term import LongTermMemory
from junjun_memory.summarizer import ChatSummarizer, BATCH_SIZE
from junjun_memory.embedding import EMBED_DIM


# ---------- fake embedding：确定性向量，不打 API ----------

def _fake_vec(text: str):
    """按字符哈希生成确定性 1024 维向量（相同文本同向量，相似文本相近）。"""
    import numpy as np
    rng = np.random.RandomState(abs(hash(text[:6])) % (2**31))
    return rng.rand(EMBED_DIM).astype("float32").tolist()


@pytest.fixture
def fake_embedding(monkeypatch):
    import junjun_memory.embedding as emb_mod

    class FakeClient:
        available = True

        async def embed(self, texts):
            return [_fake_vec(t) for t in texts]

        async def embed_one(self, text):
            return _fake_vec(text)

    fake = FakeClient()
    monkeypatch.setattr(emb_mod, "_client", fake)
    yield fake
    monkeypatch.setattr(emb_mod, "_client", None)


@pytest.fixture
def no_embedding(monkeypatch):
    import junjun_memory.embedding as emb_mod

    class DownClient:
        available = False

        async def embed(self, texts):
            return None

        async def embed_one(self, text):
            return None

    monkeypatch.setattr(emb_mod, "_client", DownClient())
    yield
    monkeypatch.setattr(emb_mod, "_client", None)


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_add_and_search_roundtrip(self, tmp_path, fake_embedding):
        ltm = LongTermMemory(data_dir=tmp_path)
        assert await ltm.add("甲最爱吃火锅", "qq:999:group")
        assert await ltm.add("乙下周三过生日", "qq:999:group")
        hits = await ltm.search("甲最爱吃火锅", top_k=2)
        assert hits and hits[0].text == "甲最爱吃火锅"  # 同文本同向量必第一

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path, fake_embedding):
        ltm1 = LongTermMemory(data_dir=tmp_path)
        await ltm1.add("持久化测试内容", "c1")
        ltm2 = LongTermMemory(data_dir=tmp_path)  # 新实例重新 load
        hits = await ltm2.search("持久化测试内容", top_k=1)
        assert hits and hits[0].text == "持久化测试内容"

    @pytest.mark.asyncio
    async def test_corrupt_meta_rebuilds_empty(self, tmp_path, fake_embedding):
        ltm1 = LongTermMemory(data_dir=tmp_path)
        await ltm1.add("x", "c1")
        (tmp_path / "metadata.json").write_text('{"dim": 512, "model": "other", "items": []}')
        ltm2 = LongTermMemory(data_dir=tmp_path)
        ltm2.load()
        assert ltm2._items == []  # 维度不匹配重建空库

    @pytest.mark.asyncio
    async def test_chat_id_filter(self, tmp_path, fake_embedding):
        ltm = LongTermMemory(data_dir=tmp_path)
        await ltm.add("A群的事", "qq:1:group")
        await ltm.add("B群的事", "qq:2:group")
        hits = await ltm.search("A群的事", top_k=5, chat_id="qq:2:group")
        assert all(h.chat_id == "qq:2:group" for h in hits)

    @pytest.mark.asyncio
    async def test_keyword_fallback_when_embedding_down(self, tmp_path, no_embedding):
        ltm = LongTermMemory(data_dir=tmp_path)
        # embedding 不可用时 add 仍成功（纯文本条目）
        assert await ltm.add("老王爱钓鱼", "c1")
        hits = await ltm.search("钓鱼", top_k=3)
        assert hits and "钓鱼" in hits[0].text
        assert not hits[0].has_vec

    @pytest.mark.asyncio
    async def test_plain_items_upgrade_visible_after_embedding_returns(self, tmp_path, no_embedding, monkeypatch):
        """embedding 恢复后，纯文本旧条目仍能被关键词补充检索到。"""
        ltm = LongTermMemory(data_dir=tmp_path)
        await ltm.add("纯文本时期的记忆钓鱼佬", "c1")
        # 切换到 fake embedding（恢复）
        import junjun_memory.embedding as emb_mod

        class FakeClient:
            available = True

            async def embed_one(self, text):
                return _fake_vec(text)
        monkeypatch.setattr(emb_mod, "_client", FakeClient())
        await ltm.add("向量时期的记忆", "c1")
        hits = await ltm.search("钓鱼佬", top_k=5)
        assert any("钓鱼佬" in h.text for h in hits)

    @pytest.mark.asyncio
    async def test_forget_removes_old_low_weight(self, tmp_path, fake_embedding):
        ltm = LongTermMemory(data_dir=tmp_path)
        await ltm.add("旧的不重要", "c1", weight=0.1)
        await ltm.add("新的重要", "c1", weight=1.0)
        ltm._items[0].timestamp = time.time() - 100 * 86400  # 100 天前
        removed = ltm.forget(max_age_days=90, min_weight=0.2)
        assert removed == 1
        assert len(ltm._items) == 1
        assert ltm._items[0].text == "新的重要"
        # 索引与向量映射仍一致
        assert ltm._index.ntotal == len(ltm._vec_map) == 1


class TestSummarizer:
    def test_batch_trigger_by_count(self, tmp_path):
        s = ChatSummarizer(data_dir=tmp_path)
        for i in range(BATCH_SIZE - 1):
            assert not s.note("c1", "甲", f"msg{i}")
        assert s.note("c1", "甲", "last")  # 满批触发

    @pytest.mark.asyncio
    async def test_summarize_writes_and_resets(self, tmp_path, fake_embedding, monkeypatch):
        import junjun_memory.long_term as lt_mod
        monkeypatch.setattr(lt_mod, "_ltm", LongTermMemory(data_dir=tmp_path / "ltm"))

        s = ChatSummarizer(data_dir=tmp_path)
        for i in range(10):
            s.note("c1", "甲", f"我最近在学做菜，第{i}天")

        class FakeModel:
            async def ainvoke(self, msgs, config=None):
                class R:
                    content = "喜好: 甲在学做菜（甲）"
                return R()

        out = await s.summarize("c1", model=FakeModel())
        assert out == ["喜好: 甲在学做菜（甲）"]
        assert s._batches["c1"].lines == []  # 批次已消费
        # 落盘验证
        files = list(tmp_path.glob("c1*.json"))
        assert files and "学做菜" in files[0].read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_worthless_chat_returns_empty(self, tmp_path):
        s = ChatSummarizer(data_dir=tmp_path)
        for _i in range(10):
            s.note("c1", "甲", "哈哈哈")

        class FakeModel:
            async def ainvoke(self, msgs, config=None):
                class R:
                    content = "无"
                return R()

        assert await s.summarize("c1", model=FakeModel()) == []

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_batch(self, tmp_path):
        s = ChatSummarizer(data_dir=tmp_path)
        for i in range(10):
            s.note("c1", "甲", f"msg{i}")

        class BrokenModel:
            async def ainvoke(self, msgs, config=None):
                raise ConnectionError("down")

        assert await s.summarize("c1", model=BrokenModel()) == []
        assert len(s._batches["c1"].lines) == 10  # 批次保留


class TestScheduler:
    @pytest.mark.asyncio
    async def test_interval_due(self):
        from junjun_agent.loop.scheduler import ScheduledTask

        async def cb():
            pass
        t = ScheduledTask("t", cb, interval=100)
        assert t.due(now=1000)
        t.mark_run(now=1000)
        assert not t.due(now=1050)
        assert t.due(now=1101)

    @pytest.mark.asyncio
    async def test_cron_once_per_day(self):
        from junjun_agent.loop.scheduler import ScheduledTask
        from datetime import datetime

        async def cb():
            pass
        t = ScheduledTask("t", cb, cron_hour=9, cron_minute=0)
        at9 = datetime(2026, 7, 16, 9, 0).timestamp()
        assert t.due(now=at9)
        t.mark_run(now=at9)
        assert not t.due(now=at9 + 30)  # 同日不重复
        next_day = datetime(2026, 7, 17, 9, 0).timestamp()
        assert t.due(now=next_day)

    @pytest.mark.asyncio
    async def test_crash_does_not_kill_loop(self):
        from junjun_agent.loop.scheduler import Scheduler, ScheduledTask
        s = Scheduler()
        s.TICK = 0.05
        ran = []

        async def bad():
            raise RuntimeError("boom")

        async def good():
            ran.append(1)

        s.add(ScheduledTask("bad", bad, interval=0.01))
        s.add(ScheduledTask("good", good, interval=0.01))
        s.start()
        await asyncio.sleep(0.25)
        await s.stop()
        assert ran  # bad 崩溃不影响 good
