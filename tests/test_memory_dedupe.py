"""长期记忆夜间整理（dedupe）测试：近重合并的安全纪律。

核心断言方向是「不误杀」：不同文本/不同会话/pinned 绝不合并；
真重复（同向量、同会话、非 pinned）必须合并且权重/时间戳取优。
embedding 用确定性假向量（同文本同向量），不打 API。
"""

import time

import numpy as np
import pytest

from junjun_memory.embedding import EMBED_DIM
from junjun_memory.long_term import LongTermMemory, MemoryItem


def _fake_vec(text: str):
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

    monkeypatch.setattr(emb_mod, "_client", FakeClient())
    yield
    monkeypatch.setattr(emb_mod, "_client", None)


def _raw_add(ltm, text, chat_id, *, weight=1.0, kind="chat", ts=None,
             has_vec=True):
    """绕过 add() 的写入期去重直接入库（测的正是漏网之鱼）。"""
    ltm.load()
    item = MemoryItem(text=text, chat_id=chat_id,
                      timestamp=ts or time.time(), weight=weight,
                      kind=kind, has_vec=has_vec)
    ltm._items.append(item)
    if has_vec:
        v = np.array([_fake_vec(text)], dtype="float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        ltm._index.add(v)
        ltm._vec_map.append(len(ltm._items) - 1)
    return item


class TestDedupe:
    def test_exact_dupes_merged(self, tmp_path, fake_embedding):
        ltm = LongTermMemory(data_dir=tmp_path)
        old = _raw_add(ltm, "小明喜欢吃苹果", "qq:g1:group",
                       weight=1.2, ts=time.time() - 3600)
        new = _raw_add(ltm, "小明喜欢吃苹果", "qq:g1:group", weight=1.0)
        _raw_add(ltm, "完全无关的另一件事", "qq:g1:group")
        merged = ltm.dedupe(threshold=0.95)
        assert merged == 1
        assert len(ltm._items) == 2
        survivor = ltm._items[0]
        assert survivor.text == "小明喜欢吃苹果"
        assert survivor.weight == pytest.approx(min(2.0, 1.2 + 0.1))
        assert survivor.timestamp == new.timestamp, "时间戳取新（防被时效衰减误伤）"
        assert ltm._index.ntotal == len(ltm._vec_map), "索引不变量必须保住"

    def test_distinct_texts_untouched(self, tmp_path, fake_embedding):
        """误判回归：不同内容一条都不许并。"""
        ltm = LongTermMemory(data_dir=tmp_path)
        texts = ["甲喜欢猫", "乙在备考期末", "丙的猫生病了",
                 "周五晚上开黑", "丁失恋了求安慰", "食堂新窗口难吃"]
        for t in texts:
            _raw_add(ltm, t, "qq:g1:group")
        assert ltm.dedupe(threshold=0.95) == 0
        assert len(ltm._items) == len(texts)

    def test_pinned_never_merged(self, tmp_path, fake_embedding):
        """pinned 是用户显式钉的——既不当保留方合并别人，也不被别人并掉。"""
        ltm = LongTermMemory(data_dir=tmp_path)
        _raw_add(ltm, "重要的事", "qq:g1:group", kind="pinned")
        _raw_add(ltm, "重要的事", "qq:g1:group")
        assert ltm.dedupe(threshold=0.95) == 0
        assert len(ltm._items) == 2

    def test_cross_chat_not_merged(self, tmp_path, fake_embedding):
        """同一事实分属不同会话 = 两段各自的人际关系上下文，不许并。"""
        ltm = LongTermMemory(data_dir=tmp_path)
        _raw_add(ltm, "同样的话", "qq:g1:group")
        _raw_add(ltm, "同样的话", "qq:g2:group")
        assert ltm.dedupe(threshold=0.95) == 0
        assert len(ltm._items) == 2

    def test_plain_text_exact_merged(self, tmp_path, fake_embedding):
        """embedding 不可用期写入的纯文本条目：归一化全文精确合并。"""
        ltm = LongTermMemory(data_dir=tmp_path)
        _raw_add(ltm, "丙  喜欢熬夜", "qq:g1:group", has_vec=False, weight=1.0)
        _raw_add(ltm, "丙 喜欢熬夜", "qq:g1:group", has_vec=False, weight=1.5)
        _raw_add(ltm, "丁喜欢早睡", "qq:g1:group", has_vec=False)
        assert ltm.dedupe(threshold=0.95) == 1
        assert len(ltm._items) == 2
        assert ltm._items[0].weight == pytest.approx(1.6)

    def test_empty_library_noop(self, tmp_path, fake_embedding):
        ltm = LongTermMemory(data_dir=tmp_path)
        assert ltm.dedupe() == 0

    def test_merged_state_persists(self, tmp_path, fake_embedding):
        """合并结果落盘：新实例加载后保持合并后状态。"""
        ltm = LongTermMemory(data_dir=tmp_path)
        _raw_add(ltm, "重复的事", "qq:g1:group")
        _raw_add(ltm, "重复的事", "qq:g1:group")
        _raw_add(ltm, "独特的事", "qq:g1:group")
        assert ltm.dedupe(threshold=0.95) == 1
        ltm2 = LongTermMemory(data_dir=tmp_path)
        ltm2.load()
        assert len(ltm2._items) == 2
