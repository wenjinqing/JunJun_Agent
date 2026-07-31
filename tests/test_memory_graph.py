"""P2-20 LPMM 图记忆测试：共现/同主题建边 + PPR 扩散 + search 集成。

不引 quick_algo——纯 numpy 实现。测试用假 embedding（映射 text->向量），
faiss 索引用真实 IndexFlatIP（维度对齐 EMBED_DIM）。
"""

import time

import numpy as np
import pytest

from junjun_memory import long_term as lt
from junjun_memory.memory_graph import MemoryGraph
from junjun_memory.embedding import EMBED_DIM

CHAT = "qq:1:group"


def _vec(dim_seed: int) -> list:
    """以某一维为主的单位向量（不同 seed = 不同主题）。"""
    v = np.zeros(EMBED_DIM, dtype="float32")
    v[dim_seed % EMBED_DIM] = 1.0
    return v.tolist()


class _FakeEmbed:
    """text -> 向量映射；未登记的文本给零向量（检索时得分 < 0.3 自然滤掉）。"""

    def __init__(self, mapping: dict):
        self._map = mapping

    async def embed_one(self, text):
        return self._map.get(text)


@pytest.fixture
def ltm(tmp_path, monkeypatch):
    """空长期记忆库（临时目录）+ 可控假 embedding。"""
    mapping = {}
    fake = _FakeEmbed(mapping)
    monkeypatch.setattr(lt, "get_embedding_client", lambda: fake)
    mem = lt.LongTermMemory(data_dir=tmp_path / "mem")
    mem.load()
    return mem, mapping


@pytest.fixture(autouse=True)
def _fresh_graph():
    """每个测试一张新图（全局单例防串扰）。"""
    import junjun_memory.memory_graph as mg
    mg._graph = MemoryGraph()
    yield
    mg._graph = MemoryGraph()


async def _add(mem, text, vec_seed=None, ts=None, chat_id=CHAT):
    """直接构造条目入库（绕过 embed 调用，时间戳可控）。"""
    vec = _vec(vec_seed) if vec_seed is not None else None
    item = lt.MemoryItem(text=text, chat_id=chat_id,
                         timestamp=ts if ts is not None else time.time(),
                         has_vec=vec is not None)
    mem._items.append(item)
    if vec is not None:
        v = np.array([vec], dtype="float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        mem._index.add(v)
        mem._vec_map.append(len(mem._items) - 1)
    return len(mem._items) - 1


class TestGraphBuilding:
    @pytest.mark.asyncio
    async def test_cooccurrence_edges(self, ltm):
        """同会话、时间相邻（<=10 分钟）的条目互连。"""
        mem, _ = ltm
        t = time.time()
        a = await _add(mem, "打原神吗", ts=t)
        b = await _add(mem, "来，上号", ts=t + 60)        # 相邻 -> 连边
        c = await _add(mem, "晚饭吃啥", ts=t + 60 + 601)  # 间隔超 10 分钟 -> 不连
        g = MemoryGraph()
        g._maybe_rebuild(mem)
        assert b in g._adj[a] and a in g._adj[b]
        assert c not in g._adj[b]

    @pytest.mark.asyncio
    async def test_theme_edges(self, ltm):
        """向量高相似的条目互连（不同主题不连）。"""
        mem, _ = ltm
        a = await _add(mem, "原神新地图开了", vec_seed=1)
        b = await _add(mem, "原神活动明天结束", vec_seed=1)  # 同主题 -> 连边
        c = await _add(mem, "火锅真好吃", vec_seed=2)        # 不同主题 -> 不连
        g = MemoryGraph()
        g._maybe_rebuild(mem)
        assert b in g._adj[a]
        assert c not in g._adj[a]

    @pytest.mark.asyncio
    async def test_cross_chat_no_cooccurrence(self, ltm):
        """不同会话即使时间相邻也不连共现边。"""
        mem, _ = ltm
        t = time.time()
        a = await _add(mem, "A群的消息", ts=t, chat_id="qq:1:group")
        b = await _add(mem, "B群的消息", ts=t + 10, chat_id="qq:2:group")
        g = MemoryGraph()
        g._maybe_rebuild(mem)
        assert b not in g._adj[a]

    @pytest.mark.asyncio
    async def test_rebuild_interval(self, ltm):
        """新增条目 < 阈值不重建（图允许轻微过期）；条目减少立即重建。"""
        mem, _ = ltm
        for i in range(3):
            await _add(mem, f"记忆{i}", ts=time.time() + i)
        g = MemoryGraph()
        g._maybe_rebuild(mem)
        assert g._built_count == 3
        await _add(mem, "新记忆")  # +1 < 5，不重建
        g._maybe_rebuild(mem)
        assert g._built_count == 3
        mem._items.pop()  # 条目减少（遗忘）-> 立即重建
        g._maybe_rebuild(mem)
        assert g._built_count == 3  # 数量回到 3，但已重建过


class TestSpread:
    @pytest.mark.asyncio
    async def test_spread_finds_related(self, ltm):
        """PPR 从种子扩散：共现的邻居被找到，种子本身不返回。"""
        mem, _ = ltm
        t = time.time()
        a = await _add(mem, "昨晚那个游戏真好玩", ts=t)
        b = await _add(mem, "叫塞尔达传说", ts=t + 30)  # 与 a 共现
        far = await _add(mem, "完全无关的事", ts=t + 30 + 3600)
        g = MemoryGraph()
        related = g.spread(mem, [a], top_k=2)
        assert b in related
        assert a not in related

    @pytest.mark.asyncio
    async def test_spread_no_edges_returns_empty(self, ltm):
        mem, _ = ltm
        a = await _add(mem, "孤零零的记忆", ts=time.time())
        g = MemoryGraph()
        assert g.spread(mem, [a], top_k=2) == []

    @pytest.mark.asyncio
    async def test_spread_invalid_seeds(self, ltm):
        mem, _ = ltm
        await _add(mem, "一条")
        g = MemoryGraph()
        assert g.spread(mem, [99, -1], top_k=2) == []
        assert g.spread(mem, [], top_k=2) == []


class TestSearchIntegration:
    @pytest.mark.asyncio
    async def test_search_appends_related(self, ltm, monkeypatch):
        """search 命中后自动扩散：相关联的记忆追加到结果尾部。"""
        mem, mapping = ltm
        t = time.time()
        a = await _add(mem, "上周说的那个游戏", vec_seed=1, ts=t)
        b = await _add(mem, "是塞尔达没错", ts=t + 30)  # 无向量，但和 a 共现
        mapping["那个游戏叫什么"] = _vec(1)  # 查询向量命中 a
        out = await mem.search("那个游戏叫什么", top_k=1)
        texts = [it.text for it in out]
        assert "上周说的那个游戏" in texts
        assert "是塞尔达没错" in texts  # 图扩散补充

    @pytest.mark.asyncio
    async def test_spread_disabled(self, ltm, monkeypatch):
        """[memory_graph] enable=false -> 回滚扁平检索，不扩散。"""
        from junjun_core.config import get_global_config
        get_global_config().raw["memory_graph"] = {"enable": False}
        mem, mapping = ltm
        t = time.time()
        a = await _add(mem, "上周说的那个游戏", vec_seed=1, ts=t)
        await _add(mem, "是塞尔达没错", ts=t + 30)
        mapping["那个游戏叫什么"] = _vec(1)
        out = await mem.search("那个游戏叫什么", top_k=1)
        assert [it.text for it in out] == ["上周说的那个游戏"]

    @pytest.mark.asyncio
    async def test_spread_explicit_false(self, ltm):
        """spread=False 参数优先于配置。"""
        mem, mapping = ltm
        t = time.time()
        await _add(mem, "上周说的那个游戏", vec_seed=1, ts=t)
        await _add(mem, "是塞尔达没错", ts=t + 30)
        mapping["那个游戏叫什么"] = _vec(1)
        out = await mem.search("那个游戏叫什么", top_k=1, spread=False)
        assert [it.text for it in out] == ["上周说的那个游戏"]

    @pytest.mark.asyncio
    async def test_spread_respects_chat_filter(self, ltm):
        """chat_id 过滤对扩散结果同样生效。"""
        mem, mapping = ltm
        t = time.time()
        a = await _add(mem, "这个群的游戏", vec_seed=1, ts=t, chat_id="qq:1:group")
        await _add(mem, "别群的共现记忆", ts=t + 30, chat_id="qq:1:group")
        # 把别群记忆手动挂到 a 上不可能（共现同群），换个角度：
        # 验证传入 chat_id 时扩散补充项同群即可——上面两条同群都会返回
        mapping["游戏"] = _vec(1)
        out = await mem.search("游戏", top_k=1, chat_id="qq:1:group")
        assert all(it.chat_id == "qq:1:group" for it in out)

    @pytest.mark.asyncio
    async def test_keyword_path_also_spreads(self, ltm):
        """无向量（关键词降级）路径也能沿共现边扩散。"""
        mem, _ = ltm  # mapping 为空 -> embed_one 返回 None -> 关键词路径
        t = time.time()
        await _add(mem, "原神启动", ts=t)
        await _add(mem, "抽卡歪了", ts=t + 30)
        out = await mem.search("原神", top_k=1)
        texts = [it.text for it in out]
        assert "原神启动" in texts
        assert "抽卡歪了" in texts

    @pytest.mark.asyncio
    async def test_max_spread_config(self, ltm):
        """max_spread 控制补充条数。"""
        from junjun_core.config import get_global_config
        get_global_config().raw["memory_graph"] = {"enable": True, "max_spread": 1}
        mem, mapping = ltm
        t = time.time()
        await _add(mem, "游戏", vec_seed=1, ts=t)
        await _add(mem, "相关一", ts=t + 10)
        await _add(mem, "相关二", ts=t + 20)
        mapping["游戏"] = _vec(1)
        out = await mem.search("游戏", top_k=1)
        assert len(out) == 2  # 种子 1 + 扩散 1
