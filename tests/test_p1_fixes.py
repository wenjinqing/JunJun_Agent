"""P1 修复测试：retry 排超时 / embedding 查询缓存 / 记忆备份回退 /
forget 容量淘汰 / PPR 稀疏实现等价性。
"""

import asyncio
import json
import time

import numpy as np
import pytest


class TestRetrySkipsTimeout:
    @pytest.mark.asyncio
    async def test_timeout_not_retried(self):
        """TimeoutError 不重试——30s 超时重试 3 次会把阻塞放大到 93s。"""
        from junjun_core.retry import retry_async
        calls = []

        async def _fn():
            calls.append(1)
            raise asyncio.TimeoutError("mcp 30s 超时")

        with pytest.raises(asyncio.TimeoutError):
            await retry_async(_fn, attempts=3, base_delay=0.01)
        assert len(calls) == 1  # 没有重试

    @pytest.mark.asyncio
    async def test_transient_still_retried(self):
        """瞬态网络错误照常重试（回归保护）。"""
        import httpx
        from junjun_core.retry import retry_async
        calls = []

        async def _fn():
            calls.append(1)
            if len(calls) < 2:
                raise httpx.ConnectError("抖一下")
            return "ok"

        assert await retry_async(_fn, attempts=3, base_delay=0.01) == "ok"
        assert len(calls) == 2


class TestEmbeddingQueryCache:
    @pytest.mark.asyncio
    async def test_repeated_query_hits_cache(self, monkeypatch):
        from junjun_memory.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        from collections import OrderedDict
        client._query_cache = OrderedDict()
        api_calls = []

        async def _embed(texts):
            api_calls.append(texts)
            return [[1.0] * 4 for _ in texts]
        monkeypatch.setattr(client, "embed", _embed)

        v1 = await client.embed_one("原神是什么")
        v2 = await client.embed_one("原神是什么")
        assert v1 == v2 == [1.0] * 4
        assert len(api_calls) == 1  # 第二次走缓存

    @pytest.mark.asyncio
    async def test_failure_not_cached(self, monkeypatch):
        from collections import OrderedDict
        from junjun_memory.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._query_cache = OrderedDict()
        calls = []

        async def _embed(texts):
            calls.append(1)
            return None  # 失败
        monkeypatch.setattr(client, "embed", _embed)
        assert await client.embed_one("x") is None
        assert await client.embed_one("x") is None
        assert len(calls) == 2  # 失败不缓存，下次还试

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, monkeypatch):
        from collections import OrderedDict
        import junjun_memory.embedding as emb
        monkeypatch.setattr(emb, "_QUERY_CACHE_MAX", 3)
        client = emb.EmbeddingClient.__new__(emb.EmbeddingClient)
        client._query_cache = OrderedDict()

        async def _embed(texts):
            return [[float(len(t))] for t in texts]
        monkeypatch.setattr(client, "embed", _embed)
        for i in range(5):
            await client.embed_one(f"q{i}")
        assert len(client._query_cache) == 3
        assert "q0" not in client._query_cache and "q1" not in client._query_cache
        assert "q4" in client._query_cache


class TestLongTermBackupRecovery:
    def _mem(self, tmp_path, monkeypatch):
        from junjun_memory import long_term as lt

        async def _none_embed(text):
            return None
        fake = type("E", (), {"_model": "BAAI/bge-m3", "available": False,
                              "embed_one": staticmethod(_none_embed)})()
        monkeypatch.setattr(lt, "get_embedding_client", lambda: fake)
        return lt, lt.LongTermMemory(data_dir=tmp_path / "mem")

    @pytest.mark.asyncio
    async def test_corrupted_primary_recovers_from_bak(self, tmp_path, monkeypatch):
        """主文件损坏 -> 从 .bak 恢复，不清空全库。"""
        lt, mem = self._mem(tmp_path, monkeypatch)
        mem.load()
        await mem.add("珍贵记忆一", chat_id="qq:1:group")
        await mem.add("珍贵记忆二", chat_id="qq:1:group")
        # 主 metadata 写坏（模拟断电半写）
        mem._meta_path().write_text("{损坏的json", encoding="utf-8")

        _, mem2 = self._mem(tmp_path, monkeypatch)
        mem2.load()
        texts = [it.text for it in mem2._items]
        assert "珍贵记忆一" in texts and "珍贵记忆二" in texts

    @pytest.mark.asyncio
    async def test_both_corrupted_rebuilds_empty(self, tmp_path, monkeypatch):
        """主备都坏才重建空库（保底不崩）。"""
        lt, mem = self._mem(tmp_path, monkeypatch)
        mem.load()
        await mem.add("记忆", chat_id="qq:1:group")
        mem._meta_path().write_text("{坏", encoding="utf-8")
        mem._meta_path().with_suffix(".bak").write_text("{也坏", encoding="utf-8")

        _, mem2 = self._mem(tmp_path, monkeypatch)
        mem2.load()
        assert mem2._items == []

    @pytest.mark.asyncio
    async def test_fresh_library_loads(self, tmp_path, monkeypatch):
        """全新库（无任何文件）正常初始化为空库。"""
        _, mem = self._mem(tmp_path, monkeypatch)
        mem.load()
        assert mem._items == [] and mem._index is not None


class TestForgetCapacity:
    @pytest.mark.asyncio
    async def test_capacity_eviction_by_weight_and_age(self, tmp_path, monkeypatch):
        """超容量：低权重 + 最旧的先淘汰（原 min_weight 条件够不着的也能清）。"""
        from junjun_memory import long_term as lt
        fake = type("E", (), {"_model": "BAAI/bge-m3", "available": False})()
        monkeypatch.setattr(lt, "get_embedding_client", lambda: fake)
        mem = lt.LongTermMemory(data_dir=tmp_path / "mem")
        mem.load()
        now = time.time()
        # 全部 weight=1.0（生产常态）——旧 forget 条件一条都删不掉
        for i in range(10):
            mem._items.append(lt.MemoryItem(
                text=f"记忆{i}", chat_id="c", timestamp=now - (10 - i) * 100,
                weight=1.0))
        removed = mem.forget(max_items=6)
        assert removed == 4
        texts = [it.text for it in mem._items]
        assert "记忆9" in texts and "记忆8" in texts  # 新的留下
        assert "记忆0" not in texts  # 最旧的淘汰

    @pytest.mark.asyncio
    async def test_under_capacity_no_eviction(self, tmp_path, monkeypatch):
        from junjun_memory import long_term as lt
        fake = type("E", (), {"_model": "BAAI/bge-m3", "available": False})()
        monkeypatch.setattr(lt, "get_embedding_client", lambda: fake)
        mem = lt.LongTermMemory(data_dir=tmp_path / "mem")
        mem.load()
        mem._items.append(lt.MemoryItem(text="唯一", chat_id="c",
                                        timestamp=time.time(), weight=1.0))
        assert mem.forget(max_items=100) == 0


class TestSparsePPR:
    @pytest.mark.asyncio
    async def test_sparse_matches_dense_reference(self, tmp_path, monkeypatch):
        """稀疏幂迭代结果与 dense 参考实现一致（同一图上对比）。"""
        from junjun_memory import long_term as lt
        from junjun_memory.memory_graph import MemoryGraph
        fake = type("E", (), {"_model": "BAAI/bge-m3", "available": False})()
        monkeypatch.setattr(lt, "get_embedding_client", lambda: fake)
        mem = lt.LongTermMemory(data_dir=tmp_path / "mem")
        mem.load()
        t = time.time()
        # 链式共现：0-1-2-3
        for i in range(4):
            mem._items.append(lt.MemoryItem(
                text=f"m{i}", chat_id="c", timestamp=t + i * 10, weight=1.0))

        g = MemoryGraph()
        g._maybe_rebuild(mem)
        sparse_result = g.spread(mem, [0], top_k=3)

        # dense 参考实现（旧算法）
        n = len(g._adj)
        W = np.zeros((n, n), dtype="float32")
        for i, nbrs in enumerate(g._adj):
            if nbrs:
                for j in nbrs:
                    W[j, i] = 1.0 / len(nbrs)
        p0 = np.zeros(n, dtype="float32")
        p0[0] = 1.0
        p = p0.copy()
        for _ in range(30):
            p = 0.85 * (W @ p) + 0.15 * p0
        p[0] = 0.0
        dense_order = [int(i) for i in np.argsort(-p) if p[i] > 0]

        assert sparse_result == dense_order
