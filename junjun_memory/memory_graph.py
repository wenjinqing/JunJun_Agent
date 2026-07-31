"""LPMM 图记忆（P2-20 试点）：记忆条目间建边，检索时 PPR 扩散取相关簇。

只读增强层：不改写入路径，[memory_graph] enable=false 即回滚扁平检索。

边（两种，无向）：
- 共现边：同会话、入库时间相邻（间隔 <= 10 分钟）的条目互连——一起聊到的就是相关的
- 同主题边：向量余弦相似 >= 0.75（双方都已向量化才算；防 O(n²) 只取最近 1500 条）

PPR（个性化 PageRank）：以检索命中的条目为种子，沿边扩散，把「相关联的记忆」
补充进检索结果——治「上次说的那个游戏」类指代问题（种子不一定直接命中，
但相关簇里常有答案）。

纯 numpy 实现（百~千级条目毫秒级），不引 quick_algo 原生依赖——
条目数到万级再考虑换原生图库。
"""

from typing import List

from junjun_core.observability import get_logger

logger = get_logger("memory.graph")

_COOCCUR_GAP = 600.0       # 共现边：同会话相邻入库间隔上限（秒）
_THEME_SIM = 0.75          # 同主题边：余弦相似阈值
_THEME_MAX_NODES = 1500    # 同主题边最多参与的条目数（取最近，防 O(n²) 爆炸）
_REBUILD_INTERVAL = 5      # 新增条目积累到该数才重建（图允许轻微过期）
_PPR_ITERS = 30            # PPR 幂迭代轮数


class MemoryGraph:
    """长期记忆的关联图（惰性构建，条目数变化积累到阈值才重建）。"""

    def __init__(self):
        self._built_count = -1          # 构建时的条目数（-1 = 未构建）
        self._adj: List[set] = []       # 邻接表：item 下标 -> 邻居下标集合

    # ---------- 构建 ----------

    def _maybe_rebuild(self, ltm) -> None:
        n = len(ltm._items)
        if self._built_count >= 0 and 0 <= n - self._built_count < _REBUILD_INTERVAL:
            return
        try:
            self._rebuild(ltm)
            self._built_count = n
        except Exception as e:
            logger.warning(f"记忆图构建失败（降级无图）: {type(e).__name__}: {e}")
            self._adj = [set() for _ in range(n)]
            self._built_count = n

    def _rebuild(self, ltm) -> None:
        import numpy as np
        items = ltm._items
        adj = [set() for _ in items]

        # 共现边：按会话分组、时间排序，相邻且间隔 <= _COOCCUR_GAP 互连
        by_chat = {}
        for i, it in enumerate(items):
            by_chat.setdefault(it.chat_id, []).append(i)
        for ids in by_chat.values():
            ids.sort(key=lambda i: items[i].timestamp)
            for a, b in zip(ids, ids[1:]):
                if items[b].timestamp - items[a].timestamp <= _COOCCUR_GAP:
                    adj[a].add(b)
                    adj[b].add(a)

        # 同主题边：向量条目两两余弦 >= _THEME_SIM 互连（只取最近 _THEME_MAX_NODES 条）
        vec_positions = list(range(len(ltm._vec_map)))
        if len(vec_positions) > _THEME_MAX_NODES:
            vec_positions = vec_positions[-_THEME_MAX_NODES:]
        if len(vec_positions) >= 2 and ltm._index is not None:
            vecs = np.stack([ltm._index.reconstruct(p) for p in vec_positions]).astype("float32")
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            sim = vecs @ vecs.T
            rows, cols = np.where(sim >= _THEME_SIM)
            for a, b in zip(rows.tolist(), cols.tolist()):
                if a >= b:
                    continue
                ia, ib = ltm._vec_map[vec_positions[a]], ltm._vec_map[vec_positions[b]]
                adj[ia].add(ib)
                adj[ib].add(ia)

        n_edges = sum(len(s) for s in adj) // 2
        logger.info(f"记忆图已构建: {len(items)} 节点 / {n_edges} 边")
        self._adj = adj

    # ---------- PPR 扩散 ----------

    def spread(self, ltm, seed_indices: List[int], *, top_k: int,
               damping: float = 0.85) -> List[int]:
        """PPR 从种子条目扩散，返回相关条目下标（不含种子），按得分降序。"""
        import numpy as np
        self._maybe_rebuild(ltm)
        n = len(self._adj)
        seeds = [s for s in seed_indices if 0 <= s < n]
        if not seeds or n == 0:
            return []

        # 转移矩阵（列归一）；n 百~千级 dense 足够快
        W = np.zeros((n, n), dtype="float32")
        for i, nbrs in enumerate(self._adj):
            if nbrs:
                for j in nbrs:
                    W[j, i] = 1.0 / len(nbrs)
        p0 = np.zeros(n, dtype="float32")
        for s in seeds:
            p0[s] = 1.0 / len(seeds)
        p = p0.copy()
        for _ in range(_PPR_ITERS):
            p = damping * (W @ p) + (1.0 - damping) * p0
        p[seeds] = 0.0  # 种子本身不算「相关记忆」
        order = np.argsort(-p)
        return [int(i) for i in order if p[i] > 0][:top_k]


_graph: "MemoryGraph | None" = None


def get_memory_graph() -> MemoryGraph:
    global _graph
    if _graph is None:
        _graph = MemoryGraph()
    return _graph
