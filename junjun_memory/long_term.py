"""长期记忆：faiss 向量库（落盘持久化）+ 关键词降级检索。

对齐原 memory_system 语义：
- faiss IndexFlatIP（余弦相似，向量归一化后内积）+ JSON 元数据成对落盘
- 索引头记录维度+模型名，不匹配拒绝加载并重建（防换模型炸索引）
- embedding 不可用时**写入仍成功**（纯文本条目，关键词可检索），
  向量条目与文本条目共存：invariant 为 index.ntotal == len(vec_map)
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from junjun_core.observability import get_logger
from junjun_memory.embedding import get_embedding_client, EMBED_DIM, EMBED_MODEL

logger = get_logger("memory.longterm")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "memory"


@dataclass
class MemoryItem:
    text: str
    chat_id: str
    timestamp: float
    weight: float = 1.0
    kind: str = "chat"       # chat / summary / fact
    has_vec: bool = False    # 是否已向量化（False = 仅关键词可检索）


class LongTermMemory:
    """单实例记忆库（全会话共享，检索按 chat_id 过滤可选）。"""

    def __init__(self, data_dir: Optional[Path] = None):
        self._dir = data_dir or DATA_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index = None                 # faiss index（惰性）
        self._items: List[MemoryItem] = []
        self._vec_map: List[int] = []      # faiss 位置 -> _items 下标
        self._loaded = False

    # ---------- 持久化 ----------

    def _index_path(self) -> Path:
        return self._dir / "faiss_index.bin"

    def _meta_path(self) -> Path:
        return self._dir / "metadata.json"

    def load(self) -> None:
        """启动加载；索引与向量条目数不一致或维度/模型不匹配时重建。"""
        if self._loaded:
            return
        self._loaded = True
        import faiss
        meta_p, idx_p = self._meta_path(), self._index_path()
        if not meta_p.exists():
            self._index = faiss.IndexFlatIP(EMBED_DIM)
            return
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("dim") != EMBED_DIM or meta.get("model") != EMBED_MODEL:
                raise ValueError(f"索引维度/模型不匹配: {meta.get('dim')}/{meta.get('model')}")
            items = [MemoryItem(**it) for it in meta.get("items", [])]
            vec_map = [i for i, it in enumerate(items) if it.has_vec]
            if vec_map:
                if not idx_p.exists():
                    raise ValueError("有向量条目但索引文件缺失")
                index = faiss.read_index(str(idx_p))
                if index.ntotal != len(vec_map):
                    raise ValueError(f"索引({index.ntotal})与向量条目({len(vec_map)})数量不一致")
            else:
                index = faiss.IndexFlatIP(EMBED_DIM)
            self._index, self._items, self._vec_map = index, items, vec_map
            logger.info(f"长期记忆已加载: {len(items)} 条（{len(vec_map)} 条已向量化）")
        except Exception as e:
            logger.warning(f"长期记忆索引损坏，重建空库: {e}")
            self._index = faiss.IndexFlatIP(EMBED_DIM)
            self._items, self._vec_map = [], []

    def save(self) -> None:
        """原子成对落盘（先写临时文件再替换）。"""
        if self._index is None:
            return
        import faiss
        tmp_idx = self._index_path().with_suffix(".tmp")
        tmp_meta = self._meta_path().with_suffix(".tmp")
        faiss.write_index(self._index, str(tmp_idx))
        tmp_meta.write_text(json.dumps({
            "dim": EMBED_DIM, "model": EMBED_MODEL,
            "items": [vars(it) for it in self._items],
        }, ensure_ascii=False), encoding="utf-8")
        tmp_idx.replace(self._index_path())
        tmp_meta.replace(self._meta_path())

    # ---------- 写入 ----------

    async def add(self, text: str, chat_id: str, *, weight: float = 1.0, kind: str = "chat") -> bool:
        """入库。embedding 可用则向量化；不可用存纯文本条目（关键词可检索）。

        永远成功（返回 True），除非文本为空。
        """
        if not (text or "").strip():
            return False
        self.load()
        vec = await get_embedding_client().embed_one(text)
        item = MemoryItem(text=text, chat_id=chat_id, timestamp=time.time(),
                          weight=weight, kind=kind, has_vec=vec is not None)
        self._items.append(item)
        if vec is not None:
            v = np.array([vec], dtype="float32")
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            self._index.add(v)
            self._vec_map.append(len(self._items) - 1)
        self.save()
        return True

    # ---------- 检索 ----------

    async def search(self, query: str, *, top_k: int = 5,
                     chat_id: Optional[str] = None) -> List[MemoryItem]:
        """向量检索 + 纯文本条目关键词补充；embedding 不可用全走关键词。"""
        self.load()
        if not self._items:
            return []
        vec = await get_embedding_client().embed_one(query) if self._vec_map else None
        if vec is None:
            return self._keyword_search(query, top_k=top_k, chat_id=chat_id)

        v = np.array([vec], dtype="float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        k = min(top_k * 4, self._index.ntotal)
        scores, ids = self._index.search(v, k)
        out = []
        for score, pos in zip(scores[0], ids[0], strict=False):
            if pos < 0 or score < 0.3:
                continue
            item = self._items[self._vec_map[int(pos)]]
            if chat_id and item.chat_id != chat_id:
                continue
            out.append(item)
            if len(out) >= top_k:
                break
        # 纯文本条目关键词补充（向量检索覆盖不到它们）
        if len(out) < top_k:
            plain = [it for it in self._keyword_search(query, top_k=top_k, chat_id=chat_id)
                     if not it.has_vec and it not in out]
            out.extend(plain[:top_k - len(out)])
        return out

    def _keyword_search(self, query: str, *, top_k: int, chat_id: Optional[str]) -> List[MemoryItem]:
        """降级：2-gram 重叠计分。"""
        grams = {query[i:i + 2] for i in range(len(query) - 1)} or {query}
        scored = []
        for item in self._items:
            if chat_id and item.chat_id != chat_id:
                continue
            hits = sum(1 for g in grams if g in item.text)
            if hits:
                scored.append((hits, item))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:top_k]]

    # ---------- 遗忘 ----------

    def forget(self, *, max_age_days: float = 90, min_weight: float = 0.2) -> int:
        """删除过期低权重记忆并重建索引。返回删除数。"""
        self.load()
        if not self._items:
            return 0
        import faiss
        cutoff = time.time() - max_age_days * 86400
        keep_ids = [i for i, it in enumerate(self._items)
                    if not (it.timestamp < cutoff and it.weight < min_weight)]
        removed = len(self._items) - len(keep_ids)
        if not removed:
            return 0
        # 重建：向量条目从旧索引 reconstruct
        old_pos = {item_idx: pos for pos, item_idx in enumerate(self._vec_map)}
        new_index = faiss.IndexFlatIP(EMBED_DIM)
        new_items, new_vec_map = [], []
        vecs = []
        for i in keep_ids:
            item = self._items[i]
            new_items.append(item)
            if item.has_vec and i in old_pos:
                vecs.append(self._index.reconstruct(old_pos[i]))
                new_vec_map.append(len(new_items) - 1)
        if vecs:
            new_index.add(np.vstack(vecs))
        self._index, self._items, self._vec_map = new_index, new_items, new_vec_map
        self.save()
        logger.info(f"遗忘 {removed} 条记忆，索引已重建（{len(new_items)} 条保留）")
        return removed


_ltm: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    global _ltm
    if _ltm is None:
        _ltm = LongTermMemory()
    return _ltm
