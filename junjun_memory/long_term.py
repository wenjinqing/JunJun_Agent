"""长期记忆：faiss 向量库（落盘持久化）+ 关键词降级检索。

对齐原 memory_system 语义：
- faiss IndexFlatIP（余弦相似，向量归一化后内积）+ JSON 元数据成对落盘
- 索引头记录维度+模型名，不匹配拒绝加载并重建（防换模型炸索引）
- embedding 不可用时**写入仍成功**（纯文本条目，关键词可检索），
  向量条目与文本条目共存：invariant 为 index.ntotal == len(vec_map)
"""

import json
import threading
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


def _graph_cfg() -> dict:
    """[memory_graph] 配置（P2-20 图记忆扩散）；读取失败按空配置（默认开）。"""
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("memory_graph", {})
    except Exception:
        return {}


def _chat_allowed(item_chat: str, chat_id) -> bool:
    """chat_id 过滤：None 放行；str 精确匹配；集合/元组 任一命中
    （processor 检索传 (会话, "knowledge")，知识库条目也能被召回）。"""
    if chat_id is None:
        return True
    if isinstance(chat_id, str):
        return item_chat == chat_id
    return item_chat in chat_id


def _effective_weight(it: "MemoryItem") -> float:
    """时效衰减后的权重：每周 ×0.95（Ebbinghaus 式自然沉底）。

    pinned 不衰减（用户显式钉住的事不该自然遗忘）；其余条目不复习就沉底，
    被检索命中会 +0.05 复习强化（search 内）——「常用的记得牢，不用的慢慢忘」。
    此前 weight 只增不减，「90 天+低权重」遗忘口永远不触发（严厉审查 P2-10）。
    """
    if it.kind == "pinned":
        return it.weight
    age_weeks = max(0.0, (time.time() - it.timestamp) / 604800.0)
    return it.weight * (0.95 ** age_weeks)


def _recall_min_score() -> float:
    """向量召回相似度下限（[memory] recall_min_score，默认 0.55）。

    曾用 0.3——所用 embedding 模型对不相关中文文本的余弦相似度也有 0.4~0.6，
    0.3 形同虚设，「你忽然想起」注的都是弱相关噪声（严厉审查 P2-10）。
    """
    try:
        from junjun_core.config import get_global_config
        return float(get_global_config().raw.get("memory", {}).get("recall_min_score", 0.55))
    except Exception:
        return 0.55


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
        self._dirty = False                # 有未落盘变更（批量落盘用）
        # 2026-08-13 审查 P2：flush/forget/dedupe 移进 asyncio.to_thread 工作线程
        # 后，与事件循环上的 add/search 并发——共享态（_items/_index/_vec_map/
        # _dirty）全部走这把可重入锁（save/flush、forget/_rebuild/save 有嵌套）。
        # embed 网络等待一律在锁外，锁内只剩内存操作+落盘。
        self._lock = threading.RLock()

    # ---------- 持久化 ----------

    def _index_path(self) -> Path:
        return self._dir / "faiss_index.bin"

    def _meta_path(self) -> Path:
        return self._dir / "metadata.json"

    @staticmethod
    def _model_tag() -> str:
        """索引头模型名：用客户端实际模型（换模型触发重建），常量仅兜底。"""
        try:
            return get_embedding_client()._model or EMBED_MODEL
        except Exception:
            return EMBED_MODEL

    def load(self) -> None:
        """启动加载；索引与向量条目数不一致或维度/模型不匹配时重建。

        恢复策略：主文件损坏先回退 .bak 备份对（上一次良好落盘），
        备份也坏了才重建空库——记忆是不可再生资产，绝不一次损坏就清零。
        """
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            for meta_p, idx_p, tag in (
                (self._meta_path(), self._index_path(), "主文件"),
                (self._meta_path().with_suffix(".bak"), self._index_path().with_suffix(".bak"), "备份"),
            ):
                try:
                    if self._try_load(meta_p, idx_p):
                        if tag == "备份":
                            logger.warning("长期记忆主文件损坏，已从 .bak 备份恢复")
                        return
                except Exception as e:
                    logger.warning(f"长期记忆{tag}加载失败: {e}")
            logger.warning("长期记忆主文件与备份均不可用，重建空库")
            import faiss
            self._index = faiss.IndexFlatIP(EMBED_DIM)
            self._items, self._vec_map = [], []

    def _try_load(self, meta_p: Path, idx_p: Path) -> bool:
        """尝试从指定文件对加载。文件对不存在返回 False（静默），损坏抛异常。"""
        import faiss
        if not meta_p.exists():
            if not self._items and self._index is None:
                self._index = faiss.IndexFlatIP(EMBED_DIM)
            return meta_p.name != "metadata.json"  # 主文件不存在=新库可用；备份不存在=跳过
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("dim") != EMBED_DIM or meta.get("model") != self._model_tag():
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
        return True

    def save(self) -> None:
        """原子成对落盘（先写临时文件再替换）；替换成功后同步 .bak 备份对。

        备份在替换【之后】做：崩溃发生在替换中途时，.bak 仍是上一次
        完整良好状态（若在替换前备份，备份永远落后一代，白丢一条记忆）。
        """
        with self._lock:
            if self._index is None:
                return
            import faiss
            import shutil
            tmp_idx = self._index_path().with_suffix(".tmp")
            tmp_meta = self._meta_path().with_suffix(".tmp")
            faiss.write_index(self._index, str(tmp_idx))
            tmp_meta.write_text(json.dumps({
                "dim": EMBED_DIM, "model": self._model_tag(),
                "items": [vars(it) for it in self._items],
            }, ensure_ascii=False), encoding="utf-8")
            tmp_idx.replace(self._index_path())
            tmp_meta.replace(self._meta_path())
            # 主文件对落盘成功 -> 同步为备份对（加载失败时的回退点）
            for src in (self._index_path(), self._meta_path()):
                try:
                    shutil.copy2(src, src.with_suffix(".bak"))
                except Exception:
                    pass
            self._dirty = False

    def flush(self) -> None:
        """有脏数据才落盘（定时任务周期调用）。

        add() 不再每次全量落盘（faiss+JSON+双 .bak 的 MB 级同步写跑在事件
        循环上，条目越多越卡——O(n²) 写放大，严厉审查 P2-10）；代价是崩溃
        最多丢一个 flush 周期的记忆，可接受。
        """
        with self._lock:
            if self._dirty:
                self.save()

    # ---------- 写入 ----------

    async def add(self, text: str, chat_id: str, *, weight: float = 1.0, kind: str = "chat") -> bool:
        """入库。embedding 可用则向量化；不可用存纯文本条目（关键词可检索）。

        永远成功（返回 True），除非文本为空。
        """
        if not (text or "").strip():
            return False
        self.load()
        vec = await get_embedding_client().embed_one(text)   # 网络等待在锁外
        with self._lock:
            # 写入去重：同会话近义条目（向量相似 >0.92 或归一化文本相同）合并加权
            # 而非新插——否则 LLM 每轮都能把同一事实写 N 份挤占检索 top-k
            # （严厉审查 P2-10）。pinned 不参与合并（用户显式钉的每一条都算数）。
            norm = " ".join(text.split())
            for it in reversed(self._items[-200:]):
                if it.chat_id != chat_id or it.kind == "pinned":
                    continue
                if " ".join(it.text.split()) == norm:
                    it.weight = min(2.0, it.weight + 0.1)
                    it.timestamp = time.time()
                    self._dirty = True
                    return True
            if vec is not None and self._vec_map:
                v = np.array([vec], dtype="float32")
                v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                scores, ids = self._index.search(v, 1)
                if int(ids[0][0]) >= 0 and float(scores[0][0]) > 0.92:
                    dup = self._items[self._vec_map[int(ids[0][0])]]
                    if dup.chat_id == chat_id and dup.kind != "pinned":
                        dup.weight = min(2.0, dup.weight + 0.1)
                        dup.timestamp = time.time()
                        self._dirty = True
                        return True
            item = MemoryItem(text=text, chat_id=chat_id, timestamp=time.time(),
                              weight=weight, kind=kind, has_vec=vec is not None)
            self._items.append(item)
            if vec is not None:
                v = np.array([vec], dtype="float32")
                v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                self._index.add(v)
                self._vec_map.append(len(self._items) - 1)
            self._dirty = True   # 批量落盘：flush() 周期写，不再每次全量重写
            return True

    # ---------- 检索 ----------

    async def search(self, query: str, *, top_k: int = 5,
                     chat_id: Optional[str] = None,
                     spread: Optional[bool] = None) -> List[MemoryItem]:
        """向量检索 + 纯文本条目关键词补充；embedding 不可用全走关键词。

        spread（P2-20 图记忆扩散，只读增强）：命中条目作为种子沿记忆图
        PPR 扩散，追加 top 相关记忆（[memory_graph] max_spread 条）。
        None 走配置 [memory_graph] enable（默认开）；False 回滚扁平检索。
        """
        self.load()
        if not self._items:
            return []
        vec = await get_embedding_client().embed_one(query) if self._vec_map else None
        with self._lock:
            if vec is None:
                out = self._keyword_search(query, top_k=top_k, chat_id=chat_id)
            else:
                v = np.array([vec], dtype="float32")
                v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                k = min(top_k * 4, self._index.ntotal)
                scores, ids = self._index.search(v, k)
                # 复合打分：相关性 × 重要性（时效衰减后）——weight 不再只是遗忘
                # 参数，用户标重要的事召回概率就该更高（严厉审查 P2-10）
                min_score = _recall_min_score()
                cands = []
                for score, pos in zip(scores[0], ids[0], strict=False):
                    if pos < 0 or score < min_score:
                        continue
                    item = self._items[self._vec_map[int(pos)]]
                    if not _chat_allowed(item.chat_id, chat_id):
                        continue
                    w_norm = min(_effective_weight(item), 2.0) / 2.0   # 0..1
                    cands.append((float(score) * (0.7 + 0.3 * w_norm), item))
                cands.sort(key=lambda x: -x[0])
                out = [it for _, it in cands[:top_k]]
                # 纯文本条目关键词补充（向量检索覆盖不到它们）
                if len(out) < top_k:
                    plain = [it for it in self._keyword_search(query, top_k=top_k, chat_id=chat_id)
                             if not it.has_vec and it not in out]
                    out.extend(plain[:top_k - len(out)])

            if spread is None:
                spread = bool(_graph_cfg().get("enable", True))
            if spread and out:
                out = self._spread_related(out, chat_id=chat_id)
            # 检索即复习：被召回的条目权重微涨（对抗时效衰减，常用的记得牢）
            for it in out:
                it.weight = min(2.0, it.weight + 0.05)
            if out:
                self._dirty = True
            return out

    def _spread_related(self, seeds: List[MemoryItem], *,
                        chat_id) -> List[MemoryItem]:
        """PPR 图扩散：把种子的相关簇追加到结果尾部。任何失败返回原结果。"""
        try:
            from junjun_memory.memory_graph import get_memory_graph
            cfg = _graph_cfg()
            idx_of = {id(it): i for i, it in enumerate(self._items)}
            seed_ids = [idx_of[id(it)] for it in seeds if id(it) in idx_of]
            related = get_memory_graph().spread(
                self, seed_ids,
                top_k=int(cfg.get("max_spread", 2)),
                damping=float(cfg.get("damping", 0.85)),
            )
            in_out = {id(it) for it in seeds}
            picked = []
            for i in related:
                it = self._items[i]
                if id(it) in in_out:
                    continue
                if not _chat_allowed(it.chat_id, chat_id):
                    continue
                picked.append(it)
            if picked:
                logger.debug(f"图扩散补充 {len(picked)} 条相关记忆")
            return list(seeds) + picked
        except Exception as e:
            logger.debug(f"图扩散失败（返回原结果）: {e}")
            return seeds

    def _keyword_search(self, query: str, *, top_k: int, chat_id) -> List[MemoryItem]:
        """降级：2-gram 重叠计分。"""
        grams = {query[i:i + 2] for i in range(len(query) - 1)} or {query}
        scored = []
        for item in self._items:
            if not _chat_allowed(item.chat_id, chat_id):
                continue
            hits = sum(1 for g in grams if g in item.text)
            if hits:
                scored.append((hits, item))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:top_k]]

    # ---------- 遗忘 ----------

    def forget(self, *, max_age_days: float = 90, min_weight: float = 0.2,
               max_items: Optional[int] = None) -> int:
        """删除记忆并重建索引。返回删除数。

        两个淘汰口（权重均按时效衰减后的有效权重计算——不复习就沉底，
        老记忆终会被淘汰，遗忘口不再是摆设）：
        1. 过期低权重（90 天 + 有效权重 < min_weight；pinned 不衰减不受影响）
        2. 容量上限（真正的闸门）：超过 max_items 时按（有效权重低优先、
           时间老优先）淘汰到上限内——防 faiss/metadata/记忆图无界增长。
        max_items 默认读 [memory] ltm_max_items（5000）。
        """
        self.load()
        with self._lock:
            if not self._items:
                return 0
            import faiss
            if max_items is None:
                try:
                    from junjun_core.config import get_global_config
                    max_items = int(get_global_config().raw.get("memory", {}).get("ltm_max_items", 5000))
                except Exception:
                    max_items = 5000
            cutoff = time.time() - max_age_days * 86400
            keep_ids = [i for i, it in enumerate(self._items)
                        if not (it.timestamp < cutoff and _effective_weight(it) < min_weight)]
            if len(keep_ids) > max_items:
                ranked = sorted(keep_ids, key=lambda i: (_effective_weight(self._items[i]),
                                                         self._items[i].timestamp))
                drop = set(ranked[:len(keep_ids) - max_items])
                keep_ids = [i for i in keep_ids if i not in drop]
            removed = len(self._items) - len(keep_ids)
            if not removed:
                return 0
            self._rebuild(keep_ids)
            logger.info(f"遗忘 {removed} 条记忆，索引已重建（{len(keep_ids)} 条保留）")
            return removed

    def remove_where(self, pred) -> int:
        """按谓词删除记忆并重建索引（如 /forget 关键词清理）。返回删除数。"""
        self.load()
        with self._lock:
            keep_ids = [i for i, it in enumerate(self._items) if not pred(it)]
            removed = len(self._items) - len(keep_ids)
            if not removed:
                return 0
            self._rebuild(keep_ids)
            logger.info(f"按条件删除 {removed} 条记忆（{len(keep_ids)} 条保留）")
            return removed

    def dedupe(self, *, threshold: Optional[float] = None) -> int:
        """全局近重合并（夜间整理）。返回合并掉的条目数。

        写入期去重（add 内）只吃全库 top-1 + 近 200 条窗口，跨窗口/分批写入
        的同一事实会漏网（典型：摘要系统隔几天用不同措辞沉淀同一件事）。
        这里全库逐条找近邻合并。安全纪律（记忆是不可再生资产，宁漏勿错杀）：
        - 阈值默认 0.95，比写入期（0.92）更严
        - 只并同会话、非 pinned 条目（用户钉的每一条都算数）
        - 保留方取权重 max+0.1（封顶 2.0）、时间戳取新，文本留先见的一条
        """
        self.load()
        with self._lock:
            # 全库逐条近邻扫描持锁（faiss 读写不可跨线程并发）；每日一次，
            # 最坏秒级停顿——原来整个跑在事件循环上更糟（2026-08-13 审查 P2）
            if threshold is None:
                try:
                    from junjun_core.config import get_global_config
                    threshold = float(get_global_config().raw.get("memory", {})
                                      .get("dedupe_threshold", 0.95))
                except Exception:
                    threshold = 0.95
            drop: set = set()
            # 向量条目：全库近邻合并
            for pos, item_idx in enumerate(self._vec_map):
                if item_idx in drop:
                    continue
                cur = self._items[item_idx]
                if cur.kind == "pinned":
                    continue
                v = self._index.reconstruct(pos).reshape(1, -1)
                scores, ids = self._index.search(v, 6)
                for score, npos in zip(scores[0], ids[0], strict=False):
                    npos = int(npos)
                    if npos < 0 or npos == pos or float(score) < threshold:
                        continue
                    nidx = self._vec_map[npos]
                    if nidx in drop:
                        continue
                    nit = self._items[nidx]
                    if nit.chat_id != cur.chat_id or nit.kind == "pinned":
                        continue
                    cur.weight = min(2.0, max(cur.weight, nit.weight) + 0.1)
                    cur.timestamp = max(cur.timestamp, nit.timestamp)
                    drop.add(nidx)
            # 纯文本条目（embedding 不可用期写入的）：归一化全文精确合并
            seen: dict = {}
            for i, it in enumerate(self._items):
                if i in drop or it.has_vec or it.kind == "pinned":
                    continue
                key = (it.chat_id, " ".join(it.text.split()))
                if key in seen:
                    cur = self._items[seen[key]]
                    cur.weight = min(2.0, max(cur.weight, it.weight) + 0.1)
                    cur.timestamp = max(cur.timestamp, it.timestamp)
                    drop.add(i)
                else:
                    seen[key] = i
            if not drop:
                return 0
            self._rebuild([i for i in range(len(self._items)) if i not in drop])
            logger.info(f"夜间整理合并 {len(drop)} 条近重记忆（阈值 {threshold}，"
                        f"{len(self._items)} 条保留）")
            return len(drop)


    def pinned(self, chat_id: str, *, limit: int = 50) -> List[MemoryItem]:
        """用户钉住的记忆（/记住、pin_memory，kind="pinned"）：每轮优先注入，
        不占语义召回额度（P6-2 用户可控记忆）。"""
        self.load()
        with self._lock:
            out = [it for it in self._items if it.kind == "pinned" and it.chat_id == chat_id]
            return out[-limit:]

    def _rebuild(self, keep_ids: list) -> None:
        """按保留下标重建 items + faiss 索引并落盘。向量条目从旧索引 reconstruct。"""
        import faiss
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


_ltm: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    global _ltm
    if _ltm is None:
        _ltm = LongTermMemory()
    return _ltm
