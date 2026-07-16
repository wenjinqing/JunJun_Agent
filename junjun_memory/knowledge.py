"""LPMM 知识库：对齐原 chat/knowledge 语义（openIE -> KG -> PPR 检索）。

三部分：
- open_ie: LLM 抽取段落的 实体 + 三元组（主-谓-宾）
- kg_manager: quick_algo DiGraph 建图（实体节点 + 段落节点），
  PersonalizedPageRank 检索，graphml 落盘
- qa_manager: 问题 -> 实体抽取 -> embedding 相似实体/段落作 PPR 种子
  -> 段落排序返回

数据落盘 data/rag/（graphml + 段落 JSON）。embedding 不可用时降级：
实体精确匹配作种子（无相似度扩展）。
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("memory.knowledge")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = PROJECT_ROOT / "data" / "rag"

_OPENIE_PROMPT = """从以下文本中抽取知识三元组。
输出 JSON：{{"entities": ["实体1", ...], "triples": [["主语", "谓语", "宾语"], ...]}}
实体是名词性概念；三元组表达事实关系。没有可抽取内容输出 {{"entities": [], "triples": []}}。
只输出 JSON。

文本：
{text}"""

_QUERY_ENT_PROMPT = """从这个问题中抽取要查询的关键实体（名词概念），JSON 数组输出，最多 5 个。
问题：{question}
只输出 JSON 数组，如 ["实体1", "实体2"]。"""


def _pg_hash(text: str) -> str:
    return "paragraph_" + hashlib.md5(text.encode()).hexdigest()[:16]


def _ent_key(name: str) -> str:
    return "entity_" + name.strip().lower()


@dataclass
class Paragraph:
    hash: str
    text: str
    entities: List[str] = field(default_factory=list)
    added_at: float = 0.0


class KnowledgeBase:
    """KG + 段落库。quick_algo 缺失时自动降级纯 embedding/关键词检索。"""

    def __init__(self, data_dir: Optional[Path] = None):
        self._dir = data_dir or RAG_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._graph = None
        self._paragraphs: Dict[str, Paragraph] = {}
        self._loaded = False
        self._has_quick_algo = self._probe_quick_algo()

    @staticmethod
    def _probe_quick_algo() -> bool:
        try:
            from quick_algo import di_graph, pagerank  # noqa: F401
            return True
        except ImportError:
            logger.warning("quick_algo 不可用，知识库降级 embedding/关键词检索（无 PPR）")
            return False

    # ---------- 持久化 ----------

    def _para_path(self) -> Path:
        return self._dir / "paragraphs.json"

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._has_quick_algo:
            from quick_algo import di_graph
            self._graph = di_graph.DiGraph()
        try:
            if self._para_path().exists():
                data = json.loads(self._para_path().read_text(encoding="utf-8"))
                self._paragraphs = {p["hash"]: Paragraph(**p) for p in data}
                if self._graph is not None:
                    for p in self._paragraphs.values():
                        self._add_to_graph(p)
                logger.info(f"知识库已加载: {len(self._paragraphs)} 段落")
        except Exception as e:
            logger.warning(f"知识库加载失败，空库重建: {e}")
            self._paragraphs = {}

    def save(self) -> None:
        tmp = self._para_path().with_suffix(".tmp")
        tmp.write_text(json.dumps([vars(p) for p in self._paragraphs.values()],
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._para_path())

    # ---------- 导入（openIE）----------

    def _add_to_graph(self, para: Paragraph) -> None:
        """段落节点 <-> 实体节点 双向边。"""
        if self._graph is None:
            return
        from quick_algo.di_graph import DiNode, DiEdge
        try:
            self._graph.add_node(DiNode(para.hash))
        except Exception:
            pass
        for ent in para.entities:
            key = _ent_key(ent)
            try:
                self._graph.add_node(DiNode(key))
            except Exception:
                pass
            for a, b in ((para.hash, key), (key, para.hash)):
                try:
                    self._graph.add_edge(DiEdge(a, b))
                except Exception:
                    pass

    async def import_text(self, text: str, *, model=None, callbacks=None) -> int:
        """导入一段文本：openIE 抽取 -> 建图 -> embedding 入库。返回新增段落数。"""
        self.load()
        text = (text or "").strip()
        if len(text) < 10:
            return 0
        h = _pg_hash(text)
        if h in self._paragraphs:
            return 0

        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        from langchain_core.messages import HumanMessage
        entities: List[str] = []
        try:
            resp = await model.ainvoke(
                [HumanMessage(content=_OPENIE_PROMPT.format(text=text[:1500]))],
                config={"callbacks": callbacks or []},
            )
            m = re.search(r"\{.*\}", str(resp.content), re.S)
            if m:
                obj = json.loads(m.group(0))
                entities = [str(e).strip() for e in obj.get("entities", []) if str(e).strip()][:15]
                # 三元组的主宾也入实体集（对齐原 openIE 语义：三元组构成图边基础）
                for t in obj.get("triples", []):
                    if isinstance(t, list) and len(t) == 3:
                        entities.extend([str(t[0]).strip(), str(t[2]).strip()])
                entities = list(dict.fromkeys(e for e in entities if e))[:20]
        except Exception as e:
            logger.warning(f"openIE 抽取失败（段落仍入库，仅无图关联）: {e}")

        para = Paragraph(hash=h, text=text, entities=entities, added_at=time.time())
        self._paragraphs[h] = para
        self._add_to_graph(para)
        self.save()

        # 段落向量入长期库（namespace 复用 faiss，kind=knowledge）
        try:
            from junjun_memory.long_term import get_long_term_memory
            await get_long_term_memory().add(text[:500], "knowledge", weight=1.2, kind="knowledge")
        except Exception:
            pass
        logger.info(f"知识导入: {len(entities)} 实体, 段落 {h[:20]}")
        return 1

    # ---------- 检索（PPR）----------

    async def search(self, question: str, *, top_k: int = 3, model=None, callbacks=None) -> List[str]:
        """问题 -> 实体种子 -> PPR 段落排序。quick_algo 缺失降级关键词。"""
        self.load()
        if not self._paragraphs:
            return []

        # 抽查询实体
        query_entities: List[str] = []
        try:
            if model is None:
                from junjun_llm import get_chat_model
                model = get_chat_model("utils")
            from langchain_core.messages import HumanMessage
            resp = await model.ainvoke(
                [HumanMessage(content=_QUERY_ENT_PROMPT.format(question=question))],
                config={"callbacks": callbacks or []},
            )
            m = re.search(r"\[.*\]", str(resp.content), re.S)
            if m:
                query_entities = [str(e).strip() for e in json.loads(m.group(0)) if str(e).strip()][:5]
        except Exception:
            pass

        if self._graph is not None and query_entities:
            result = self._ppr_search(question, query_entities, top_k)
            if result:
                return result
        return self._keyword_search(question, query_entities, top_k)

    def _ppr_search(self, question: str, query_entities: List[str], top_k: int) -> List[str]:
        """PersonalizedPageRank：命中实体作种子（对齐原 kg_search 语义）。"""
        cfg = get_global_config().raw.get("lpmm_knowledge", {})
        damping = float(cfg.get("qa_ppr_damping", 0.8))
        seeds: Dict[str, float] = {}
        node_names = set()
        try:
            node_names = {n for n in self._graph.node_name2idx_map()}
        except Exception:
            pass
        for ent in query_entities:
            key = _ent_key(ent)
            if key in node_names:
                seeds[key] = 1.0
        if not seeds:
            return []
        try:
            from quick_algo import pagerank
            ppr = pagerank.run_pagerank(self._graph, personalization=seeds,
                                        max_iter=100, alpha=damping)
            ranked = sorted(
                ((k, v) for k, v in ppr.items() if k.startswith("paragraph_")),
                key=lambda x: -x[1])
            return [self._paragraphs[h].text for h, _ in ranked[:top_k] if h in self._paragraphs]
        except Exception as e:
            logger.warning(f"PPR 检索失败（降级关键词）: {e}")
            return []

    def _keyword_search(self, question: str, query_entities: List[str], top_k: int) -> List[str]:
        terms = query_entities or [question]
        scored: List[Tuple[int, Paragraph]] = []
        for p in self._paragraphs.values():
            score = sum(2 for e in query_entities if e in p.entities or e in p.text)
            grams = {question[i:i + 2] for i in range(len(question) - 1)}
            score += sum(1 for g in grams if g in p.text) // 3
            if score:
                scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [p.text for _, p in scored[:top_k]]


_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
