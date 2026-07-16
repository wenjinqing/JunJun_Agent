"""话题摘要（中期记忆）：对齐原 hippo_memorizer/chat_history_summarizer 语义。

机制（含话题缓存累积，对齐原 TopicCacheItem）：
- 每会话累积消息批次，满 N 条或超时间窗触发摘要
- 摘要产出「话题: 内容」行 -> 并入话题缓存：同话题跨批次累积（update_count+1，
  内容由 LLM 合并压缩），新话题建缓存项
- 话题达到 FLUSH_UPDATES 次更新（热话题成熟）或超过 TOPIC_TTL 未再更新（冷话题
  定稿）时，写长期记忆并从缓存移除
- 落盘 data/hippo/<chat_id>.json（话题缓存 + 已归档摘要）
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.summarizer")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIPPO_DIR = PROJECT_ROOT / "data" / "hippo"

BATCH_SIZE = 25          # 满 N 条触发
BATCH_MAX_AGE = 3600.0   # 或首条消息超 1h 触发
FLUSH_UPDATES = 3        # 话题更新满 N 次 -> 成熟落长期库
TOPIC_TTL = 6 * 3600.0   # 话题 N 秒未更新 -> 定稿落库

_SUMMARY_PROMPT = """以下是 QQ 群/私聊的一段对话记录。请提取 1-3 条值得长期记住的信息，
每条一行，格式「主题: 具体内容（涉及的人）」。只提取有记忆价值的（谁喜欢什么、发生了什么事、
约定/计划、关系变化），纯水聊输出「无」。

对话：
{conversation}"""

_MERGE_PROMPT = """同一话题的两段记忆，合并压缩成一条（保留双方信息，一句话）：
旧：{old}
新：{new}
只输出合并后的一句话。"""


@dataclass
class TopicBatch:
    lines: List[str] = field(default_factory=list)
    started_at: float = 0.0


@dataclass
class TopicCacheItem:
    """话题缓存项（对齐原 hippo TopicCacheItem）。"""
    title: str
    content: str
    update_count: int = 1
    updated_at: float = field(default_factory=time.time)


class ChatSummarizer:
    def __init__(self, data_dir: Optional[Path] = None):
        self._dir = data_dir or HIPPO_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._batches: Dict[str, TopicBatch] = {}
        self._topics: Dict[str, Dict[str, TopicCacheItem]] = {}  # chat_id -> {title: item}
        self._topics_loaded: set = set()

    # ---------- 批次 ----------

    def _batch(self, chat_id: str) -> TopicBatch:
        if chat_id not in self._batches:
            self._batches[chat_id] = TopicBatch(started_at=time.time())
        return self._batches[chat_id]

    def note(self, chat_id: str, nickname: str, text: str) -> bool:
        """记录一条消息。返回 True 表示批次已满应触发摘要。"""
        b = self._batch(chat_id)
        if not b.lines:
            b.started_at = time.time()
        b.lines.append(f"{nickname}: {text}")
        return len(b.lines) >= BATCH_SIZE or (time.time() - b.started_at) > BATCH_MAX_AGE

    # ---------- 话题缓存持久化 ----------

    def _file(self, chat_id: str) -> Path:
        return self._dir / f"{chat_id.replace(':', '_')}.json"

    def _load_topics(self, chat_id: str) -> Dict[str, TopicCacheItem]:
        if chat_id not in self._topics_loaded:
            self._topics_loaded.add(chat_id)
            try:
                data = json.loads(self._file(chat_id).read_text(encoding="utf-8"))
                self._topics[chat_id] = {
                    t["title"]: TopicCacheItem(**t) for t in data.get("topics", [])}
            except Exception:
                self._topics[chat_id] = {}
        return self._topics.setdefault(chat_id, {})

    def _persist(self, chat_id: str, archived: Optional[List[str]] = None) -> None:
        p = self._file(chat_id)
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            data = {}
        data["topics"] = [asdict(t) for t in self._topics.get(chat_id, {}).values()]
        if archived:
            hist = data.get("archived", [])
            hist.extend({"time": time.time(), "summary": s} for s in archived)
            data["archived"] = hist[-200:]
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 摘要 + 话题合并 ----------

    @staticmethod
    def _split_title(line: str) -> tuple:
        m = re.match(r"[「]?([^:：「」]{1,20})[:：]\s*(.+)", line)
        return (m.group(1).strip(), m.group(2).strip()) if m else (line[:12], line)

    async def summarize(self, chat_id: str, *, model=None, callbacks=None) -> List[str]:
        """摘要当前批次并并入话题缓存；成熟/过期话题落长期库。返回本批摘要行。"""
        b = self._batches.get(chat_id)
        if not b or len(b.lines) < 5:  # 太少不值得烧 token
            return []
        conversation = "\n".join(b.lines)
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        from langchain_core.messages import HumanMessage
        try:
            resp = await model.ainvoke(
                [HumanMessage(content=_SUMMARY_PROMPT.format(conversation=conversation))],
                config={"callbacks": callbacks or []},
            )
            raw = str(resp.content).strip()
        except Exception as e:
            logger.warning(f"[{chat_id}] 话题摘要失败（批次保留）: {e}")
            return []

        # 无论结果如何批次已消费
        self._batches[chat_id] = TopicBatch(started_at=time.time())

        if not raw or raw == "无":
            await self._flush_stale(chat_id)
            return []
        summaries = [line.strip() for line in raw.splitlines()
                     if line.strip() and line.strip() != "无"][:3]

        # 并入话题缓存（同标题累积，新标题建项）
        topics = self._load_topics(chat_id)
        for s in summaries:
            title, content = self._split_title(s)
            existing = topics.get(title)
            if existing is None:
                topics[title] = TopicCacheItem(title=title, content=content)
            else:
                existing.content = await self._merge(existing.content, content, model, callbacks)
                existing.update_count += 1
                existing.updated_at = time.time()

        await self._flush_mature(chat_id)
        await self._flush_stale(chat_id)
        self._persist(chat_id)
        logger.info(f"[{chat_id}] 话题摘要 {len(summaries)} 条（缓存 {len(topics)} 话题）")
        return summaries

    async def _merge(self, old: str, new: str, model, callbacks) -> str:
        if new in old:
            return old
        try:
            from langchain_core.messages import HumanMessage
            resp = await model.ainvoke(
                [HumanMessage(content=_MERGE_PROMPT.format(old=old, new=new))],
                config={"callbacks": callbacks or []},
            )
            merged = str(resp.content).strip().splitlines()[0]
            return merged[:200] if merged else f"{old}；{new}"
        except Exception:
            return f"{old}；{new}"[:200]

    async def _flush_mature(self, chat_id: str) -> None:
        """热话题（更新满 FLUSH_UPDATES 次）落长期库。"""
        topics = self._topics.get(chat_id, {})
        mature = [t for t in topics.values() if t.update_count >= FLUSH_UPDATES]
        await self._archive(chat_id, mature)

    async def _flush_stale(self, chat_id: str) -> None:
        """冷话题（超 TTL 未更新）定稿落库。"""
        topics = self._topics.get(chat_id, {})
        now = time.time()
        stale = [t for t in topics.values() if now - t.updated_at > TOPIC_TTL]
        await self._archive(chat_id, stale)

    async def _archive(self, chat_id: str, items: List[TopicCacheItem]) -> None:
        if not items:
            return
        from junjun_memory.long_term import get_long_term_memory
        ltm = get_long_term_memory()
        archived = []
        topics = self._topics.get(chat_id, {})
        for t in items:
            text = f"{t.title}: {t.content}"
            # 热度加权：更新越多权重越高
            await ltm.add(text, chat_id, weight=1.0 + 0.3 * t.update_count, kind="summary")
            archived.append(text)
            topics.pop(t.title, None)
        self._persist(chat_id, archived=archived)
        logger.info(f"[{chat_id}] 归档 {len(archived)} 个话题到长期记忆")


_summarizer: Optional[ChatSummarizer] = None


def get_summarizer() -> ChatSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = ChatSummarizer()
    return _summarizer
