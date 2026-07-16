"""话题摘要（中期记忆）：对齐原 hippo_memorizer/chat_history_summarizer 语义。

机制：
- 每会话累积消息批次，满 N 条或超时间窗触发摘要
- utils 小模型提取话题摘要 -> 写长期记忆（kind="summary"）+ 落盘 JSON
- 摘要失败静默，批次保留至下次
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.summarizer")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIPPO_DIR = PROJECT_ROOT / "data" / "hippo"

BATCH_SIZE = 25          # 满 N 条触发
BATCH_MAX_AGE = 3600.0   # 或首条消息超 1h 触发

_SUMMARY_PROMPT = """以下是 QQ 群/私聊的一段对话记录。请提取 1-3 条值得长期记住的信息，
每条一行，格式「主题: 具体内容（涉及的人）」。只提取有记忆价值的（谁喜欢什么、发生了什么事、
约定/计划、关系变化），纯水聊输出「无」。

对话：
{conversation}"""


@dataclass
class TopicBatch:
    lines: List[str] = field(default_factory=list)
    started_at: float = 0.0


class ChatSummarizer:
    def __init__(self, data_dir: Optional[Path] = None):
        self._dir = data_dir or HIPPO_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._batches: Dict[str, TopicBatch] = {}

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

    async def summarize(self, chat_id: str, *, model=None, callbacks=None) -> List[str]:
        """摘要当前批次并写长期记忆。返回摘要行（失败/无价值为空列表）。"""
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
            return []
        summaries = [line.strip() for line in raw.splitlines()
                     if line.strip() and line.strip() != "无"][:3]

        # 写长期记忆 + 落盘
        from junjun_memory.long_term import get_long_term_memory
        ltm = get_long_term_memory()
        for s in summaries:
            await ltm.add(s, chat_id, weight=1.5, kind="summary")
        self._persist(chat_id, summaries)
        logger.info(f"[{chat_id}] 话题摘要 {len(summaries)} 条")
        return summaries

    def _persist(self, chat_id: str, summaries: List[str]) -> None:
        safe = chat_id.replace(":", "_")
        p = self._dir / f"{safe}.json"
        try:
            existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        except Exception:
            existing = []
        existing.extend({"time": time.time(), "summary": s} for s in summaries)
        p.write_text(json.dumps(existing[-200:], ensure_ascii=False, indent=1), encoding="utf-8")


_summarizer: Optional[ChatSummarizer] = None


def get_summarizer() -> ChatSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = ChatSummarizer()
    return _summarizer
