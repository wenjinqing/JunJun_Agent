"""黑话识别与记忆：对齐原 jargon/jargon_explainer 语义。

- match_jargon_from_text: 上下文渲染时标注已知黑话解释
- record_jargon: 记录/强化黑话（count 累积）
- all_global=true 时黑话跨会话共享（chat_id 存空）
"""

from typing import Dict, List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.jargon")


def _is_global() -> bool:
    return bool(get_global_config().raw.get("jargon", {}).get("all_global", True))


def _enabled() -> bool:
    return bool(get_global_config().raw.get("memory", {}).get("enable_jargon_detection", True))


def record_jargon(term: str, explanation: str, chat_id: str = "") -> None:
    """记录黑话；已存在则计数+1 并更新解释（新解释非空时）。"""
    if not _enabled() or not term:
        return
    from junjun_core.database import Jargon, db
    store_chat = "" if _is_global() else chat_id
    with db.atomic():
        row = Jargon.get_or_none((Jargon.term == term) & (Jargon.chat_id == store_chat))
        if row is None:
            Jargon.create(term=term, explanation=explanation, chat_id=store_chat, count=1)
            logger.info(f"新黑话记录: {term} = {explanation[:30]}")
        else:
            row.count += 1
            if explanation:
                row.explanation = explanation
            row.save()


def lookup_jargon(term: str, chat_id: str = "") -> Optional[str]:
    from junjun_core.database import Jargon
    q = Jargon.select().where(Jargon.term == term)
    if not _is_global():
        q = q.where(Jargon.chat_id.in_(["", chat_id]))
    row = q.first()
    return row.explanation if row else None


def match_jargon_from_text(text: str, chat_id: str = "", *, max_hits: int = 3) -> List[Dict]:
    """扫描文本中的已知黑话。返回 [{term, explanation}]。

    黑话库通常很小（<几千条），全量拉 term 做 in 匹配即可。
    """
    if not _enabled() or not text:
        return []
    from junjun_core.database import Jargon
    q = Jargon.select(Jargon.term, Jargon.explanation).where(Jargon.count >= 2)  # 出现≥2次才可信
    if not _is_global():
        q = q.where(Jargon.chat_id.in_(["", chat_id]))
    hits = []
    for row in q:
        if row.term in text:
            hits.append({"term": row.term, "explanation": row.explanation})
            if len(hits) >= max_hits:
                break
    return hits


def build_jargon_block(text: str, chat_id: str = "") -> str:
    """拼 prompt 黑话块；无命中返回空串。"""
    hits = match_jargon_from_text(text, chat_id)
    if not hits:
        return ""
    lines = ["群黑话释义："]
    for h in hits:
        lines.append(f"- 「{h['term']}」= {h['explanation']}")
    return "\n".join(lines)
