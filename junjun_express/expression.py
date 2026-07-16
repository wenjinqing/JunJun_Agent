"""表达学习：对齐原 express/expression_learner + expression_selector 语义。

- learner: 定期从群聊消息提取他人表达方式（situation 语境 -> style 句式），
  存 Expression 表，重复出现 count+1
- selector: 回复前按当前语境选适配表达注入 prompt（学了要用）
- learning_list 按会话配置强度；expression_groups 跨会话共享（简化：全局共享）
"""

import json
import re
import time
from typing import List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.expression")

_LEARN_PROMPT = """以下是 QQ 群聊片段。找出其中有特色的表达方式（口头禅、句式、梗的用法），
提取 0-3 条。每条格式：{{"situation": "什么语境下", "style": "怎么表达"}}
例：{{"situation": "表示震惊", "style": "我直接一个爆炸"}}
没有特色表达就输出 []。只输出 JSON 数组。

聊天片段：
{conversation}"""

_MIN_LEARN_BATCH = 15   # 累积消息数触发学习
_MAX_INJECT = 3         # 注入 prompt 的表达数上限


class ExpressionLearner:
    def __init__(self):
        self._buffers: dict = {}  # chat_id -> list[str]

    def _enabled(self, chat_id: str) -> bool:
        rules = get_global_config().raw.get("expression", {}).get("learning_list", [])
        if not rules:
            return True  # 未配置默认全开
        for r in rules:
            target = str(r.get("target", r)) if isinstance(r, dict) else str(r)
            if target in ("", chat_id):
                return True
        return False

    def note(self, chat_id: str, nickname: str, text: str) -> bool:
        """积累群友消息（bot 自己的不学）。返回 True 表示应触发学习。"""
        if not self._enabled(chat_id) or len(text) < 4:
            return False
        buf = self._buffers.setdefault(chat_id, [])
        buf.append(f"{nickname}: {text}")
        return len(buf) >= _MIN_LEARN_BATCH

    async def learn(self, chat_id: str, *, model=None, callbacks=None) -> int:
        """LLM 提取表达并入库。返回学到条数。"""
        buf = self._buffers.get(chat_id) or []
        if len(buf) < 5:
            return 0
        conversation = "\n".join(buf)
        self._buffers[chat_id] = []
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        from langchain_core.messages import HumanMessage
        try:
            resp = await model.ainvoke(
                [HumanMessage(content=_LEARN_PROMPT.format(conversation=conversation))],
                config={"callbacks": callbacks or []},
            )
            m = re.search(r"\[.*\]", str(resp.content), re.S)
            entries = json.loads(m.group(0)) if m else []
        except Exception as e:
            logger.warning(f"[{chat_id}] 表达学习失败（本批丢弃）: {e}")
            return 0

        from junjun_core.database import Expression
        learned = 0
        for e in entries[:3]:
            situation = str(e.get("situation", "")).strip()[:100]
            style = str(e.get("style", "")).strip()[:200]
            if not situation or not style:
                continue
            row = Expression.get_or_none(
                (Expression.chat_id == chat_id) & (Expression.style == style))
            if row:
                row.count += 1
                row.last_active_time = time.time()
                row.save()
            else:
                Expression.create(chat_id=chat_id, situation=situation, style=style,
                                  count=1, last_active_time=time.time())
                learned += 1
        if learned:
            logger.info(f"[{chat_id}] 学到 {learned} 条新表达")
        return learned


def select_expressions(chat_id: str, context: str, *, top_k: int = _MAX_INJECT) -> List[dict]:
    """按语境关键词选适配表达（count 加权）。跨会话共享：本会话优先，全局补充。"""
    from junjun_core.database import Expression
    rows = list(Expression.select().order_by(Expression.count.desc()).limit(100))
    if not rows:
        return []
    own = [r for r in rows if r.chat_id == chat_id]
    others = [r for r in rows if r.chat_id != chat_id]

    def score(r):
        s = r.count
        # 语境关键词重叠加分
        for w in (r.situation or "").split():
            if w and w in context:
                s += 5
        return s

    ranked = sorted(own, key=score, reverse=True)[:top_k]
    if len(ranked) < top_k:
        ranked += sorted(others, key=score, reverse=True)[:top_k - len(ranked)]
    return [{"situation": r.situation, "style": r.style} for r in ranked]


def build_expression_block(chat_id: str, context: str) -> str:
    """拼 prompt 表达块；无数据返回空串。"""
    exprs = select_expressions(chat_id, context)
    if not exprs:
        return ""
    lines = ["群友的表达方式参考（自然融入，别生搬硬套）："]
    for e in exprs:
        lines.append(f"- {e['situation']}时：「{e['style']}」")
    return "\n".join(lines)


expression_learner = ExpressionLearner()
