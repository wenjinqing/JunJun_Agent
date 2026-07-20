"""表达反思：对齐原 express/expression_reflector 语义。

定期（10-15 分钟随机间隔）把最近学到的表达发给管理员（reflect_operator_id）
求证是否恰当；管理员回复「删除 N」则移除该表达。
配置 [expression] reflect / reflect_operator_id（"qq:123:private" 格式）。
"""

import random
import time
from typing import Dict, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.reflector")


def _cfg() -> dict:
    return get_global_config().raw.get("expression", {})


class ExpressionReflector:
    def __init__(self):
        # 冷启动保护：初始设为当前时间，避免进程一启动就立刻给管理员发求证
        self.last_ask_time: float = time.time()
        self._pending: Dict[int, int] = {}  # 提问序号 -> Expression.id

    def _interval(self) -> float:
        return random.uniform(10 * 60, 15 * 60)

    def enabled(self) -> bool:
        return bool(_cfg().get("reflect", False)) and bool(_cfg().get("reflect_operator_id", ""))

    async def check_and_ask(self) -> bool:
        """调度器任务：到间隔且有新表达时向管理员提问。"""
        if not self.enabled():
            return False
        now = time.time()
        if now - self.last_ask_time < self._interval():
            return False

        from junjun_core.database import Expression
        recent = list(Expression.select()
                      .where(Expression.last_active_time > self.last_ask_time)
                      .order_by(Expression.last_active_time.desc()).limit(5))
        if not recent:
            return False
        self.last_ask_time = now

        self._pending = {i + 1: r.id for i, r in enumerate(recent)}
        lines = ["我最近学了这些表达，帮我看看有没有不合适的（回「删除 编号」清掉）："]
        for i, r in enumerate(recent):
            lines.append(f"{i + 1}. [{r.situation}]「{r.style}」")

        operator = str(_cfg().get("reflect_operator_id", ""))
        parts = operator.split(":")
        if len(parts) != 3:
            logger.warning(f"reflect_operator_id 格式错误: {operator}")
            return False
        platform, target_id, kind = parts

        from junjun_core.contracts import ReplySet, ReplySegment
        from junjun_core.gateway.router import get_gateway
        await get_gateway().send_reply(ReplySet(
            platform=platform,
            target_group_id=target_id if kind == "group" else None,
            target_user_id=target_id if kind != "group" else None,
            segments=[ReplySegment(type="text", data="\n".join(lines))],
            should_reply=True,
        ))
        logger.info(f"表达反思提问已发 -> {operator}（{len(recent)} 条）")
        return True

    def handle_operator_reply(self, chat_id: str, text: str) -> Optional[str]:
        """管理员回复处理：命中「删除 N」删表达。返回回执文本（非管理员消息返回 None）。"""
        operator = str(_cfg().get("reflect_operator_id", ""))
        if not operator or chat_id != operator or not self._pending:
            return None
        import re
        # 支持半角/全角数字，以及「删」「删掉」「删除」等前缀
        m = re.search(r"删(?:除|掉)?\s*([0-9０-９]+)", text)
        if not m:
            return None
        idx = int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789")))
        expr_id = self._pending.pop(idx, None)
        if expr_id is None:
            return f"编号 {idx} 不在待确认列表里。"
        from junjun_core.database import Expression
        n = Expression.delete().where(Expression.id == expr_id).execute()
        logger.info(f"表达反思: 管理员删除表达 id={expr_id}")
        return "已删掉，谢谢指正～" if n else "该表达已不存在。"


expression_reflector = ExpressionReflector()
