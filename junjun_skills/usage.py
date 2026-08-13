"""工具使用统计落库（2026-08-13 审查 P1）：registry 错误包装层单点挂钩。

铁律：统计绝不挡工具主路径——任何异常静默吞掉（写走 db_writer 单写队列，
未启动时直写，测试/脚本场景自然兼容）。
"""

import time

from junjun_core.observability import get_logger

logger = get_logger("skills.usage")


def record(tool: str, ok: bool, error_kind: str = "", chat_id: str = "") -> None:
    """记录一次工具调用（成功/失败+类别）。永不抛异常。"""
    try:
        from junjun_core.database import db_writer
        db_writer.submit(_insert, tool, ok, error_kind, chat_id, time.time())
    except Exception:
        pass


def _insert(tool: str, ok: bool, error_kind: str, chat_id: str, ts: float) -> None:
    try:
        from junjun_core.database.models import ToolUsage
        ToolUsage.create(time=ts, tool=tool, ok=ok,
                         error_kind=error_kind, chat_id=chat_id)
    except Exception as e:
        logger.warning(f"工具统计落库失败（忽略）: {type(e).__name__}: {e}")
