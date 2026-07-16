"""统计任务：对齐原 chat/statistics 的 StatisticOutputTask 语义（简化）。

定期输出运行统计：消息量 / 回复量 / token 用量（按 request_type 聚合）。
数据供阶段 7 WebUI 展示；日志同步一份。
"""

import time

from junjun_core.observability import get_logger

logger = get_logger("loop.statistics")

_started_at = time.time()


async def output_statistics() -> None:
    """调度器任务：聚合最近 24h 数据打日志。"""
    from junjun_core.database import Messages, LLMUsage
    from peewee import fn

    since = time.time() - 86400
    try:
        total = Messages.select().where(Messages.time >= since).count()
        replied = Messages.select().where(
            (Messages.time >= since) & (Messages.is_bot == True)).count()  # noqa: E712

        usage = (LLMUsage
                 .select(LLMUsage.request_type,
                         fn.SUM(LLMUsage.prompt_tokens).alias("pt"),
                         fn.SUM(LLMUsage.completion_tokens).alias("ct"),
                         fn.COUNT(LLMUsage.id).alias("n"))
                 .where(LLMUsage.time >= since)
                 .group_by(LLMUsage.request_type))
        usage_str = " | ".join(
            f"{u.request_type or '?'}: {u.n}次 in={int(u.pt or 0)} out={int(u.ct or 0)}"
            for u in usage) or "无"

        uptime_h = (time.time() - _started_at) / 3600
        logger.info(
            f"[统计24h] 收 {total - replied} 条 / 回 {replied} 条 | "
            f"token: {usage_str} | 本次运行 {uptime_h:.1f}h"
        )
    except Exception as e:
        logger.warning(f"统计任务失败（忽略）: {e}")
