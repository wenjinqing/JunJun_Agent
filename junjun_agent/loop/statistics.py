"""统计任务：对齐原 chat/statistics 的 StatisticOutputTask + OnlineTimeRecordTask 语义。

StatisticOutputTask：定期输出运行统计（消息量/回复量/token 用量），数据供 WebUI。
OnlineTimeRecordTask：每分钟续 OnlineTime.end_timestamp，记录 bot 在线时段。
"""

import time

from junjun_core.observability import get_logger

logger = get_logger("loop.statistics")

_started_at = time.time()
_online_record_id = None


async def record_online_time() -> None:
    """调度器任务（60s）：对齐原 OnlineTimeRecordTask——有记录则续期，无则新建。"""
    global _online_record_id
    from junjun_core.database import OnlineTime
    now = time.time()
    try:
        if _online_record_id is not None:
            n = (OnlineTime.update(end_timestamp=now + 60)
                 .where(OnlineTime.id == _online_record_id).execute())
            if n:
                return  # 续期成功
        # 无记录或记录丢失：找最近一分钟内的记录续上，否则新开一段
        row = (OnlineTime.select().where(OnlineTime.end_timestamp >= now - 60)
               .order_by(OnlineTime.end_timestamp.desc()).first())
        if row is not None:
            _online_record_id = row.id
            OnlineTime.update(end_timestamp=now + 60).where(OnlineTime.id == row.id).execute()
        else:
            _online_record_id = OnlineTime.create(
                start_timestamp=now, end_timestamp=now + 60).id
    except Exception as e:
        logger.warning(f"在线时长记录失败（忽略）: {e}")


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
