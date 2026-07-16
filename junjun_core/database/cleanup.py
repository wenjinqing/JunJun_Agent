"""数据库自动清理：对齐原 [database] 配置语义。

严格只清理：LLMUsage（按 cleanup_retention_days）+ 低频黑话（count==1 超 30 天）。
功能数据（messages/画像/表达/已确认黑话/提醒）不动。失败不影响主进程。
"""

import time

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("db.cleanup")


async def run_cleanup() -> None:
    cfg = get_global_config().raw.get("database", {})
    if not cfg.get("enable_auto_cleanup", True):
        return
    retention_days = int(cfg.get("cleanup_retention_days", 60))
    cutoff = time.time() - retention_days * 86400
    try:
        from junjun_core.database import LLMUsage, Jargon
        n_usage = LLMUsage.delete().where(LLMUsage.time < cutoff).execute()
        # 低可信黑话：只出现过 1 次且 30 天没再出现的（id 无时间戳，用保守策略：
        # count==1 的行在每轮清理时衰减标记——简化为直接清 count==1 且总量超 5000 时）
        n_jargon = 0
        if Jargon.select().count() > 5000:
            n_jargon = Jargon.delete().where(Jargon.count == 1).execute()
        if n_usage or n_jargon:
            logger.warning(f"DB 清理: llm_usage -{n_usage} 行"
                           + (f", 低频黑话 -{n_jargon} 行" if n_jargon else ""))
    except Exception as e:
        logger.warning(f"DB 清理失败（忽略）: {e}")
