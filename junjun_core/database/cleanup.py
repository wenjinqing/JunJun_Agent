"""数据库自动清理：对齐原 [database] 配置语义。

清理范围：LLMUsage（按 cleanup_retention_days）+ 低频黑话（count==1 超 30 天）
+ Messages（按 messages_retention_days，默认 365 天保守界，0=不清——2026-08-13
审查 P1：表必须有界，但聊天记录是记忆原料，窗口给足一年）。
其余功能数据（画像/表达/已确认黑话/提醒）不动。失败不影响主进程。
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
    msg_days = int(cfg.get("messages_retention_days", 365))
    msg_cutoff = time.time() - msg_days * 86400 if msg_days > 0 else 0.0
    # 走 db_writer 单写队列：原实现在事件循环里同步 delete（阻塞循环），
    # 且绕过单写约定与 writer 并发写 SQLite（偶发 database is locked 被静默吞掉）
    from junjun_core.database import db_writer
    db_writer.submit(_do_cleanup, cutoff, msg_cutoff)


def _do_cleanup(cutoff: float, msg_cutoff: float = 0.0) -> None:
    """实际清理（db_writer executor 线程内同步执行）。"""
    try:
        from junjun_core.database import LLMUsage, Jargon, Messages, ToolUsage
        n_usage = LLMUsage.delete().where(LLMUsage.time < cutoff).execute()
        # 工具统计与 token 用量同窗（同属遥测数据）
        n_usage += ToolUsage.delete().where(ToolUsage.time < cutoff).execute()
        # 低可信黑话：只出现过 1 次且 30 天没再出现的（id 无时间戳，用保守策略：
        # count==1 的行在每轮清理时衰减标记——简化为直接清 count==1 且总量超 5000 时）
        n_jargon = 0
        if Jargon.select().count() > 5000:
            n_jargon = Jargon.delete().where(Jargon.count == 1).execute()
        n_msg = 0
        if msg_cutoff:
            n_msg = Messages.delete().where(Messages.time < msg_cutoff).execute()
        if n_usage or n_jargon or n_msg:
            logger.warning(f"DB 清理: llm_usage -{n_usage} 行"
                           + (f", 低频黑话 -{n_jargon} 行" if n_jargon else "")
                           + (f", 老消息 -{n_msg} 行" if n_msg else ""))
    except Exception as e:
        logger.warning(f"DB 清理失败（忽略）: {e}")
