"""统一定时任务调度器（interval / cron 两模式）。

合并原 AsyncTask（后台 interval）与 ScheduledTask（cron）两套为一个实现。
每任务独立 try/except，崩溃打 WARN 自动继续，不拖垮其他任务。
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Dict, Optional

from junjun_core.observability import get_logger

logger = get_logger("loop.scheduler")


@dataclass
class ScheduledTask:
    name: str
    callback: Callable[[], Awaitable[None]]
    interval: Optional[float] = None      # 间隔秒（interval 模式）
    cron_hour: Optional[int] = None       # cron 模式：每天 HH:MM
    cron_minute: Optional[int] = None
    enabled: bool = True
    _last_run: float = 0.0
    _last_cron_date: str = ""

    def due(self, now: Optional[float] = None) -> bool:
        if not self.enabled:
            return False
        now = now if now is not None else time.time()
        if self.interval is not None:
            return (now - self._last_run) >= self.interval
        if self.cron_hour is not None:
            dt = datetime.fromtimestamp(now)
            today = dt.strftime("%Y-%m-%d")
            return (dt.hour == self.cron_hour
                    and dt.minute == (self.cron_minute or 0)
                    and self._last_cron_date != today)
        return False

    def mark_run(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        self._last_run = now
        self._last_cron_date = datetime.fromtimestamp(now).strftime("%Y-%m-%d")


class Scheduler:
    TICK = 20.0  # 检查粒度（秒）

    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._runner: Optional[asyncio.Task] = None

    def add(self, task: ScheduledTask) -> None:
        self._tasks[task.name] = task
        logger.info(f"定时任务注册: {task.name} "
                    f"({'every %ss' % task.interval if task.interval else 'daily %02d:%02d' % (task.cron_hour, task.cron_minute or 0)})")

    def start(self) -> None:
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._loop(), name="scheduler")
            logger.info(f"调度器已启动（{len(self._tasks)} 个任务）")

    async def stop(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
            self._runner = None

    async def _loop(self) -> None:
        while True:
            for task in list(self._tasks.values()):
                if task.due():
                    task.mark_run()
                    try:
                        await task.callback()
                    except Exception as e:
                        logger.warning(f"定时任务 {task.name} 异常（继续调度）: {type(e).__name__}: {e}")
            await asyncio.sleep(self.TICK)


scheduler = Scheduler()


def register_default_tasks() -> None:
    """注册阶段 4 默认任务（幂等由 add 覆盖保证）。"""

    async def memory_forget():
        from junjun_memory.long_term import get_long_term_memory
        removed = get_long_term_memory().forget()
        if removed:
            logger.info(f"记忆遗忘任务: 清理 {removed} 条")

    async def flush_pending_summaries():
        """兜底：超时未满批的摘要批次也定期消费。"""
        from junjun_memory.summarizer import get_summarizer, BATCH_MAX_AGE
        s = get_summarizer()
        now = time.time()
        for chat_id, batch in list(s._batches.items()):
            if batch.lines and (now - batch.started_at) > BATCH_MAX_AGE:
                await s.summarize(chat_id)

    scheduler.add(ScheduledTask("memory_forget", memory_forget, interval=6 * 3600))
    scheduler.add(ScheduledTask("flush_summaries", flush_pending_summaries, interval=600))
