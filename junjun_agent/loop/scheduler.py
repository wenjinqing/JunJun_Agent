"""统一定时任务调度器（interval / cron 两模式）。

合并原 AsyncTask（后台 interval）与 ScheduledTask（cron）两套为一个实现。
每任务独立 try/except，崩溃打 WARN 自动继续，不拖垮其他任务。

2026-07-31 P2 加固：
- 到期任务 create_task 并发执行（原串行 await——topic_finder 一次 LLM 生成
  30s 会把 reminders 等全部任务延后）；同名任务重入锁防自身并发
- cron 模式加 10 分钟迟到容忍（原要求 hour==且minute==，机器睡眠/调度
  繁忙错过分钟窗口当天就丢了）
- plugin 字段：插件被禁用时其后台任务跳过（原禁用只管工具/命令/拦截器）
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Dict, Optional

from junjun_core.observability import get_logger

logger = get_logger("loop.scheduler")

_CRON_LATE_TOLERANCE = 600.0  # cron 迟到容忍窗口（秒）


@dataclass
class ScheduledTask:
    name: str
    callback: Callable[[], Awaitable[None]]
    interval: Optional[float] = None      # 间隔秒（interval 模式）
    cron_hour: Optional[int] = None       # cron 模式：每天 HH:MM
    cron_minute: Optional[int] = None
    enabled: bool = True
    plugin: str = ""                      # 所属插件（插件禁用时任务跳过；空=内置）
    _last_run: float = 0.0
    _last_cron_date: str = ""
    _running: bool = False                # 重入锁：上一次还没跑完不再触发

    def due(self, now: Optional[float] = None) -> bool:
        if not self.enabled or self._running:
            return False
        now = now if now is not None else time.time()
        if self.interval is not None:
            return (now - self._last_run) >= self.interval
        if self.cron_hour is not None:
            dt = datetime.fromtimestamp(now)
            today = dt.strftime("%Y-%m-%d")
            if self._last_cron_date == today:
                return False
            scheduled = dt.replace(hour=self.cron_hour,
                                   minute=(self.cron_minute or 0),
                                   second=0, microsecond=0)
            lag = now - scheduled.timestamp()
            return 0 <= lag <= _CRON_LATE_TOLERANCE
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
            # 启动错峰：interval 任务首轮按 20s 间隔错开（_last_run=0 会让
            # 全部任务在启动瞬间同时到期，LLM 调用洪峰直接打满端点限流——
            # 2026-07-29 启动卡死排查：/chat/completions 被十几个首轮任务打爆）
            now = time.time()
            interval_tasks = [t for t in self._tasks.values() if t.interval is not None]
            for i, t in enumerate(interval_tasks):
                stagger = min(i * 20.0, t.interval * 0.5)
                t._last_run = now - t.interval + stagger
            self._runner = asyncio.create_task(self._loop(), name="scheduler")
            logger.info(f"调度器已启动（{len(self._tasks)} 个任务，首轮错峰 20s/个）")

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
                    # 并发执行：慢任务（LLM 生成/VLM 注册）不阻塞 reminders 等其他任务
                    asyncio.create_task(self._run_one(task),
                                        name=f"sched-{task.name}")
            await asyncio.sleep(self.TICK)

    async def _run_one(self, task: ScheduledTask) -> None:
        task._running = True
        try:
            if task.plugin:
                # 插件被禁用 -> 后台任务同样静默（禁用语义覆盖全生命周期）
                from junjun_skills import registry
                if not registry.is_plugin_enabled(task.plugin):
                    return
            await task.callback()
        except Exception as e:
            logger.warning(f"定时任务 {task.name} 异常（继续调度）: {type(e).__name__}: {e}")
        finally:
            task._running = False


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

    async def reminders():
        from junjun_agent.loop.reminder import check_due_reminders
        await check_due_reminders()

    async def proactive_scan():
        from junjun_agent.loop.proactive import proactive_manager
        await proactive_manager.scan()

    async def emoji_register():
        from junjun_express.emoji import emoji_manager
        await emoji_manager.register_pending()

    async def statistics():
        from junjun_agent.loop.statistics import output_statistics
        await output_statistics()

    async def online_time():
        from junjun_agent.loop.statistics import record_online_time
        await record_online_time()

    async def db_cleanup():
        from junjun_core.database.cleanup import run_cleanup
        await run_cleanup()

    async def expression_reflect():
        from junjun_express.reflector import expression_reflector
        await expression_reflector.check_and_ask()

    from junjun_core.config import get_global_config
    raw = get_global_config().raw
    interval = int(raw.get("reminder", {}).get("check_interval_seconds", 60))
    proactive_min = int(raw.get("proactive_chat", {}).get("check_interval_minutes", 30))
    emoji_min = int(raw.get("emoji", {}).get("check_interval", 10))
    cleanup_h = int(raw.get("database", {}).get("cleanup_interval_hours", 24))

    # P1-8 会话淘汰回调：core 不 import 上层，用钩子释放 agent 资源 + 队列条目
    from junjun_core.gateway.session_manager import get_session_manager
    from junjun_agent.funnel.session_queue import session_queues
    get_session_manager().on_evict = lambda s: session_queues.drop(s.chat_id)

    scheduler.add(ScheduledTask("memory_forget", memory_forget, interval=6 * 3600))
    scheduler.add(ScheduledTask("flush_summaries", flush_pending_summaries, interval=600))
    scheduler.add(ScheduledTask("reminders", reminders, interval=interval))
    scheduler.add(ScheduledTask("proactive_chat", proactive_scan, interval=proactive_min * 60))
    scheduler.add(ScheduledTask("emoji_register", emoji_register, interval=emoji_min * 60))
    scheduler.add(ScheduledTask("statistics", statistics, interval=4 * 3600))
    scheduler.add(ScheduledTask("online_time", online_time, interval=60))
    scheduler.add(ScheduledTask("db_cleanup", db_cleanup, interval=cleanup_h * 3600))
    scheduler.add(ScheduledTask("expression_reflect", expression_reflect, interval=5 * 60))
