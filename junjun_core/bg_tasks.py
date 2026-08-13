"""fire-and-forget 后台任务的强引用留存 + 异常必响。

`asyncio.create_task` 创建的任务只被事件循环弱引用：调用方不存强引用，
任务可能在任意 await 点被 GC 收走——CancelledError 是 BaseException 不进
`except Exception`，任务死得无声无息（2026-08-13 生产实锤：TaskKernel
后台规划任务跑了 8 分钟后人间蒸发，无日志、无汇报、用户干等）。

统一入口 fire_and_forget：留强引用直到完成 + done 回调把取消/异常打进
日志——「静默失败」这一类食物链一次端掉。凡是需要「发了就不管」的后台
任务都走这里，不要裸调 create_task。
"""

import asyncio
from typing import Coroutine

from junjun_core.observability import get_logger

logger = get_logger("bg_tasks")

_BG_TASKS: "set[asyncio.Task]" = set()


def fire_and_forget(coro: Coroutine, *, name: str) -> asyncio.Task:
    """起后台任务并持有强引用直到完成；异常/取消必落日志。"""
    task = asyncio.create_task(coro, name=name)
    _BG_TASKS.add(task)

    def _done(t: asyncio.Task) -> None:
        _BG_TASKS.discard(t)
        if t.cancelled():
            logger.warning(f"后台任务被取消（GC 收走或主动取消）: {t.get_name()}")
            return
        exc = t.exception()
        if exc is not None:
            logger.error(f"后台任务异常逃逸: {t.get_name()}: "
                         f"{type(exc).__name__}: {exc}")

    task.add_done_callback(_done)
    return task


def pending_count() -> int:
    """在途任务数（测试/监控用）。"""
    return len(_BG_TASKS)
