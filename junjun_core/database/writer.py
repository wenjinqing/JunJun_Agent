"""单写协程队列：SQLite 并发写防锁。

所有写操作提交为 thunk，由单一后台协程串行执行（executor 线程池跑同步 peewee），
读操作可直接走 model（WAL 模式读写不互斥）。

P1-8 加固（原：无界队列 + executor 无超时 + 协程死亡无看护）：
- 队列有界（2000）：满了丢最新 + 节流告警——下游卡死时内存不再无限积压
- executor 调用 60s 超时：SQLite 锁等待/磁盘卡顿不永久挂住写协程
- submit 时检查写协程存活：意外死亡自动重启（积压条目不丢，队列复用）
"""

import asyncio
import time
from typing import Callable, Optional

from junjun_core.observability import get_logger

logger = get_logger("db.writer")

_QUEUE_MAX = 2000
_EXEC_TIMEOUT = 60.0
_WARN_INTERVAL = 300.0  # 队列满告警节流（秒）


class DBWriter:
    def __init__(self):
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._last_full_warn = 0.0

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._task = asyncio.create_task(self._loop(), name="db-writer")
        logger.info("DB 写队列已启动")

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)  # 哨兵退出
        await self._task
        self._task = None
        logger.info("DB 写队列已停止")

    def submit(self, fn: Callable, *args, **kwargs) -> None:
        """提交写操作（fire-and-forget）。未启动时直接同步执行（测试/脚本场景）。"""
        if self._queue is None:
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"DB 直写失败: {e}")
            return
        # 看门狗：写协程意外死亡则重启（队列复用，积压条目不丢）
        if self._task is not None and self._task.done():
            exc = None
            try:
                exc = self._task.exception()
            except asyncio.CancelledError:
                pass
            logger.error(f"DB 写协程已死亡，重启（原因: {exc}）")
            self._task = None
            self.start()
        try:
            self._queue.put_nowait((fn, args, kwargs))
        except asyncio.QueueFull:
            now = time.monotonic()
            if now - self._last_full_warn >= _WARN_INTERVAL:
                self._last_full_warn = now
                logger.error(f"DB 写队列已满（{_QUEUE_MAX}），丢弃新写入: "
                             f"{getattr(fn, '__qualname__', fn)}（下游疑似卡死，检查磁盘/SQLite）")

    async def _loop(self) -> None:
        import functools
        loop = asyncio.get_running_loop()
        while True:
            item = await self._queue.get()
            if item is None:
                break
            fn, args, kwargs = item
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, functools.partial(fn, *args, **kwargs)),
                    timeout=_EXEC_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"DB 写入超时（{_EXEC_TIMEOUT:.0f}s，跳过不阻塞后续）: "
                               f"{getattr(fn, '__qualname__', fn)}")
            except Exception as e:
                logger.warning(f"DB 写入失败（忽略不阻塞）: {type(e).__name__}: {e}")
            finally:
                self._queue.task_done()


db_writer = DBWriter()
