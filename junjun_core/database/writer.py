"""单写协程队列：SQLite 并发写防锁。

所有写操作提交为 thunk，由单一后台协程串行执行（executor 线程池跑同步 peewee），
读操作可直接走 model（WAL 模式读写不互斥）。
"""

import asyncio
from typing import Callable, Optional

from junjun_core.observability import get_logger

logger = get_logger("db.writer")


class DBWriter:
    def __init__(self):
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._queue = asyncio.Queue()
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
        self._queue.put_nowait((fn, args, kwargs))

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await self._queue.get()
            if item is None:
                break
            fn, args, kwargs = item
            try:
                await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
            except Exception as e:
                logger.warning(f"DB 写入失败（忽略不阻塞）: {type(e).__name__}: {e}")
            finally:
                self._queue.task_done()


db_writer = DBWriter()
