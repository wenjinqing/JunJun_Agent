"""会话级串行队列：同会话消息按序处理，处理中堆积的消息合并进上下文。

对齐阶段 3 计划：LLM 处理慢时新消息不排队多次触发决策——
堆积消息全部入记忆，只对最新一条触发一次决策；超时消息丢弃打 WARN。

Timing Gate（enable_timing_gate=true 时，默认关）：消息取出后先等
timing_gate_wait_seconds 聚拢连发，窗口内只评估一次，超时强制继续。
"""

import asyncio
import time
from typing import Dict, Optional

from junjun_core.observability import get_logger

logger = get_logger("funnel.queue")

_STALE_SECONDS = 60.0


def _timing_gate_wait() -> float:
    from junjun_core.config import get_global_config
    chat = get_global_config().raw.get("chat", {})
    if not chat.get("enable_timing_gate", False):
        return 0.0
    return float(chat.get("timing_gate_wait_seconds", 5.0))


class SessionQueue:
    """单会话：一个 worker 协程串行消费。"""

    def __init__(self, chat_id: str, handler):
        self.chat_id = chat_id
        self._handler = handler  # async (session, meta) -> None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name=f"session-{self.chat_id}")

    def put(self, session, meta) -> None:
        self._queue.put_nowait((session, meta, time.time()))
        self.start()

    async def _loop(self) -> None:
        while True:
            try:
                session, meta, ts = await asyncio.wait_for(self._queue.get(), timeout=300)
            except asyncio.TimeoutError:
                break  # 5 分钟无消息，worker 退出（下次 put 重启）
            if time.time() - ts > _STALE_SECONDS:
                logger.warning(f"[{self.chat_id}] 丢弃过期消息（排队 >{_STALE_SECONDS}s）: {meta.text[:40]}")
                self._queue.task_done()
                continue
            wait = _timing_gate_wait()
            if wait > 0:
                # Timing Gate：等连发聚拢，窗口内只评估最新一条；超时强制继续
                await asyncio.sleep(wait)
                drained = 0
                while not self._queue.empty():
                    _, meta, ts2 = self._queue.get_nowait()
                    self._queue.task_done()
                    drained += 1
                    if time.time() - ts2 > _STALE_SECONDS:
                        logger.warning(f"[{self.chat_id}] 丢弃过期消息（聚拢窗口内）: {meta.text[:40]}")
                if drained:
                    logger.debug(f"[{self.chat_id}] timing gate 聚拢 {drained} 条，只评估最新一条")
            try:
                await self._handler(session, meta)
            except Exception as e:
                logger.error(f"[{self.chat_id}] 会话处理异常: {type(e).__name__}: {e}")
            finally:
                self._queue.task_done()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


class SessionQueueManager:
    def __init__(self):
        self._queues: Dict[str, SessionQueue] = {}

    def dispatch(self, session, meta, handler) -> None:
        q = self._queues.get(session.chat_id)
        if q is None:
            q = SessionQueue(session.chat_id, handler)
            self._queues[session.chat_id] = q
        q.put(session, meta)

    async def stop_all(self) -> None:
        for q in self._queues.values():
            await q.stop()
        self._queues.clear()


session_queues = SessionQueueManager()
