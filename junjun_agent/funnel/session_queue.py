"""会话级串行队列：同会话消息按序处理，处理中堆积的消息合并进上下文。

对齐阶段 3 计划：LLM 处理慢时新消息不排队多次触发决策——
堆积消息全部入记忆，只对最新一条触发一次决策；超时消息丢弃打 WARN。

决策目标选择（Q1，2026-08-09）：合并后不再无条件取最新一条——
batch 里有 addressed（@/直呼）消息时，决策目标回拨到最新 addressed。
此前「取最新」让 @bot 的提问被后到的闲聊顶替、决策门判沉默，
@必回 契约被破坏（温衿青事故：@你 提问被「老婆老婆」顶掉零回复）。
全不 addressed 维持取最新（交给决策门判沉默），私聊行为不变。

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

    def __init__(self, chat_id: str, handler, pre_handler=None, addressed_fn=None):
        self.chat_id = chat_id
        self._handler = handler  # async (session, meta) -> None
        # pre_handler: 决策前段（命令/拦截器/预热等 0 token 副作用）。
        # 被合并的消息也必须过这一段——否则「/sub add xxx」+「你在吗」连发时
        # 斜杠命令被静默吞掉零日志（严厉审查 P1-7）
        self._pre_handler = pre_handler
        # addressed_fn(session, meta) -> bool：合并批次里选决策目标用
        # （Q1：@bot 消息不能被后进闲聊顶替）。None = 旧行为（取最新）。
        self._addressed_fn = addressed_fn
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

            # 合并：把队列里剩余消息全部 drain 进上下文（预期语义：连发消息
            # 合并一次回复，不是逐条触发）。决策目标由 _select_target 选——
            # batch 里有 addressed 消息时回拨到最新 addressed（Q1）。
            drained = []
            while not self._queue.empty():
                try:
                    _, m2, ts2 = self._queue.get_nowait()
                    self._queue.task_done()
                    if time.time() - ts2 > _STALE_SECONDS:
                        logger.warning(f"[{self.chat_id}] 丢弃过期消息（合并窗口内）: {m2.text[:40]}")
                        continue
                    drained.append(m2)
                except asyncio.QueueEmpty:
                    break

            wait = _timing_gate_wait()
            if wait > 0:
                await asyncio.sleep(wait)
                # timing gate 窗口内再 drain 一次，并入同一批次统一选目标
                while not self._queue.empty():
                    try:
                        _, m2, ts2 = self._queue.get_nowait()
                        self._queue.task_done()
                        if time.time() - ts2 <= _STALE_SECONDS:
                            drained.append(m2)
                    except asyncio.QueueEmpty:
                        break

            if drained:
                batch = [meta] + drained
                meta = self._select_target(session, batch)
                # 除决策目标外的所有消息各过一遍决策前段：
                # 命令/拦截器/预热不该因合并且丢（P1-7）；目标的前段在
                # _handler 内部执行，这里不能重复跑
                for m in batch:
                    if m is not meta:
                        await self._run_pre(session, m)
                logger.debug(f"[{self.chat_id}] 合并 {len(drained)} 条连发消息，"
                             f"决策目标: {meta.text[:30]}")

            try:
                await self._handler(session, meta)
            except Exception as e:
                logger.error(f"[{self.chat_id}] 会话处理异常: {type(e).__name__}: {e}")
            finally:
                self._queue.task_done()

    def _select_target(self, session, batch):
        """决策目标：batch（首条+drain，时间升序）里最新 addressed 消息。

        无 addressed_fn 或全不 addressed -> 取最新（旧语义，交给决策门判沉默）。
        addressed_fn 判定异常 -> 回退取最新（保守方向：退化成现状，不丢决策）。
        """
        target = batch[-1]
        if self._addressed_fn is None:
            return target
        for m in reversed(batch):
            try:
                addressed = self._addressed_fn(session, m)
            except Exception:
                return target
            if addressed:
                if m is not target:
                    logger.info(f"[{self.chat_id}] 决策目标回拨到被 @ 的消息: {m.text[:30]}")
                return m
        return target

    async def _run_pre(self, session, meta) -> None:
        """对被合并的消息跑决策前段（无 pre_handler 时跳过，保持旧行为）。"""
        if self._pre_handler is None:
            return
        try:
            await self._pre_handler(session, meta)
        except Exception as e:
            logger.error(f"[{self.chat_id}] 合并消息前段处理异常: {type(e).__name__}: {e}")

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

    def dispatch(self, session, meta, handler, pre_handler=None, addressed_fn=None) -> None:
        q = self._queues.get(session.chat_id)
        # worker 已退出（5 分钟空闲超时）且队列空：旧条目回收重建，防只增不减
        if q is not None and q._task is not None and q._task.done() and q._queue.empty():
            q = None
        if q is None:
            q = SessionQueue(session.chat_id, handler, pre_handler=pre_handler,
                             addressed_fn=addressed_fn)
            self._queues[session.chat_id] = q
        else:
            if pre_handler is not None and q._pre_handler is None:
                q._pre_handler = pre_handler
            if addressed_fn is not None and q._addressed_fn is None:
                q._addressed_fn = addressed_fn
        q.put(session, meta)

    def drop(self, chat_id: str) -> None:
        """会话淘汰时清队列条目（有消息在飞的会话不会被淘汰——last_active 新）。"""
        q = self._queues.get(chat_id)
        if q is not None and (q._task is None or q._task.done()) and q._queue.empty():
            self._queues.pop(chat_id, None)

    async def stop_all(self) -> None:
        for q in self._queues.values():
            await q.stop()
        self._queues.clear()


session_queues = SessionQueueManager()
