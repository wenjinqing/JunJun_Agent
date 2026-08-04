"""异步任务管理器：慢工具「提交即返回，完成直发」的通用基建。

适用：成品内容型工具（图片/语音/视频文件）——产出不需要 LLM 再加工。
不适用：信息素材型工具（搜索/天气文本）——必须 LLM 组织语言，保持同步。

用法（工具内）::

    from junjun_agent.tasks import task_manager

    ack = await task_manager.submit(
        kind="ai_draw",
        work=lambda: poll_and_build_segments(task_id),  # async () -> list[ReplySegment] | None
        done_text="画好了！",
        fail_text="这次画失败了，再试一次？",
        timeout=150,
    )
    return ack   # 直接作为工具返回值：接受时是「在弄了」，占线时是「上一个还没完」

路由：复用 memory_skills.current_chat_id（格式 ``qq:ID:group|private``），
工具在 processor 流程内调用时天然携带当前会话，无需新增 contextvar。

可靠性：
- 每会话同 kind 任务去重（占线时返回占线话术，不并行堆任务）
- work 全程 try/except + 硬超时，炸了只发降级文案，绝不影响主流程
- 失败自动重试一次（[tasks] auto_retry，默认开）——ModelScope 类抖动占失败大头
- 结果登记 + 决策注入（2026-08-04「图呢」事件）：任务结局对在途会话可见，
  Agent 记得自己答应的事办成了没有；完成/失败话术同步写进短期记忆
- 进程重启丢任务可接受（内存任务）；shutdown() 优雅取消全部
"""

import asyncio
import random
import time
from collections import deque
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from junjun_core.contracts import ReplySegment, ReplySet
from junjun_core.observability import get_logger

logger = get_logger("agent.tasks")

_DEFAULT_TIMEOUT = 150.0

# 结果登记：chat_id -> deque[{ts, kind, status, detail}]（决策注入与状态查询的数据源）
_OUTCOME_MAX = 10
_OUTCOME_TTL = 1800.0   # 注入窗口 30 分钟——再久的结局对方早忘了，不用提醒

# kind -> 中文名（状态注入/查询用「画图」而不是 ai_draw）
_KIND_CN = {
    "ai_draw": "画图", "tts": "语音", "bilibili": "B站视频",
    "douyin": "抖音视频", "video_watch": "看视频",
}


def _kind_cn(kind: str) -> str:
    return _KIND_CN.get(kind, kind)

# 完成话术模板池（贴合人设，可按 kind 覆盖；空列表表示不发完成语，只发内容段）
# 警告：这些模板直发不经过 echo guard——池子小了必成口头禅
# （bot 说过的每句话都会进记忆被模型学走），每条池保持 6+，定期添新。
_DONE_TEMPLATES: Dict[str, List[str]] = {
    "ai_draw": ["画好了画好了~", "肝完了，看看吧", "当当——画完啦",
                "画好啦，来验收", "完工，看看合不合心意", "画好了，这次手感不错"],
    "tts": [],          # 语音本身即内容，不需要完成语
    "bilibili": [],     # 结果自带标题/信息卡，不发完成语
    "douyin": [],       # 解析结果自带说明
    "_default": ["弄好了~", "弄好啦，看看", "搞定了", "办妥了，来验收",
                 "好了好了，久等", "弄完啦"],
}

# 占线话术模板池（同上：保持 6+）
_BUSY_TEMPLATES = [
    "上一个还在弄呢，等下吧。",
    "手头的还没完，别急。",
    "一次只能弄一个，等这个好了再说。",
    "排队排队，前一个还没好。",
    "忙着呢，这单完了就轮到你。",
    "稍等，手上这个马上完。",
]


def _parse_route(chat_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    """chat_id（qq:ID:group|private）-> (platform, target_user_id, target_group_id)。"""
    parts = (chat_id or "").split(":")
    if len(parts) < 3:
        return parts[0] if parts and parts[0] else "qq", None, None
    platform, target, kind = parts[0], parts[1], parts[2]
    if kind == "group":
        return platform, None, target
    return platform, target, None


class TaskManager:
    """后台任务登记/去重/兜底/直发/结果追踪。"""

    def __init__(self) -> None:
        self._running: Dict[Tuple[str, str], asyncio.Task] = {}
        self._started: Dict[Tuple[str, str], float] = {}   # key -> monotonic 起点
        self._outcomes: Dict[str, deque] = {}

    @staticmethod
    def _current_chat_id() -> str:
        try:
            from junjun_skills.builtin.memory_skills import current_chat_id
            return current_chat_id.get() or ""
        except Exception:
            return ""

    @staticmethod
    def _auto_retry() -> bool:
        """[tasks] auto_retry（默认开）：失败后自动重试一次再认输。"""
        try:
            from junjun_core.config import get_global_config
            return bool(get_global_config().raw.get("tasks", {}).get("auto_retry", True))
        except Exception:
            return True

    def is_busy(self, chat_id: str, kind: str) -> bool:
        task = self._running.get((chat_id, kind))
        return task is not None and not task.done()

    async def submit(
        self,
        *,
        kind: str,
        work: Callable[[], Awaitable[Optional[List[ReplySegment]]]],
        ack_text: str = "在弄了，好了直接发出来。",
        done_text: str = "",
        fail_text: str = "这次失败了，再试一次？",
        busy_text: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
        chat_id: str = "",
        cleanup: "Callable[[], Awaitable[None]] | None" = None,
    ) -> str:
        """登记后台任务，返回应作为工具返回值的话术。

        work: 异步可调用，成功返回内容段列表，失败/超时返回 None 或抛异常。
        cleanup: 发送尝试之后调用的收尾钩子（无论成败）——成品文件（视频等）
                 必须等发送后才删，用这个钩子，不要在 work 的 finally 里删。
        返回值：接受时为 ack_text（由调用方原样 return）；占线时为占线话术。
        """
        chat_id = chat_id or self._current_chat_id()
        if not chat_id:
            return fail_text  # 拿不到会话路由，后台发了也到不了——让工具走同步降级
        if self.is_busy(chat_id, kind):
            logger.info(f"[{chat_id}] {kind} 任务占线，拒绝新提交")
            return busy_text or random.choice(_BUSY_TEMPLATES)

        async def _runner() -> None:
            started = time.monotonic()
            attempts = 2 if self._auto_retry() else 1
            segments = None
            err_detail = ""
            for i in range(attempts):
                try:
                    segments = await asyncio.wait_for(work(), timeout=timeout)
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    err_detail = "超时"
                    segments = None
                except Exception as e:
                    err_detail = type(e).__name__
                    logger.warning(f"[{chat_id}] {kind} 后台任务异常: {type(e).__name__}: {e}")
                    segments = None
                if segments:
                    break
                if i < attempts - 1:
                    logger.info(f"[{chat_id}] {kind} 失败（{err_detail or '无产出'}），自动重试一次")
                    await asyncio.sleep(3.0)
            elapsed = time.monotonic() - started
            try:
                if segments:
                    if done_text:
                        text = done_text
                    else:
                        pool = _DONE_TEMPLATES.get(kind, _DONE_TEMPLATES["_default"])
                        text = random.choice(pool) if pool else ""
                    out = ([ReplySegment(type="text", data=text)] if text else []) + segments
                    sent = await self._send(chat_id, out)
                    logger.info(f"[{chat_id}] {kind} 后台任务完成，已直发（{elapsed:.1f}s）")
                    self._record_outcome(chat_id, kind, "done",
                                         f"耗时{elapsed:.0f}s" + ("" if sent else "，但发送失败"),
                                         said=text)
                else:
                    sent = False
                    if fail_text:
                        sent = await self._send(chat_id, [ReplySegment(type="text", data=fail_text)])
                    logger.info(f"[{chat_id}] {kind} 后台任务失败，已发降级文案（{elapsed:.1f}s）")
                    self._record_outcome(chat_id, kind, "failed",
                                         (err_detail or "无产出") + ("" if sent else "，降级文案也没发出去"),
                                         said=fail_text if sent else "")
            finally:
                # 成品文件等「发送尝试之后」才清理——提前删会让 NapCat 拿到不存在的路径
                if cleanup is not None:
                    try:
                        await cleanup()
                    except Exception as e:
                        logger.warning(f"[{chat_id}] {kind} 任务收尾清理异常: {e}")

        task = asyncio.create_task(_runner(), name=f"bg-{kind}-{chat_id}")
        key = (chat_id, kind)
        self._running[key] = task
        self._started[key] = time.monotonic()

        def _pop(_t, k=key):
            self._running.pop(k, None)
            self._started.pop(k, None)
        task.add_done_callback(_pop)
        logger.info(f"[{chat_id}] {kind} 后台任务已登记")
        return ack_text

    def _record_outcome(self, chat_id: str, kind: str, status: str,
                        detail: str, said: str = "") -> None:
        """登记结局：① 决策注入数据源 ② 话术写进短期记忆——
        直发消息不经过 inbound 管线，不手动记的话模型会忘了自己说过
        「画好啦/画砸了」，下轮被问「图呢」只能装傻（2026-08-04 实战）。"""
        dq = self._outcomes.setdefault(chat_id, deque(maxlen=_OUTCOME_MAX))
        dq.append({"ts": time.time(), "kind": kind, "status": status, "detail": detail})
        if said:
            try:
                from junjun_core.gateway.session_manager import get_session_manager
                s = get_session_manager().all_sessions().get(chat_id)
                if s is not None and getattr(s, "memory", None) is not None:
                    note = said if status == "done" else f"（后台任务{ _kind_cn(kind) }失败：{detail}）{said}"
                    s.memory.add_bot(note)
            except Exception:
                pass

    def _status_lines(self, chat_id: str) -> List[str]:
        """在途 + 近 30 分钟结局的人类可读行（注入与查询共用）。"""
        lines = []
        now = time.time()
        for (cid, kind), t in list(self._running.items()):
            if cid != chat_id or t.done():
                continue
            started = self._started.get((cid, kind))
            mins = int((time.monotonic() - started) / 60) if started else 0
            lines.append(f"- {_kind_cn(kind)}：进行中（已 {mins} 分钟）")
        for o in reversed(self._outcomes.get(chat_id, ())):
            if now - o["ts"] > _OUTCOME_TTL:
                continue
            when = time.strftime("%H:%M", time.localtime(o["ts"]))
            status = "完成" if o["status"] == "done" else "失败"
            lines.append(f"- {_kind_cn(o['kind'])}：{status}（{when}，{o['detail']}）")
        return lines

    def task_status_block(self, chat_id: str) -> str:
        """决策注入块：让 Agent 记得自己答应过的事办得怎么样。"""
        lines = self._status_lines(chat_id)
        if not lines:
            return ""
        return ("【你的后台任务近况】（你答应过的事，对方问起照实说；"
                "失败了主动提补救，别装没这回事）\n" + "\n".join(lines))

    def list_for_chat(self, chat_id: str) -> str:
        """list_background_tasks 工具合并用；无任务返回空串。"""
        return "\n".join(self._status_lines(chat_id))

    async def _send(self, chat_id: str, segments: List[ReplySegment]) -> bool:
        """直发到会话（gateway 不可用时静默——测试环境允许）。返回是否送达。"""
        try:
            from junjun_core.gateway import router as router_mod
            gateway = router_mod.get_gateway()
            platform, user_id, group_id = _parse_route(chat_id)
            await gateway.send_reply(ReplySet(
                platform=platform,
                target_user_id=user_id,
                target_group_id=group_id,
                segments=segments,
                should_reply=True,
            ))
            return True
        except Exception as e:
            logger.warning(f"[{chat_id}] 后台任务发送失败: {type(e).__name__}: {e}")
            return False

    async def shutdown(self) -> None:
        """优雅退出：取消全部未完成任务。"""
        pending = [t for t in self._running.values() if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info(f"后台任务已全部取消（{len(pending)} 个）")
        self._running.clear()
        self._started.clear()


task_manager = TaskManager()
