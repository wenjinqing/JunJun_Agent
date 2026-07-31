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
- 进程重启丢任务可接受（内存任务）；shutdown() 优雅取消全部
"""

import asyncio
import random
import time
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from junjun_core.contracts import ReplySegment, ReplySet
from junjun_core.observability import get_logger

logger = get_logger("agent.tasks")

_DEFAULT_TIMEOUT = 150.0

# 完成话术模板池（贴合人设，可按 kind 覆盖；None 表示不发完成语，只发内容段）
_DONE_TEMPLATES: Dict[str, List[str]] = {
    "ai_draw": ["画好了画好了~", "肝完了，看看吧", "当当——画完啦"],
    "tts": [],          # 语音本身即内容，不需要完成语
    "bilibili": [],     # 结果自带标题/信息卡，不发完成语
    "douyin": [],       # 解析结果自带说明
    "_default": ["弄好了~"],
}

# 占线话术模板池
_BUSY_TEMPLATES = [
    "上一个还在弄呢，等下吧。",
    "手头的还没完，别急。",
    "一次只能弄一个，等这个好了再说。",
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
    """后台任务登记/去重/兜底/直发。"""

    def __init__(self) -> None:
        self._running: Dict[Tuple[str, str], asyncio.Task] = {}

    @staticmethod
    def _current_chat_id() -> str:
        try:
            from junjun_skills.builtin.memory_skills import current_chat_id
            return current_chat_id.get() or ""
        except Exception:
            return ""

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
            try:
                segments = await asyncio.wait_for(work(), timeout=timeout)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[{chat_id}] {kind} 后台任务异常: {type(e).__name__}: {e}")
                segments = None
            elapsed = time.monotonic() - started
            try:
                if segments:
                    if done_text:
                        text = done_text
                    else:
                        pool = _DONE_TEMPLATES.get(kind, _DONE_TEMPLATES["_default"])
                        text = random.choice(pool) if pool else ""
                    out = ([ReplySegment(type="text", data=text)] if text else []) + segments
                    await self._send(chat_id, out)
                    logger.info(f"[{chat_id}] {kind} 后台任务完成，已直发（{elapsed:.1f}s）")
                else:
                    if fail_text:
                        await self._send(chat_id, [ReplySegment(type="text", data=fail_text)])
                    logger.info(f"[{chat_id}] {kind} 后台任务失败，已发降级文案（{elapsed:.1f}s）")
            finally:
                # 成品文件等「发送尝试之后」才清理——提前删会让 NapCat 拿到不存在的路径
                if cleanup is not None:
                    try:
                        await cleanup()
                    except Exception as e:
                        logger.warning(f"[{chat_id}] {kind} 任务收尾清理异常: {e}")

        task = asyncio.create_task(_runner(), name=f"bg-{kind}-{chat_id}")
        self._running[(chat_id, kind)] = task
        task.add_done_callback(lambda _t: self._running.pop((chat_id, kind), None))
        logger.info(f"[{chat_id}] {kind} 后台任务已登记")
        return ack_text

    async def _send(self, chat_id: str, segments: List[ReplySegment]) -> None:
        """直发到会话（gateway 不可用时静默——测试环境允许）。"""
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
        except Exception as e:
            logger.warning(f"[{chat_id}] 后台任务发送失败: {type(e).__name__}: {e}")

    async def shutdown(self) -> None:
        """优雅退出：取消全部未完成任务。"""
        pending = [t for t in self._running.values() if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info(f"后台任务已全部取消（{len(pending)} 个）")
        self._running.clear()


task_manager = TaskManager()
