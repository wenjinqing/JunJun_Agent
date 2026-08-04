"""异步任务队列：接单 -> 后台跑 -> 主动汇报（持久化，重启可恢复）。

与 agent/tasks.py（TaskManager）的分工：
- TaskManager：内存型「慢工具直发」——画图/语音等成品文件，重启丢了无所谓
- 本模块：DB 持久化任务队列——跨重启恢复、可列表/可取消、有并发上限与
  超时重试，handler 按 kind 注册（首个实现是 async_task 插件的 agent_task：
  隔离上下文的一次性子 agent，没有人格不碰用户，跑完即焚——「agent 型
  job」的口子，深度研究后续直接挂这里）

执行模型：
- submit 落表（pending）并立即 kick；调度 sweep 兜底（重启恢复 pending /
  处理卡在 running 的崩溃残留）
- pending->running 用条件 UPDATE 原子认领，immediate kick 与 sweep 天然互斥
- 完成/失败走 gateway 主动推送：utils 模型写一句人设开场白 + 原始结果正文
  （正文不过 LLM——调研报告保真优先，改写会丢信息）
"""

import asyncio
import json
import time
import uuid
from typing import Awaitable, Callable, Dict, Optional, Tuple

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("loop.async_jobs")

Handler = Callable[..., Awaitable[str]]  # async (job, payload: dict) -> 结果正文

_HANDLERS: Dict[str, Handler] = {}
_running: Dict[str, asyncio.Task] = {}          # job_id -> task（取消/扫残留用）
_sem: Tuple[Optional[asyncio.AbstractEventLoop], Optional[asyncio.Semaphore]] = (None, None)

_SWEEP_LIMIT = 10        # 每轮 sweep 最多补捞几个 pending
_STUCK_FACTOR = 2        # running 超过 timeout*N 视为崩溃残留
_MAX_ATTEMPTS = 2        # 崩溃残留最多重试几次
_RETENTION_DAYS = 7      # 终态任务保留天数

_STATUS_CN = {"pending": "排队中", "running": "执行中", "done": "已完成",
              "failed": "失败", "cancelled": "已取消"}
_FINAL = ("done", "failed", "cancelled")


def _cfg() -> dict:
    """读取 [async_task] 配置节（热改生效）。"""
    try:
        return get_global_config().raw.get("async_task", {}) or {}
    except Exception:
        return {}


def register_handler(kind: str, fn: Handler) -> None:
    """注册任务执行器。重名报错（拒绝静默覆盖）。"""
    if kind in _HANDLERS:
        raise ValueError(f"job handler 重名: {kind}")
    _HANDLERS[kind] = fn
    logger.info(f"job handler 注册: {kind}")


def _get_sem() -> asyncio.Semaphore:
    """并发上限信号量（绑定当前 loop；loop 变了重建——测试每用例一个 loop）。"""
    global _sem
    loop = asyncio.get_running_loop()
    if _sem[0] is not loop:
        _sem = (loop, asyncio.Semaphore(int(_cfg().get("max_concurrent", 2))))
    return _sem[1]


def submit_job(kind: str, title: str, payload: dict, chat_id: str,
               user_id: str = "", nickname: str = "", *,
               kick: bool = True):
    """建任务。返回 (AsyncJob, "") 或 (None, 错误话术)。"""
    from junjun_core.database.models import AsyncJob
    if not bool(_cfg().get("enable", True)):
        return None, "后台任务功能现在关着，这次只能我直接做了。"
    if kind not in _HANDLERS:
        return None, f"没有「{kind}」这类任务的执行器。"
    if not chat_id:
        return None, "拿不到当前会话，任务完成了也没法汇报，先不接了。"
    cap = int(_cfg().get("max_pending_per_chat", 5))
    active = (AsyncJob.select()
              .where((AsyncJob.chat_id == chat_id)
                     & (AsyncJob.status.in_(("pending", "running"))))
              .count())
    if active >= cap:
        return None, f"手头排队+进行中的任务已经 {active} 个了（上限 {cap}），做完几个再派新的。"
    job = AsyncJob.create(
        job_id=uuid.uuid4().hex[:10], kind=kind, title=(title or "")[:80],
        payload=json.dumps(payload or {}, ensure_ascii=False),
        chat_id=chat_id, user_id=user_id, user_nickname=nickname)
    logger.info(f"[{chat_id}] 任务已建 #{job.job_id} [{kind}] {job.title[:40]}")
    if kick:
        _kick(job.job_id)
    return job, ""


def _kick(job_id: str) -> None:
    """立即起协程执行（拿不到 running loop 就留给 sweep，静默）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if job_id in _running:
        return
    _running[job_id] = loop.create_task(_run(job_id), name=f"job-{job_id}")


async def _run(job_id: str) -> None:
    """认领并执行一个任务。任何路径都不抛给调用方（CancelledError 除外）。"""
    from junjun_core.database.models import AsyncJob
    try:
        async with _get_sem():
            now = time.time()
            claimed = (AsyncJob.update(status="running", started_at=now,
                                       attempts=AsyncJob.attempts + 1)
                       .where((AsyncJob.job_id == job_id)
                              & (AsyncJob.status == "pending"))
                       .execute())
            if not claimed:
                return  # 被并发 kick/sweep 抢先认领，或已被取消
            job = AsyncJob.get_or_none(AsyncJob.job_id == job_id)
            if job is None:
                return
            handler = _HANDLERS.get(job.kind)
            timeout = float(_cfg().get("job_timeout_seconds", 600))
            try:
                if handler is None:
                    raise RuntimeError(f"执行器未注册: {job.kind}")
                payload = json.loads(job.payload or "{}")
                result = await asyncio.wait_for(handler(job, payload), timeout=timeout)
                job.status = "done"
                job.result = str(result)[:int(_cfg().get("result_max_chars", 3000))]
                logger.info(f"[{job.chat_id}] 任务完成 #{job_id}（{len(job.result)} 字）")
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.finished_at = time.time()
                try:
                    job.save()
                except Exception:
                    pass
                raise
            except asyncio.TimeoutError:
                job.status = "failed"
                job.error = f"超时（>{int(timeout)}s）"
                logger.warning(f"[{job.chat_id}] 任务超时 #{job_id}")
            except Exception as e:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"[:200]
                logger.warning(f"[{job.chat_id}] 任务失败 #{job_id}: {job.error}")
            job.finished_at = time.time()
            try:
                job.save()
            except Exception as e:
                logger.warning(f"任务状态落库失败 #{job_id}: {e}")
            await _notify(job)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"任务执行框架异常 #{job_id}: {type(e).__name__}: {e}")
    finally:
        _running.pop(job_id, None)


async def sweep_jobs() -> None:
    """调度兜底：崩溃残留回炉/判死、补捞 pending、清理老尸体。"""
    if not bool(_cfg().get("enable", True)):
        return
    from junjun_core.database.models import AsyncJob
    now = time.time()
    timeout = float(_cfg().get("job_timeout_seconds", 600))

    # 1) 卡在 running 太久的：本进程还在跑的不动；残留回炉（重试超限判死）
    stuck = list(AsyncJob.select().where(
        (AsyncJob.status == "running") & (AsyncJob.started_at < now - timeout * _STUCK_FACTOR)))
    for job in stuck:
        if job.job_id in _running:
            continue
        if job.attempts >= _MAX_ATTEMPTS:
            job.status = "failed"
            job.error = "执行中断（进程重启）且重试超限"
            job.finished_at = now
            job.save()
            logger.warning(f"任务 #{job.job_id} 残留重试超限，判失败")
            await _notify(job)
        else:
            job.status = "pending"
            job.save()
            logger.info(f"任务 #{job.job_id} 崩溃残留，回炉重试（第 {job.attempts + 1} 次）")

    # 2) 补捞 pending（重启恢复 / 并发满时被跳过的）
    pending = list(AsyncJob.select().where(AsyncJob.status == "pending")
                   .order_by(AsyncJob.created_at).limit(_SWEEP_LIMIT))
    for job in pending:
        _kick(job.job_id)

    # 3) 老尸体清理（表必须有界）
    cutoff = now - _RETENTION_DAYS * 86400
    removed = (AsyncJob.delete()
               .where((AsyncJob.status.in_(_FINAL)) & (AsyncJob.finished_at < cutoff))
               .execute())
    if removed:
        logger.info(f"任务尸体清理: {removed} 条")


def cancel_job(job_id: str, caller: str) -> str:
    """取消任务（排队中直接置 cancelled；执行中 cancel 协程）。本人或管理员。"""
    from junjun_core.database.models import AsyncJob
    from junjun_core.security import is_admin
    jid = (job_id or "").strip().lstrip("#")
    job = AsyncJob.get_or_none(AsyncJob.job_id == jid)
    if job is None or job.status in _FINAL:
        return f"没找到进行中/排队的任务 #{jid}。"
    if caller != job.user_id and not is_admin(caller):
        return f"#{jid} 是 {job.user_nickname or job.user_id} 派的，只有本人或管理员能取消。"
    if job.status == "pending":
        job.status = "cancelled"
        job.finished_at = time.time()
        job.save()
    task = _running.get(job.job_id)
    if task is not None and not task.done():
        task.cancel()  # _run 的 CancelledError 分支兜底置 cancelled（running 情形）
    elif job.status == "running":
        # 不在本进程（异常残留），直接置
        job.status = "cancelled"
        job.finished_at = time.time()
        job.save()
    logger.info(f"任务已取消 #{jid}")
    return f"已取消任务 #{jid}（{job.title}）。"


def list_for_chat(chat_id: str, limit: int = 10) -> str:
    """会话任务列表（工具/命令共用）。"""
    from junjun_core.database.models import AsyncJob
    rows = list(AsyncJob.select().where(AsyncJob.chat_id == chat_id)
                .order_by(AsyncJob.created_at.desc()).limit(limit))
    if not rows:
        return "这个会话还没有后台任务。费时的活（查大量资料/整理报告）可以派给我慢慢做。"
    lines = ["后台任务（新->旧）："]
    for j in rows:
        status = _STATUS_CN.get(j.status, j.status)
        who = j.user_nickname or j.user_id
        extra = ""
        if j.status == "failed" and j.error:
            extra = f"（{j.error[:30]}）"
        lines.append(f"- #{j.job_id} [{status}] {j.title}——{who} 派的{extra}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 完成汇报

_LEAD_PROMPT = """你是"{nickname}"——{persona_brief}
你之前帮对方跑的后台任务{outcome}：「{title}」（委托人 QQ:{user_id}）。
用你的口吻写一句简短开场白：开头带 @{user_id} 提醒对方，说明任务{outcome}。
结果正文会跟在这句后面发，不要在开场白里复述结果内容。只输出这一句，口语化。"""


async def _persona_lead(job, ok: bool) -> str:
    """utils 模型写开场白（人设口吻），失败降级模板。"""
    outcome = "已经完成了" if ok else "搞砸了"
    fallback = (f"@{job.user_id} 你拜托我的「{job.title}」做好啦：" if ok
                else f"@{job.user_id} 你拜托我的「{job.title}」搞砸了……")
    try:
        from junjun_llm import get_chat_model, get_callbacks
        from langchain_core.messages import HumanMessage
        from junjun_agent.persona import persona_brief
        cfg = get_global_config()
        resp = await get_chat_model("utils").ainvoke(
            [HumanMessage(content=_LEAD_PROMPT.format(
                nickname=cfg.bot.nickname, persona_brief=persona_brief(),
                outcome=outcome, title=job.title, user_id=job.user_id))],
            config={"callbacks": get_callbacks()})
        out = str(resp.content).strip()
        if out:
            return out
    except Exception:
        pass
    return fallback


async def _notify(job) -> None:
    """完成/失败主动推送到任务所在会话。开场白保持人设，正文保真不过 LLM。"""
    ok = job.status == "done"
    body = job.result if ok else f"失败原因：{job.error or '未知'}"
    report_max = int(_cfg().get("report_max_chars", 1500))
    truncated = len(body) > report_max
    text = f"{await _persona_lead(job, ok)}\n{body[:report_max]}"
    if truncated:
        text += "\n……（内容太长只发这部分）"
    try:
        from junjun_core.contracts import ReplySegment
        from junjun_agent.outbound import send_proactive
        # 统一出站口：清洗 + 记忆回填 + 落库（任务汇报 bot 自己也得记得说过）
        ok_send = await send_proactive(job.chat_id, [ReplySegment(type="text", data=text)],
                                       source=f"async_job:{job.job_id}")
        if not ok_send:
            logger.warning(f"任务汇报发送失败 #{job.job_id}")
    except Exception as e:
        logger.warning(f"任务汇报发送失败 #{job.job_id}: {type(e).__name__}: {e}")
