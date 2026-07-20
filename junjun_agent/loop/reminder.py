"""提醒任务：对齐原 proactive_system/reminder_task_manager 语义。

- skill 写 ReminderTasks 表；调度器 60s 轮询到期任务
- 到期 -> LLM 拟人化提醒文案 -> gateway 发送
- repeat_type: "" / daily / weekly；启动 load_pending 恢复
"""

import time
import uuid
from typing import List

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("loop.reminder")

_REMIND_PROMPT = """你是"{nickname}"。到点提醒群友了。提醒内容：「{content}」（提醒对象 QQ:{user_id}）。
用你的人设语气写一句自然的提醒消息（直接输出消息本身，简短口语化，可以带点调侃）。"""


def create_reminder(chat_id: str, user_id: str, content: str,
                    remind_time: float, repeat_type: str = "") -> str:
    """建提醒。返回 task_id。"""
    from junjun_core.database import ReminderTasks
    task_id = uuid.uuid4().hex[:12]
    ReminderTasks.create(
        task_id=task_id, chat_id=chat_id, user_id=user_id,
        content=content, remind_time=remind_time,
        repeat_type=repeat_type if repeat_type in ("daily", "weekly") else "",
    )
    logger.info(f"提醒已建 [{task_id}] {content[:30]} @ {time.strftime('%m-%d %H:%M', time.localtime(remind_time))}")
    return task_id


def list_pending(chat_id: str) -> List[dict]:
    from junjun_core.database import ReminderTasks
    rows = (ReminderTasks.select()
            .where((ReminderTasks.chat_id == chat_id)
                   & (ReminderTasks.is_completed == False)   # noqa: E712
                   & (ReminderTasks.is_cancelled == False))  # noqa: E712
            .order_by(ReminderTasks.remind_time))
    return [{"task_id": r.task_id, "content": r.content,
             "remind_time": r.remind_time, "repeat": r.repeat_type} for r in rows]


def cancel_reminder(task_id: str) -> bool:
    from junjun_core.database import ReminderTasks
    row = ReminderTasks.get_or_none(ReminderTasks.task_id == task_id)
    if row is None or row.is_completed or row.is_cancelled:
        return False
    row.is_cancelled = True
    row.save()
    logger.info(f"提醒已取消 [{task_id}]")
    return True


async def check_due_reminders() -> None:
    """调度器任务：轮询到期提醒并发送。单条失败不影响其余。"""
    from junjun_core.database import ReminderTasks
    now = time.time()
    due = list(ReminderTasks.select()
               .where((ReminderTasks.remind_time <= now)
                      & (ReminderTasks.is_completed == False)   # noqa: E712
                      & (ReminderTasks.is_cancelled == False))) # noqa: E712
    for task in due:
        try:
            await _fire(task)
        except Exception as e:
            logger.warning(f"提醒发送失败 [{task.task_id}]（下轮重试）: {e}")


async def _fire(task) -> None:
    cfg = get_global_config()
    # 拟人化文案（LLM 失败降级模板）
    text = f"@{task.user_id} 提醒：{task.content}"
    try:
        from junjun_llm import get_chat_model, get_callbacks
        from langchain_core.messages import HumanMessage
        model = get_chat_model("utils")
        resp = await model.ainvoke(
            [HumanMessage(content=_REMIND_PROMPT.format(
                nickname=cfg.bot.nickname, content=task.content, user_id=task.user_id))],
            config={"callbacks": get_callbacks()},
        )
        out = str(resp.content).strip()
        if out:
            text = out
    except Exception:
        pass

    # 发送（chat_id 格式 platform:id:type）
    parts = task.chat_id.split(":")
    platform, target_id, kind = parts[0], parts[1], parts[2] if len(parts) > 2 else "private"
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=[ReplySegment(type="text", data=text)],
        should_reply=True,
    ))

    # 周期任务顺延，一次性任务完成
    if task.repeat_type == "daily":
        task.remind_time += 86400
    elif task.repeat_type == "weekly":
        task.remind_time += 7 * 86400
    else:
        task.is_completed = True
    task.save()
    logger.info(f"提醒已发 [{task.task_id}] -> {task.chat_id}")


def load_pending_count() -> int:
    """启动日志用：未到期提醒数（轮询驱动无需重新调度，仅确认可见）。"""
    from junjun_core.database import ReminderTasks
    return (ReminderTasks.select()
            .where((ReminderTasks.is_completed == False)   # noqa: E712
                   & (ReminderTasks.is_cancelled == False))  # noqa: E712
            .count())
