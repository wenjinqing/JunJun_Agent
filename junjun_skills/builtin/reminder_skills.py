"""提醒 + 情绪类 skill（阶段 5 前置：提醒三件 + 情绪管理）。"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool

from junjun_skills.builtin.memory_skills import current_chat_id

_REL_RE = re.compile(r"(\d+)\s*(分钟|小时|天)后?")
# 日必须跟随月出现（否则会把 "11点" 的十位吞成日，回退后小时只剩个位）
_ABS_RE = re.compile(r"(?:(\d{1,2})月(\d{1,2})[日号]?)?\s*(\d{1,2})\s*[:点时]\s*(\d{1,2})?分?")
_DOT_RE = re.compile(r"(\d{1,2})\.(\d{2})")
_PM_WORDS = ("下午", "晚上", "傍晚")


def parse_remind_time(spec: str, *, now: Optional[datetime] = None) -> Optional[float]:
    """解析时间描述 -> timestamp。支持"10分钟后""明天8点""7月10日5:30""10.05"。"""
    now = now or datetime.now()
    spec = spec.strip()

    # 1. 相对时间: 10分钟后 / 2小时后 / 1天后
    m = _REL_RE.search(spec)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"分钟": timedelta(minutes=n), "小时": timedelta(hours=n), "天": timedelta(days=n)}[unit]
        return (now + delta).timestamp()

    # 2. 绝对时间带点号: 10.05 -> 10:05
    m = _DOT_RE.search(spec)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target = target + timedelta(days=1)
        return target.timestamp()

    # 3. 绝对时间: 明天8点 / 7月10日5:30
    base = now
    if "明天" in spec:
        base = now + timedelta(days=1)
    elif "后天" in spec:
        base = now + timedelta(days=2)

    m = _ABS_RE.search(spec)
    if m:
        month, day, hour, minute = m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 0)
        if m.group(4) is None and "半" in spec:
            minute = 30  # 12点半 -> 12:30
        # 上午/下午修饰：下午3点 -> 15:00；中午仅对凌晨小时数加 12（中午11点不加）
        if any(w in spec for w in _PM_WORDS) and hour < 12:
            hour += 12
        elif "中午" in spec and hour < 6:
            hour += 12
        if hour > 23 or minute > 59:
            return None
        target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if month and day:
            target = target.replace(month=int(month), day=int(day))
        if target <= now:  # 已过则视为明天/明年
            target = target + (timedelta(days=1) if not month else timedelta(days=365))
        return target.timestamp()
    return None


@tool
def set_reminder(content: str, time_spec: str, user_id: str) -> str:
    """设置提醒。用户说"10分钟后提醒我""明天8点叫我"时使用。
    Args:
        content: 提醒内容，如"开会"
        time_spec: 时间描述原文，如"10分钟后""明天8点""7月10日5:30"
        user_id: 要提醒的用户 QQ 号
    """
    ts = parse_remind_time(time_spec)
    if ts is None:
        return f"没听懂时间「{time_spec}」，换个说法？（支持：X分钟后 / 明天8点 / 7月10日5:30）"
    from junjun_agent.loop.reminder import create_reminder
    task_id = create_reminder(current_chat_id.get(), user_id, content, ts)
    when = time.strftime("%m月%d日 %H:%M", time.localtime(ts))
    return f"提醒已设好（{when}，编号 {task_id}）。"


@tool
def list_reminders() -> str:
    """查看当前会话待办的提醒列表。"""
    from junjun_agent.loop.reminder import list_pending
    items = list_pending(current_chat_id.get())
    if not items:
        return "当前没有待办的提醒。"
    lines = ["待办提醒："]
    for it in items:
        when = time.strftime("%m月%d日 %H:%M", time.localtime(it["remind_time"]))
        lines.append(f"- [{it['task_id']}] {when} {it['content']}")
    return "\n".join(lines)


@tool
def cancel_reminder_task(task_id: str) -> str:
    """取消一个提醒。用户说"取消那个提醒"时使用（先用 list_reminders 查编号）。
    Args:
        task_id: 提醒编号
    """
    from junjun_agent.loop.reminder import cancel_reminder
    return "已取消。" if cancel_reminder(task_id) else f"没找到编号 {task_id} 的有效提醒。"


@tool
def manage_mood(action: str, state: str = "") -> str:
    """读取或调整你自己的情绪。action="get" 查看当前情绪；action="set" 主动调整（如被安慰后心情变好）。
    Args:
        action: get 或 set
        state: set 时的新情绪短语，如"开心"
    """
    from junjun_express.mood import mood_manager
    chat_id = current_chat_id.get()
    if action == "set" and state:
        mood_manager.set_mood(chat_id, state)
        return f"情绪已调整为：{state}"
    return f"当前情绪：{mood_manager.get_mood(chat_id) or '（情绪系统未启用）'}"
