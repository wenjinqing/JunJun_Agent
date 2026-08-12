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


_WEEKDAY_MAP = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_WEEKLY_RE = re.compile(r"每(?:周|星期)([一二三四五六日天1-7])")
_WEEKDAY_RE = re.compile(r"(?:周|星期)([一二三四五六日天1-7])")
# 「天天」单独处理：「明天天气」子串撞车（router 同款坑）——前面是 明/今/昨 时不算
_TIANTIAN_EXCLUDE = ("明", "今", "昨")


def parse_repeat_type(spec: str) -> tuple:
    """识别周期表达，返回 (repeat_type, 去掉周期词后的时间描述)。

    「每天早上8点」-> ("daily", "早上8点")；「每周五晚上8点」-> ("weekly", "周五晚上8点")；
    无周期词 -> ("", 原样)。repeat_type 对应 ReminderTasks.repeat_type（""/daily/weekly）。
    """
    for w in ("每天", "每日"):
        if w in spec:
            return "daily", spec.replace(w, "", 1).strip()
    idx = spec.find("天天")
    if idx != -1 and (idx == 0 or spec[idx - 1] not in _TIANTIAN_EXCLUDE):
        return "daily", spec.replace("天天", "", 1).strip()
    m = _WEEKLY_RE.search(spec)
    if m:
        rest = (spec[:m.start()] + "周" + m.group(1) + spec[m.end():]).strip()
        return "weekly", rest
    return "", spec


def parse_weekly_time(spec: str, *, now: Optional[datetime] = None) -> Optional[float]:
    """「周五晚上8点」-> 下一个该 weekday 的 timestamp（今天已过则顺延一周）。"""
    now = now or datetime.now()
    wd = _WEEKDAY_RE.search(spec)
    hm = _ABS_RE.search(spec)
    if not wd or not hm:
        return None
    ch = wd.group(1)
    weekday = (int(ch) - 1) % 7 if ch.isdigit() else _WEEKDAY_MAP[ch]
    hour, minute = int(hm.group(3)), int(hm.group(4) or 0)
    if hm.group(4) is None and "半" in spec:
        minute = 30
    if any(w in spec for w in _PM_WORDS) and hour < 12:
        hour += 12
    elif "中午" in spec and hour < 6:
        hour += 12
    if hour > 23 or minute > 59:
        return None
    days_ahead = (weekday - now.weekday()) % 7
    target = (now + timedelta(days=days_ahead)).replace(
        hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=7)
    return target.timestamp()


@tool
def set_reminder(content: str, time_spec: str, user_id: str) -> str:
    """设置提醒（单次与周期都支持）。只在对方【明确要求】提醒时使用
    （"10分钟后提醒我""明天8点叫我""每周五晚上提醒我交周报"）——
    不许主动/顺手帮人设提醒；群里不可设（会打扰全群，引导私聊）。
    Args:
        content: 提醒内容，如"开会"
        time_spec: 时间描述原文，如"10分钟后""明天8点""每天早上8点""每周五晚上8点"
        user_id: 要提醒的用户 QQ 号
    """
    # 群禁（2026-08-12 用户裁决：群里禁用+收敛）——群提醒到点 @ 全群观感脑残，
    # 存量群提醒在触发侧改投私聊（loop/reminder.py _fire）
    if current_chat_id.get().endswith(":group"):
        return "提醒不在群里设（到点会打扰全群）。想要这个提醒的话，让对方私聊我再说一遍。"
    repeat, spec = parse_repeat_type(time_spec.strip())
    ts = parse_weekly_time(spec) if repeat == "weekly" else parse_remind_time(spec)
    if ts is None:
        return (f"没听懂时间「{time_spec}」，换个说法？"
                f"（支持：X分钟后 / 明天8点 / 每天早上8点 / 每周五晚上8点 / 7月10日5:30）")
    from junjun_agent.loop.reminder import create_reminder
    task_id = create_reminder(current_chat_id.get(), user_id, content, ts,
                              repeat_type=repeat)
    when = time.strftime("%m月%d日 %H:%M", time.localtime(ts))
    freq = {"daily": "，之后每天", "weekly": "，之后每周"}.get(repeat, "")
    return f"提醒已设好（{when} 起{freq}，编号 {task_id}）。"


@tool
def list_reminders() -> str:
    """查看当前会话待办的提醒列表。对方问「我有什么提醒/你提醒我什么了/提醒设好了吗」，
    或你要取消提醒前需要查编号时使用。返回每条提醒的编号、时间和内容。"""
    from junjun_agent.loop.reminder import list_pending
    items = list_pending(current_chat_id.get())
    if not items:
        return "当前没有待办的提醒。"
    lines = ["待办提醒："]
    for it in items:
        when = time.strftime("%m月%d日 %H:%M", time.localtime(it["remind_time"]))
        # 周期标注（2026-08-09 审查：周期提醒与一次性在列表里无区别，
        # 「取消那个每天的推送」时模型无从确认目标）
        rep = {"daily": "每天", "weekly": "每周"}.get(it.get("repeat") or "", "")
        tag = f"（{rep}）" if rep else ""
        lines.append(f"- [{it['task_id']}] {tag}{when} {it['content']}")
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
