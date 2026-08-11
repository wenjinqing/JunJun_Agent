"""事件雷达：群聊里的未来安排（开黑/考试/拼单/聚餐…）自动登记成提醒。

链路：预过滤（0 token 正则）→ utils_small 抽取 JSON → 校验（未来、≤7天）
→ 去重（同群同事项 ±30 分钟）→ 容量上限（每群 N 个待定）
→ 落 ReminderTasks（content 带「群事件：」前缀），复用提醒发送链。

只扫群聊非 bot 消息；全流程失败静默，绝不阻塞回复主链。
"""

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("loop.event_radar")

# 预过滤：含时间词才值得过 LLM（「明天见」这类误报由 LLM 判 is_event 挡掉）
_TIME_HINT = re.compile(
    r"明天|今晚|明晚|后天|大后天|下个?月|周[一二三四五六日天末]|星期[一二三四五六日天]"
    r"|下周|周末|中午|下午|晚上|早上|凌晨|\d{1,2}\s*[点：:半]\s*\d{0,2}|ddl|DDL")

_EVENT_PREFIX = "群事件："
_DEDUPE_WINDOW = 1800  # 同事项 ±30 分钟算重复

_EXTRACT_PROMPT = """你是事件提取器。判断这条群聊消息是否包含「未来的集体安排/约定」（开黑、聚餐、考试、DDL、拼单、团建等）。
当前时间：{now}（{weekday}）
消息（发送者昵称「{nickname}」）：「{text}」

注意：昵称是群友随便起的标签（可能是整段玩梗的话），只从「」内的消息正文判断，昵称内容不算。

只输出 JSON，不要别的：
- 不是未来的安排：{{"is_event": false}}
- 是：{{"is_event": true, "content": "事项（≤15字）", "time": "YYYY-MM-DD HH:MM"}}
time 按当前时间推算成绝对时间（如「周六晚八点」→ 最近的周六 20:00），只接受未来 {days} 天内的，拿不准就输出 false。"""

_WEEKDAYS = "一二三四五六日"


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("event_radar", {}) or {}
    except Exception:
        return {}


def should_scan(text: str) -> bool:
    """预过滤：长度合理且含时间词。"""
    if not text:
        return False
    text = text.strip()
    if not (4 <= len(text) <= 120):
        return False
    return bool(_TIME_HINT.search(text))


def parse_extraction(raw: str) -> Optional[dict]:
    """解析 LLM 抽取结果 -> {"content", "ts"}；非事件/格式烂/时间非法都返回 None。"""
    m = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not data.get("is_event"):
        return None
    content = str(data.get("content") or "").strip()[:20]
    if not content:
        return None
    try:
        ts = datetime.strptime(str(data.get("time")), "%Y-%m-%d %H:%M").timestamp()
    except (TypeError, ValueError):
        return None
    now = time.time()
    days = int(_cfg().get("lookahead_days", 7))
    if ts <= now + 60 or ts > now + days * 86400:  # 必须是 1 分钟后的未来，且不超前瞻
        return None
    return {"content": content, "ts": ts}


def _pending_radar(chat_id: str):
    from junjun_core.database import ReminderTasks
    return list(ReminderTasks.select().where(
        (ReminderTasks.chat_id == chat_id)
        & (ReminderTasks.is_completed == False)   # noqa: E712
        & (ReminderTasks.is_cancelled == False)   # noqa: E712
        & (ReminderTasks.content.startswith(_EVENT_PREFIX))))


def register_event(chat_id: str, user_id: str, nickname: str,
                   content: str, ts: float) -> bool:
    """去重 + 容量上限后落 ReminderTasks。返回是否真建了。"""
    pending = _pending_radar(chat_id)
    if len(pending) >= int(_cfg().get("max_pending", 5)):
        logger.info(f"[{chat_id}] 事件雷达已满（{len(pending)} 个待定），跳过: {content}")
        return False
    for t in pending:
        if content in t.content and abs(t.remind_time - ts) < _DEDUPE_WINDOW:
            return False
    lead = int(_cfg().get("lead_minutes", 15)) * 60
    fire_ts = max(ts - lead, time.time() + 60)  # 提前 lead 提醒，太近就 1 分钟后
    from junjun_agent.loop.reminder import create_reminder
    create_reminder(chat_id, user_id,
                    f"{_EVENT_PREFIX}{content}（{nickname} 说的）", fire_ts)
    logger.info(f"[{chat_id}] 雷达登记: {content} @ {time.strftime('%m-%d %H:%M', time.localtime(fire_ts))}")
    return True


def maybe_scan(chat_id: str, user_id: str, nickname: str, text: str) -> None:
    """入站钩子（同步、不阻塞）：预过滤过了才起后台任务。"""
    if not bool(_cfg().get("enable", True)):
        return
    if not should_scan(text or ""):
        return
    try:
        asyncio.get_running_loop().create_task(
            scan(chat_id, user_id, nickname, text))
    except RuntimeError:
        pass  # 无线程事件循环（测试/脚本环境）就跳过


async def scan(chat_id: str, user_id: str, nickname: str, text: str,
               *, model=None) -> bool:
    """LLM 抽取 + 登记。返回是否登记成功；任何失败静默。"""
    try:
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils_small")
        from langchain_core.messages import HumanMessage
        from junjun_memory.short_term import _sanitize_nickname
        now = datetime.now()
        resp = await model.ainvoke([HumanMessage(content=_EXTRACT_PROMPT.format(
            now=now.strftime("%Y-%m-%d %H:%M"),
            weekday=f"星期{_WEEKDAYS[now.weekday()]}",
            nickname=_sanitize_nickname(nickname) or "群友", text=text[:120],
            days=int(_cfg().get("lookahead_days", 7)),
        ))])
        event = parse_extraction(str(resp.content))
        if not event:
            return False
        return register_event(chat_id, user_id, nickname or "群友",
                              event["content"], event["ts"])
    except Exception as e:
        logger.debug(f"事件雷达扫描失败（忽略）: {type(e).__name__}: {e}")
        return False
