"""私人日记：每天定时把「今天」写成第一人称日记——自我叙事连续性的来源。

- 素材：当天自己的发言（Messages is_bot）、junzone 说说条数、各场景情绪、当前心境
- 产出：DiaryEntry 落库 + 写入长期记忆（chat_id="self:diary", kind="diary"），
  日常记忆召回自然带出（processor 召回域含 "self:diary"）；
  末尾的「今日心情」沉淀为全局自我心境（mood.set_self_mood）
- 隐私：日记是内心活动，persona 规则要求可以带着感受说话但不逐字复述
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.diary")

_JUNZONE_FEEDS = Path(__file__).resolve().parents[1] / "data" / "junzone" / "feeds_sent.json"

_DIARY_PROMPT = """你是 QQ 机器人「{nickname}」，睡前写一篇今天的私人日记（第一人称，只给自己看）。

今天的时间：{now}
你当前的心境：{self_mood}

今天发生的事：
{material}

要求：
- 第一人称口语化，像真人随手写的日记，150~250 字
- 有具体的人和事就写具体的，没什么事就写安静的一天（发呆/摸鱼/想事情都行）
- 可以有小情绪、小吐槽、小期待，不要写成工作汇报
- 不要泄露任何人的 QQ 号，提到人用昵称或「某人」
- 最后一行单独写：心情：XX（一个短语，作为今天的收尾心情）"""


def _cfg() -> dict:
    """读取 [diary] 配置节（热改生效）。"""
    try:
        return get_global_config().raw.get("diary", {}) or {}
    except Exception:
        return {}


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 素材收集

def _bot_messages_today(day: str) -> list:
    """当天自己的发言（场景 + 截断文本）。"""
    start_ts = datetime.strptime(day, "%Y-%m-%d").timestamp()
    try:
        from junjun_core.database.models import Messages
        rows = list(Messages.select()
                    .where((Messages.is_bot == True) & (Messages.time >= start_ts))  # noqa: E712
                    .order_by(Messages.time.asc()).limit(300))
    except Exception as e:
        logger.warning(f"日记素材：读消息失败: {e}")
        return []
    lines = []
    for r in rows:
        text = (r.processed_plain_text or "").strip()
        if not text:
            continue
        scene = "群里" if r.chat_id.endswith(":group") else "私聊"
        lines.append(f"[{scene}] {text[:60]}")
    # 太多则均匀抽样 30 条（保时间序，头尾都留）
    if len(lines) > 30:
        step = len(lines) / 30
        lines = [lines[int(i * step)] for i in range(30)]
    return lines


def _junzone_feed_count(day: str) -> int:
    """当天发过的说说条数。"""
    try:
        if _JUNZONE_FEEDS.exists():
            data = json.loads(_JUNZONE_FEEDS.read_text(encoding="utf-8"))
            if data.get("date") == day:
                return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def _chat_moods_snapshot() -> list:
    """各场景当前情绪（非平静的才记）。"""
    from junjun_express.mood import mood_manager, _DEFAULT_MOOD
    out = []
    for chat_id, cm in mood_manager._moods.items():
        if cm.state and cm.state != _DEFAULT_MOOD:
            scene = "群里" if chat_id.endswith(":group") else "私聊"
            out.append(f"{scene}：{cm.state}")
    return out[:5]


def _gather_material(day: str) -> str:
    lines = _bot_messages_today(day)
    parts = []
    if lines:
        parts.append("我今天说过的话（节选）：\n" + "\n".join(lines))
    feed_count = _junzone_feed_count(day)
    if feed_count:
        parts.append(f"我今天在 QQ 空间发了 {feed_count} 条说说。")
    moods = _chat_moods_snapshot()
    if moods:
        parts.append("今天各场景的情绪：" + "；".join(moods))
    if not parts:
        parts.append("今天很安静，几乎没人找我说话。")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 产出解析

def _parse_diary_output(text: str) -> tuple:
    """解析 LLM 产出 -> (正文, 心情)。心情行缺失则正文即全部。"""
    text = (text or "").strip()
    mood = ""
    m = re.search(r"^心情[：:]\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        mood = m.group(1).strip()[:20]
        text = (text[:m.start()] + text[m.end():]).strip()
    return text[:600], mood


def _save_entry(day: str, content: str, mood: str) -> None:
    from junjun_core.database.models import DiaryEntry, _bot_id
    row = DiaryEntry.get_or_none(
        (DiaryEntry.bot_id == _bot_id()) & (DiaryEntry.date == day))
    if row is None:
        DiaryEntry.create(date=day, content=content, mood=mood, created_at=time.time())
    else:
        row.content, row.mood, row.created_at = content, mood, time.time()
        row.save()


def _get_entry(day: str):
    from junjun_core.database.models import DiaryEntry, _bot_id
    return DiaryEntry.get_or_none(
        (DiaryEntry.bot_id == _bot_id()) & (DiaryEntry.date == day))


async def _index_to_memory(day: str, content: str) -> None:
    """日记进长期记忆（self:diary 域），日常召回自然带出。"""
    try:
        from junjun_memory.long_term import get_long_term_memory
        await get_long_term_memory().add(
            f"[我的日记 {day}] {content}", chat_id="self:diary", weight=1.3, kind="diary")
    except Exception as e:
        logger.warning(f"日记写入长期记忆失败（已落库，不影响）: {e}")


# ---------------------------------------------------------------- 主流程

async def write_diary(*, model=None, force: bool = False, callbacks=None) -> Optional[str]:
    """写今天的日记。已存在且非 force 时跳过（返回 None）。失败返回 None。"""
    cfg = _cfg()
    if not force and not bool(cfg.get("enable", True)):
        return None
    day = _today()
    if not force and _get_entry(day) is not None:
        return None

    try:
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        from langchain_core.messages import HumanMessage
        from junjun_express.mood import mood_manager
        prompt = _DIARY_PROMPT.format(
            nickname=get_global_config().bot.nickname,
            now=datetime.now().strftime("%Y-%m-%d %H:%M"),
            self_mood=mood_manager.get_self_mood() or "平静",
            material=_gather_material(day),
        )
        resp = await model.ainvoke([HumanMessage(content=prompt)],
                                   config={"callbacks": callbacks or []})
        content, mood = _parse_diary_output(str(resp.content))
        if not content:
            logger.warning("日记生成为空，跳过")
            return None
        _save_entry(day, content, mood)
        await _index_to_memory(day, content)
        if mood:
            mood_manager.set_self_mood(mood, reason="日记")
        logger.info(f"日记已写（{day}，{len(content)} 字，心情：{mood or '未标注'}）")
        # 意向系统反思钩子（P7）：日记 -> 「明天想做的事」意向（enable=false 零开销）
        try:
            from junjun_agent.loop.intention import on_diary_written
            await on_diary_written(content, model=model)
        except Exception:
            pass
        return content
    except Exception as e:
        logger.warning(f"写日记失败: {type(e).__name__}: {e}")
        return None


async def diary_tick() -> None:
    """定时 tick（30 分钟一轮）：到点且今天还没写就写。"""
    cfg = _cfg()
    if not bool(cfg.get("enable", True)):
        return
    day = _today()
    if _get_entry(day) is not None:
        return
    hh, mm = str(cfg.get("time", "23:30")).split(":")[:2]
    if datetime.now() >= datetime.now().replace(hour=int(hh), minute=int(mm),
                                                second=0, microsecond=0):
        await write_diary()
