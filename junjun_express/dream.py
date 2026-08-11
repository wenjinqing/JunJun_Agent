"""君君的梦境：清晨把昨天的记忆碎片揉成一段超现实的梦，发 QQ 空间。

像真人的点：真人会做梦，还会把梦讲得半真半假。素材全来自昨天真实
沉淀的东西（日记 + 话题摘要 + 心境），但经梦的滤镜重组——既天马行空，
又隐隐约约有昨天的影子。

- 产出：junzone 发空间（与 auto_post 同级风险，免审）+ 写进长期记忆
  （self:diary 域带「我的梦」前缀，日常召回自然带出）
- 隐私纪律与日记相同：素材含私聊场景时不外发，只写 private 记忆域
- 调度：dream_tick（30 分钟一轮），到点且今天没做过梦才写
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.dream")

_STATE = Path(__file__).resolve().parents[1] / "data" / "dream" / "last.json"

_DREAM_PROMPT = """你是 QQ 机器人「{nickname}」。现在是清晨，你刚睡醒，要把昨晚做的梦发到 QQ 空间（第一人称）。

你昨天经历/记住的碎片：
{fragments}

要求：
- 把这些碎片揉成一段超现实的梦：逻辑可以飞，细节要从碎片里长出来（提到人用昵称）
- 第一人称口语，100~200 字，可以无厘头、可以温柔，结尾一句刚睡醒的感慨
- 认真写，不要解释「AI 不会做梦」——这就是你的梦
- 只输出梦的正文，不要任何前后缀"""


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("dream", {}) or {}
    except Exception:
        return {}


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _last_dream_day() -> str:
    try:
        if _STATE.exists():
            return str(json.loads(_STATE.read_text(encoding="utf-8")).get("date", ""))
    except Exception:
        pass
    return ""


def _mark_dreamed(day: str) -> None:
    try:
        _STATE.parent.mkdir(parents=True, exist_ok=True)
        _STATE.write_text(json.dumps({"date": day}), encoding="utf-8")
    except Exception as e:
        logger.warning(f"梦境状态落盘失败: {e}")


# ---------------------------------------------------------------- 素材

def _gather_fragments() -> tuple:
    """昨天的碎片 -> (文本, 是否含私聊)。含私聊则不外发（隐私纪律 P0-4）。"""
    parts, private = [], False
    # 昨天的日记（一天经历的最高浓度蒸馏）
    try:
        from junjun_core.database.models import DiaryEntry, _bot_id
        yesterday = datetime.fromtimestamp(time.time() - 86400).strftime("%Y-%m-%d")
        for day in {yesterday, _today()}:
            row = DiaryEntry.get_or_none(
                (DiaryEntry.bot_id == _bot_id()) & (DiaryEntry.date == day))
            if row and row.content:
                parts.append(f"我的日记（{day}）：{row.content[:300]}")
                break
    except Exception:
        pass
    # 近 24h 归档的话题摘要（各群）
    try:
        from junjun_memory.summarizer import HIPPO_DIR
        cutoff = time.time() - 86400
        for p in sorted(HIPPO_DIR.glob("*.json"))[:10]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "private" in p.stem:
                private = True
                continue
            for a in data.get("archived", []):
                if float(a.get("time", 0) or 0) >= cutoff and a.get("summary"):
                    parts.append(str(a["summary"])[:80])
    except Exception:
        pass
    # 当前心境（梦的情绪底色）
    try:
        from junjun_express.mood import mood_manager
        mood = mood_manager.get_self_mood()
        if mood:
            parts.append(f"我睡前的心情：{mood}")
    except Exception:
        pass
    return "\n".join(f"- {s}" for s in parts[-12:]), private


# ---------------------------------------------------------------- 产出

async def write_dream(*, model=None, force: bool = False,
                      callbacks=None) -> Optional[str]:
    """写今天的梦。已写过且非 force 跳过；失败返回 None。"""
    cfg = _cfg()
    if not force and not bool(cfg.get("enable", False)):
        return None
    day = _today()
    if not force and _last_dream_day() == day:
        return None
    try:
        fragments, private = _gather_fragments()
        if not fragments:
            logger.info("昨天什么碎片都没留下，梦也做不了，跳过")
            return None
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("thinker")  # 梦境：低频重氛围，开思考
        from langchain_core.messages import HumanMessage
        resp = await model.ainvoke([HumanMessage(content=_DREAM_PROMPT.format(
            nickname=get_global_config().bot.nickname,
            fragments=fragments))], config={"callbacks": callbacks or []})
        content = str(resp.content).strip()[:500]
        if not content:
            logger.warning("梦境生成为空，跳过")
            return None
        _mark_dreamed(day)
        # 进长期记忆（self:diary 域，召回自然带出）
        try:
            from junjun_memory.long_term import get_long_term_memory
            domain = "self:diary:private" if private else "self:diary"
            await get_long_term_memory().add(
                f"[我的梦 {day}] {content}", chat_id=domain,
                weight=1.2, kind="diary")
        except Exception as e:
            logger.warning(f"梦境写入长期记忆失败（不影响发空间）: {e}")
        # 发空间（素材含私聊时只写记忆不外发）
        published = False
        if not private and str(cfg.get("target", "qzone")) == "qzone":
            try:
                from junjun_skills.plugins.junzone.tools import (
                    _with_auth_retry, publish_feed, _bot_uin)
                tid = await _with_auth_retry(publish_feed, _bot_uin(),
                                             f"【昨晚的梦】{content}")
                published = tid is not None
            except Exception as e:
                logger.warning(f"梦境发空间失败（已写记忆）: {type(e).__name__}: {e}")
        logger.info(f"梦境已写（{day}，{len(content)} 字，"
                    f"{'已发空间' if published else '仅记忆'}）")
        return content
    except Exception as e:
        logger.warning(f"写梦失败: {type(e).__name__}: {e}")
        return None


async def dream_tick() -> None:
    """定时 tick（30 分钟一轮）：到点且今天没做过梦才写。"""
    cfg = _cfg()
    if not bool(cfg.get("enable", False)):
        return
    if _last_dream_day() == _today():
        return
    hh, mm = str(cfg.get("time", "07:30")).split(":")[:2]
    if datetime.now() >= datetime.now().replace(hour=int(hh), minute=int(mm),
                                                second=0, microsecond=0):
        await write_dream()
