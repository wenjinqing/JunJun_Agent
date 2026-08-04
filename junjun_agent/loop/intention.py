"""意向系统（P7，路线核心）：她做事是因为她想做，不是因为被 @。

三段式：
1. 生成（spawn）：事件钩子（有人 emo——规则预筛 0 token）、定时巡检
   （晨起问早）、日记反思（「明天想做的事」）。生成全部走规则/轻量 LLM，
   每会话同时存活 ≤max_pending，优先级淘汰，过期即焚
2. 评估门（evaluate）：免打扰时段 / 会话与全局日限额 / 同类去重 /
   亲密度门槛 / 最小沉淀时间（emo 不当场追问，像真人过几小时再问）
3. 执行（tick）：agent 槽带动机生成消息 -> utils 槽质量自检（尬/冒犯/泄密
   直接丢弃）-> gateway 发送；连续失败熔断一天

灰度原则：[intention] enable 默认 false，管理员按会话观察后再放量。
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("loop.intention")

_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "intention_state.json"

# emo 规则预筛（事件源，0 token）：命中才 spawn 关心意向
_EMO_KEYWORDS = ("焦虑", "失眠", "崩溃", "考砸", "分手", "抑郁", "想哭",
                 "好难受", "压力大", "心累", "emo")

# 各 kind 最小沉淀时间（分钟）：emo 关心不能当场追问，像真人隔几小时再问
_MIN_AGE = {"care_followup": 180, "diary_plan": 60, "morning_greet": 0}

_GEN_PROMPT = """你是「{nickname}」——{persona_brief}
你现在想主动做一件事：{motive}
{recent_block}
写一两句自然的话去实现这个想法（像真人忽然想起来的语气，别解释你在做什么，
直接写要说的话本身）。"""

_JUDGE_PROMPT = """你是把关员。聊天机器人想主动发出这条消息：
「{message}」
背景动机：{motive}
判断这条消息主动发出是否合适：会不会尬、冒犯、泄露别人隐私（把私下的事说到群里）？
只输出：合适 / 不合适"""


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("intention", {}) or {}
    except Exception:
        return {}


def _enabled() -> bool:
    return bool(_cfg().get("enable", False))


# ---------------------------------------------------------------- 熔断状态

def _load_state() -> dict:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(st: dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_STATE_PATH)
    except Exception as e:
        logger.debug(f"intention 状态落盘失败（忽略）: {e}")


# ---------------------------------------------------------------- 会话安静模式（P7-4 用户侧控制）
# 与熔断共用 state 文件：mute = {chat_id: until_ts}，-1 = 持久（/热闹 解除）

def mute_chat(chat_id: str, *, hours: float = 24) -> None:
    """安静模式：期间意向/主动消息都不发（提醒类必达不受影响）。hours=0 表持久。"""
    st = _load_state()
    mute = st.setdefault("mute", {})
    mute[chat_id] = -1 if hours <= 0 else time.time() + hours * 3600
    _save_state(st)
    logger.info(f"[{chat_id}] 安静模式开启（{'持久' if hours <= 0 else f'{hours}h'}）")


def unmute_chat(chat_id: str) -> None:
    st = _load_state()
    if st.get("mute", {}).pop(chat_id, None) is not None:
        _save_state(st)
        logger.info(f"[{chat_id}] 安静模式解除")


def chat_muted(chat_id: str) -> bool:
    until = float(_load_state().get("mute", {}).get(chat_id, 0.0))
    if until == -1:
        return True
    if until > time.time():
        return True
    if until:  # 过期顺手清理
        unmute_chat(chat_id)
    return False


# 「别烦我」类自然语言（只有对 bot 说的才算——processor 侧按 at_bot/直呼过滤，
#  群里两个人互怼「别烦我」不能误伤全群）
_MUTE_REQUESTS = ("别烦我", "安静点", "别主动", "别来找我", "少说话", "勿扰",
                  "不要主动", "消停点")


def detect_mute_request(text: str) -> bool:
    low = (text or "").strip()
    return bool(low) and any(kw in low for kw in _MUTE_REQUESTS)


def _circuit_open() -> bool:
    return time.time() < float(_load_state().get("pause_until", 0.0))


def _record_send_result(ok: bool) -> None:
    st = _load_state()
    fails = 0 if ok else int(st.get("consec_send_fails", 0)) + 1
    st["consec_send_fails"] = fails
    if fails >= 3:
        st["pause_until"] = time.time() + 86400
        st["consec_send_fails"] = 0
        logger.warning("意向发送连续失败 3 次，熔断 24 小时")
    _save_state(st)


# ---------------------------------------------------------------- 生成（spawn）

def _pending_of_chat(chat_id: str) -> list:
    from junjun_core.database.models import Intention, _bot_id
    return list(Intention.select().where(
        (Intention.bot_id == _bot_id())
        & (Intention.chat_id == chat_id)
        & (Intention.status == "pending")))


def spawn(kind: str, chat_id: str, motive: str, *, user_id: str = "",
          user_nickname: str = "", priority: int = 5, ttl_hours: float = 24) -> bool:
    """入队一个意向。去重（同类同会话同人挂着重置动机与过期），
    超上限按优先级淘汰（新的不如旧的硬直接丢）。返回是否真正新建。"""
    if not _enabled() or not motive.strip():
        return False
    from junjun_core.database.models import Intention
    now = time.time()
    expires = now + ttl_hours * 3600
    for it in _pending_of_chat(chat_id):
        if it.kind == kind and it.user_id == user_id:
            it.motive, it.expires_at = motive[:200], expires
            it.priority = min(it.priority, priority)
            it.save()
            return False
    pending = _pending_of_chat(chat_id)
    max_pending = int(_cfg().get("max_pending_per_chat", 3))
    if len(pending) >= max_pending:
        worst = max(pending, key=lambda it: (it.priority, -it.created_at))
        if priority < worst.priority:
            worst.status = "dropped"
            worst.save()
        else:
            return False  # 新的优先级不如在队的，直接弃
    Intention.create(kind=kind, chat_id=chat_id, user_id=user_id,
                     user_nickname=user_nickname, motive=motive[:200],
                     priority=priority, status="pending",
                     created_at=now, expires_at=expires)
    logger.info(f"意向入队 [{kind}] {chat_id}: {motive[:40]}")
    return True


def spawn_care_if_needed(chat_id: str, user_id: str, nickname: str, text: str) -> bool:
    """事件钩子（processor 每条消息调）：emo 规则预筛 -> 关心意向。

    0 token；像真人：看到朋友 emo 不会当场轰炸，过几小时再问一句。
    """
    if not _enabled():
        return False
    low = (text or "").lower()
    if not any(kw in low for kw in _EMO_KEYWORDS):
        return False
    who = nickname or "ta"
    excerpt = (text or "").strip()[:40]
    return spawn("care_followup", chat_id,
                 f"{who} 之前说「{excerpt}」，似乎心情不太好，想关心一下 ta 后来好点没",
                 user_id=user_id, user_nickname=nickname, priority=3, ttl_hours=12)


async def on_diary_written(diary_content: str, *, model=None) -> int:
    """日记钩子：写完日记蒸馏「明天想做的事」-> diary_plan 意向（0-2 条）。"""
    if not _enabled():
        return 0
    try:
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        from langchain_core.messages import HumanMessage
        prompt = (
            "这是你的日记：\n" + diary_content[:800] +
            "\n\n从日记里提炼「明天想主动做的事」，最多 2 件（比如昨天谁的话没接完、"
            "谁的事值得跟进）。输出 JSON 数组，每条 {\"motive\": \"一句话\"}，"
            "没有就输出 []。只输出 JSON。")
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        import re
        m = re.search(r"\[.*\]", str(resp.content), re.S)
        plans = json.loads(m.group(0)) if m else []
    except Exception as e:
        logger.debug(f"日记意向蒸馏失败（忽略）: {type(e).__name__}: {e}")
        return 0
    chat_id = _most_active_chat_yesterday()
    if not chat_id:
        return 0
    n = 0
    for p in plans[:2]:
        motive = str((p or {}).get("motive") if isinstance(p, dict) else "").strip()
        if motive and spawn("diary_plan", chat_id, motive, priority=6, ttl_hours=24):
            n += 1
    return n


def _most_active_chat_yesterday() -> str:
    from junjun_core.database.models import Messages
    from peewee import fn
    since = time.time() - 86400
    row = (Messages.select(Messages.chat_id, fn.COUNT(Messages.id).alias("n"))
           .where(Messages.time >= since)
           .group_by(Messages.chat_id)
           .order_by(fn.COUNT(Messages.id).desc()).first())
    return row.chat_id if row else ""


def spawn_scheduled_checks() -> int:
    """定时巡检（tick 到点调）：给近 48h 活跃会话排「早上好」意向。"""
    if not _enabled():
        return 0
    morning = str(_cfg().get("morning_time", "09:00"))
    hh, mm = morning.split(":")[:2]
    now = datetime.now()
    if (now.hour, now.minute) < (int(hh), int(mm)) or now.hour >= 12:
        return 0  # 只在早晨窗口排
    from junjun_core.database.models import Messages, Intention, _bot_id
    from peewee import fn
    since = time.time() - 48 * 3600
    rows = list(Messages.select(Messages.chat_id, fn.COUNT(Messages.id).alias("n"))
                .where(Messages.time >= since)
                .group_by(Messages.chat_id)
                .order_by(fn.COUNT(Messages.id).desc()).limit(5))
    # 今天已经排过/发过的不重复（按天去重）
    today_start = now.replace(hour=0, minute=0, second=0).timestamp()
    done_today = {it.chat_id for it in Intention.select().where(
        (Intention.bot_id == _bot_id())
        & (Intention.kind == "morning_greet")
        & (Intention.created_at >= today_start))}
    n = 0
    for r in rows:
        if r.chat_id in done_today:
            continue
        if spawn("morning_greet", r.chat_id,
                 "早上了，想跟最近聊过的大家自然说声早", priority=7, ttl_hours=3):
            n += 1
    return n


# ---------------------------------------------------------------- 评估门

def _in_quiet_hours() -> bool:
    from junjun_agent.funnel.frequency import _in_range
    spec = str(_cfg().get("quiet_hours", "23:00-08:00"))
    now = datetime.now()
    return _in_range(now.hour * 60 + now.minute, spec)


def _fired_today(chat_id: str = "") -> int:
    from junjun_core.database.models import Intention, _bot_id
    today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
    cond = ((Intention.bot_id == _bot_id())
            & (Intention.status == "fired")
            & (Intention.fired_at >= today_start))
    if chat_id:
        cond &= (Intention.chat_id == chat_id)
    return Intention.select().where(cond).count()


def evaluate(it, *, now: Optional[float] = None) -> tuple:
    """评估门：过闸 (True, "")；拦截 (False, 原因)。所有拦截打日志可审计。"""
    now = now or time.time()
    cfg = _cfg()
    if not _enabled():
        return False, "disabled"
    if _circuit_open():
        return False, "circuit_open"
    if chat_muted(it.chat_id):
        return False, "chat_muted"
    if _in_quiet_hours():
        return False, "quiet_hours"
    if it.expires_at and it.expires_at <= now:
        return False, "expired"
    min_age = float(cfg.get("min_age_minutes", 60))
    min_age = _MIN_AGE.get(it.kind, min_age) * 60
    if now - it.created_at < min_age:
        return False, "too_fresh"
    if _fired_today(it.chat_id) >= int(cfg.get("max_per_chat_per_day", 2)):
        return False, "chat_daily_cap"
    if _fired_today() >= int(cfg.get("max_global_per_day", 10)):
        return False, "global_daily_cap"
    # 同类 24h 去重：刚发过同 kind 的不炒冷饭
    from junjun_core.database.models import Intention, _bot_id
    dup = (Intention.select()
           .where((Intention.bot_id == _bot_id())
                  & (Intention.chat_id == it.chat_id)
                  & (Intention.kind == it.kind)
                  & (Intention.status == "fired")
                  & (Intention.fired_at >= now - 86400))
           .count())
    if dup:
        return False, "same_kind_fired_24h"
    # 亲密度门槛：关心类意向对低好感对象不硬贴
    if it.kind == "care_followup" and it.user_id:
        try:
            from junjun_express.intimacy import get_intimacy
            score = float(get_intimacy(it.user_id)[0])
            if score < float(cfg.get("min_intimacy", 30)):
                return False, "low_intimacy"
        except Exception:
            pass  # 亲密度查询失败不拦（默认放行，宁严勿滥再议）
    return True, ""


def expire_sweep() -> int:
    """过期即焚。"""
    from junjun_core.database.models import Intention, _bot_id
    n = (Intention.update(status="expired")
         .where((Intention.bot_id == _bot_id())
                & (Intention.status == "pending")
                & (Intention.expires_at <= time.time()))
         .execute())
    return n


def due_intentions(*, now: Optional[float] = None) -> list:
    from junjun_core.database.models import Intention, _bot_id
    now = now or time.time()
    return list(Intention.select().where(
        (Intention.bot_id == _bot_id())
        & (Intention.status == "pending")
        & (Intention.expires_at > now))
        .order_by(Intention.priority.asc(), Intention.created_at.asc()))


# ---------------------------------------------------------------- 执行

def _chat_target(chat_id: str) -> tuple:
    """chat_id -> (group_id, user_id)。"""
    parts = (chat_id or "").split(":")
    if len(parts) >= 3 and parts[2] == "group":
        return parts[1], None
    return None, parts[1] if len(parts) >= 2 else None


def _recent_context(chat_id: str) -> str:
    try:
        from junjun_core.gateway.session_manager import get_session_manager
        s = get_session_manager().all_sessions().get(chat_id)
        if s and s.memory:
            return "会话最近的聊天：\n" + s.memory.render(limit=8)
    except Exception:
        pass
    return ""


async def _generate_and_judge(it, *, gen_model=None, judge_model=None) -> str:
    """动机 -> 一句自然的话 -> 质量自检。不合适返回 ""。"""
    if gen_model is None or judge_model is None:
        from junjun_llm import get_chat_model
        gen_model = gen_model or get_chat_model("agent")
        judge_model = judge_model or get_chat_model("utils")
    from langchain_core.messages import HumanMessage
    from junjun_agent.persona import persona_brief
    cfg = get_global_config()
    recent = _recent_context(it.chat_id)
    resp = await gen_model.ainvoke([HumanMessage(content=_GEN_PROMPT.format(
        nickname=cfg.bot.nickname, persona_brief=persona_brief(), motive=it.motive,
        recent_block=recent))])
    message = str(resp.content).strip().splitlines()[0][:200]
    if not message:
        return ""
    judge = await judge_model.ainvoke([HumanMessage(content=_JUDGE_PROMPT.format(
        message=message, motive=it.motive))])
    if "不合适" in str(judge.content):
        logger.info(f"意向 #{it.id} 消息被质量自检拦截: {message[:30]}")
        return ""
    return message


async def _fire(it, *, gen_model=None, judge_model=None) -> bool:
    """生成 -> 自检 -> 发送 -> 标记。返回是否发出。"""
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    message = await _generate_and_judge(it, gen_model=gen_model, judge_model=judge_model)
    if not message:
        it.status = "dropped"
        it.save()
        return False
    group_id, user_id = _chat_target(it.chat_id)
    try:
        await get_gateway().send_reply(ReplySet(
            platform="qq", target_group_id=group_id, target_user_id=user_id,
            segments=[ReplySegment(type="text", data=message)],
            should_reply=True,
        ))
    except Exception as e:
        logger.warning(f"意向 #{it.id} 发送失败: {type(e).__name__}: {e}")
        _record_send_result(False)
        return False  # 保持 pending，下个 tick 再试（熔断会兜底）
    _record_send_result(True)
    it.status, it.fired_at = "fired", time.time()
    it.save()
    # 发出的消息进会话短期记忆（她自己知道说过这话）
    try:
        from junjun_core.gateway.session_manager import get_session_manager
        s = get_session_manager().all_sessions().get(it.chat_id)
        if s and s.memory:
            s.memory.add_bot(message)
    except Exception:
        pass
    logger.info(f"意向 #{it.id} [{it.kind}] 已发出: {message[:40]}")
    return True


async def intention_tick(*, gen_model=None, judge_model=None) -> int:
    """调度入口（10 分钟一轮）：排晨检 -> 过期清理 -> 逐条过闸执行。
    返回本轮发出数。每轮最多发 2 条（防齐发像机器人）。"""
    if not _enabled():
        return 0
    try:
        spawn_scheduled_checks()
    except Exception as e:
        logger.debug(f"晨检排意向失败（忽略）: {e}")
    expire_sweep()
    fired = 0
    for it in due_intentions():
        if fired >= 2:
            break
        ok, reason = evaluate(it)
        if not ok:
            if reason != "too_fresh":  # too_fresh 是常态，不刷屏
                logger.debug(f"意向 #{it.id} 被评估门拦截: {reason}")
            continue
        # 随机抖动：到点不一定发（防钟表感），下轮还有 50% 机会
        if random.random() < 0.5:
            continue
        try:
            if await _fire(it, gen_model=gen_model, judge_model=judge_model):
                fired += 1
        except Exception as e:
            logger.warning(f"意向 #{it.id} 执行异常: {type(e).__name__}: {e}")
    return fired
