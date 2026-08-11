"""weekly_report 插件：每周群吃瓜周报 + 颁奖（人审后发群）。

一条消息两节：
- 吃瓜回顾：hippo 话题沉淀（archived 摘要 + 在缓存的热话题）交给 thinker，
  君君口吻复盘本周名场面
- 本周颁奖：Messages 表确定性统计（话痨王/夜猫子/表情帝/大文豪），
  同一次调用里让模型写颁奖词（损一点但不下狠手）

人审复用 TaskKernel/daily_report 同一套「发/算了」（各自的 pending 互不抢单），
超时默认不发。不图化：素材全在库里，崩溃丢了重触发即可重生成（反例纪律）。

状态（DATA_DIR 下，测试可 monkeypatch）：
- last_run.json  {chat_id: "2026-W32"} 按 ISO 周去重（防同一分钟重复触发）

配置 [weekly_report]：enable / day="sun" / time="20:00" / groups=[]（空=白名单群）
/ min_messages=100（冷场周不出报）/ approval_timeout_seconds=600
"""

import asyncio
import json
import time
from pathlib import Path

from junjun_agent.commands import register_command
from junjun_agent.loop.scheduler import ScheduledTask, scheduler
from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.weekly_report")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "weekly_report"

_DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("weekly_report", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------- 素材采集

def _week_stats(chat_id: str, since: float) -> dict:
    """Messages 表一周统计：总条数 + 四个奖。全确定性 SQL，0 token。"""
    from junjun_core.database.models import Messages
    rows = Messages.select(
        Messages.user_id, Messages.user_nickname, Messages.time,
        Messages.is_emoji, Messages.is_picid, Messages.processed_plain_text,
    ).where((Messages.chat_id == chat_id)
            & (Messages.time >= since)
            & (Messages.is_bot == False))  # noqa: E712
    per_user: dict = {}
    total = 0
    for r in rows:
        if not r.user_id:
            continue
        total += 1
        u = per_user.setdefault(r.user_id, {
            "nick": r.user_nickname or r.user_id,
            "count": 0, "night": 0, "emoji": 0, "chars": 0})
        u["nick"] = r.user_nickname or u["nick"]
        u["count"] += 1
        if time.localtime(r.time).tm_hour < 6:
            u["night"] += 1
        u["emoji"] += 1 if (r.is_emoji or r.is_picid) else 0
        u["chars"] += len(r.processed_plain_text or "")
    if not per_user:
        return {"total": 0, "active_users": 0, "awards": []}
    users = list(per_user.values())

    def _top(key):
        return max(users, key=lambda u: u[key])

    awards = []
    if len(users) >= 2:  # 只有一个人说话的群发奖很尴尬
        t = _top("count")
        awards.append(("话痨王", t["nick"], f"{t['count']} 条"))
        t = _top("night")
        if t["night"] > 0:
            awards.append(("夜猫子", t["nick"], f"凌晨发言 {t['night']} 条"))
        t = _top("emoji")
        if t["emoji"] > 0:
            awards.append(("表情帝", t["nick"], f"甩图 {t['emoji']} 次"))
        t = _top("chars")
        awards.append(("大文豪", t["nick"], f"输出 {t['chars']} 字"))
    return {"total": total, "active_users": len(users), "awards": awards}


def _week_summaries(chat_id: str, since: float) -> list:
    """hippo 沉淀：本周归档摘要 + 还在缓存里的热话题。"""
    from junjun_memory.summarizer import HIPPO_DIR
    p = HIPPO_DIR / f"{chat_id.replace(':', '_')}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return []
    out = [str(a.get("summary") or "") for a in data.get("archived", [])
           if float(a.get("time", 0) or 0) >= since and a.get("summary")]
    for t in data.get("topics", []):
        line = f"{t.get('title', '')}: {t.get('content', '')}".strip(": ")
        if line:
            out.append(line)
    return out[-30:]  # 封顶防 prompt 爆炸


# ---------------------------------------------------------------- 成稿

_PROMPT = """{personality}

这是你们群过去一周的素材，你要写一条「一周吃瓜周报」发到群里。{style}

【本周话题沉淀】
{summaries}

【本周数据】共 {total} 条消息，{active_users} 个人冒过泡
{awards}

要求：
- 先聊本周名场面/热点话题：挑 2-4 个有得聊的，带细节带你的吐槽，像跟朋友复盘，
  不是念稿；没有值得一提的就写一两句「这周好安静」之类的感慨
- 再写「本周颁奖」：每个奖给一句颁奖词，损一点但不下狠手，领奖人直呼其名
- 全文 ≤400 字，口语短句，不要播音腔、不要「大家好」「本周总结如下」
- 只输出周报正文，不要任何前后缀"""


async def _write_report(chat_id: str, stats: dict, summaries: list, *,
                        model=None) -> str:
    from junjun_skills.plugins.junzone.tools import _persona
    personality, style = _persona()
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("thinker")  # 周报：低频高价值，开思考
    awards_text = "\n".join(f"- {name}：{nick}（{desc}）"
                            for name, nick, desc in stats["awards"]) or "（本周无奖项）"
    from langchain_core.messages import HumanMessage
    resp = await model.ainvoke([HumanMessage(content=_PROMPT.format(
        personality=personality, style=style,
        summaries="\n".join(f"- {s}" for s in summaries) or "（无）",
        total=stats["total"], active_users=stats["active_users"],
        awards=awards_text))])
    return str(resp.content).strip()


# ---------------------------------------------------------------- 人审（发/算了）

_pending: dict = {}   # key -> {chat_id, text, timeout_task}
_APPROVE_WORDS = {"发": True, "算了": False}


async def _notify_admin(key: str, chat_id: str, text: str) -> None:
    from junjun_core.security import notify_admin
    gid = chat_id.split(":")[1] if ":" in chat_id else chat_id
    try:
        ok = await notify_admin(
            f"【群周报审批】目标群 {gid}\n\n{text}\n\n"
            f"回「发」发到群里，回「算了」本周不发。10 分钟没回默认不发。")
        if not ok:
            logger.warning(f"周报审批通知未送达（未配置 ADMIN_QQ？），超时将默认不发: {key}")
    except Exception as e:
        logger.warning(f"周报审批通知管理员失败: {type(e).__name__}: {e}")


def _arm_timeout(key: str) -> None:
    timeout = float(_cfg().get("approval_timeout_seconds", 600))

    async def _watch():
        await asyncio.sleep(timeout)
        if key in _pending:
            logger.info(f"周报审批超时（{timeout:.0f}s 无回复），默认不发: {key}")
            await approve(key, False)

    _pending[key]["timeout_task"] = asyncio.create_task(_watch())


async def approve(key: str, ok: bool) -> None:
    pend = _pending.pop(key, None)
    if not pend:
        return
    if pend.get("timeout_task"):
        pend["timeout_task"].cancel()
    if not ok:
        logger.info(f"周报被驳回/超时: {key}")
        return
    try:
        from junjun_agent.outbound import send_proactive
        from junjun_core.contracts import ReplySegment
        await send_proactive(pend["chat_id"],
                             [ReplySegment(type="text", data=pend["text"])],
                             source="weekly_report")
        logger.info(f"周报已发群: {key}")
    except Exception as e:
        logger.warning(f"周报发送失败 {key}: {type(e).__name__}: {e}")


async def approval_hook(session, meta) -> bool:
    """True=已消费。管理员本人 + 精确审批词 + 有待审批周报才拦截。"""
    from junjun_core.security import is_admin
    if not is_admin(meta.user_id):
        return False
    decision = _APPROVE_WORDS.get((meta.text or "").strip())
    if decision is None or not _pending:
        return False
    key = next(iter(_pending))  # FIFO
    asyncio.create_task(approve(key, decision))
    ack = "好，这就发到群里。" if decision else "行，本周这条不发了。"
    try:
        from junjun_agent.outbound import send_proactive
        from junjun_core.contracts import ReplySegment
        await send_proactive(session.chat_id, [ReplySegment(type="text", data=ack)],
                             source="weekly_report", remember=False)
    except Exception:
        pass
    logger.info(f"管理员审批 {'放行' if decision else '丢弃'}周报: {key}")
    return True


# ---------------------------------------------------------------- 主流程 + 调度

def _target_chats() -> list:
    cfg = _cfg()
    groups = cfg.get("groups") or []
    if not groups:
        try:
            groups = get_global_config().raw.get("chat", {}).get("group_list") or []
        except Exception:
            groups = []
    return [f"qq:{g}:group" for g in groups]


def _read_state() -> dict:
    try:
        p = DATA_DIR / "last_run.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def _write_state(d: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "last_run.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logger.warning(f"周报状态落盘失败: {e}")


async def run_for_chat(chat_id: str, *, model=None) -> str:
    """生成一群的周报并挂审批。返回结果描述（命令回执/日志用）。"""
    since = time.time() - 7 * 86400
    stats = _week_stats(chat_id, since)
    if stats["total"] < int(_cfg().get("min_messages", 100)):
        return f"本周才 {stats['total']} 条消息，太冷清了，不出周报。"
    summaries = _week_summaries(chat_id, since)
    try:
        text = await _write_report(chat_id, stats, summaries, model=model)
    except Exception as e:
        logger.warning(f"周报成稿失败 [{chat_id}]: {type(e).__name__}: {e}")
        return "周报写稿失败了，下周再试。"
    if not text:
        return "周报写稿返回空，下周再试。"
    key = f"{chat_id}:{time.strftime('%G-W%V')}"
    _pending[key] = {"chat_id": chat_id, "text": text}
    await _notify_admin(key, chat_id, text)
    _arm_timeout(key)
    return "周报写好啦，已发给管理员审批，放行就发群。"


async def weekly_report_tick() -> None:
    """每分钟检查：到点（配置的星期+时刻）且本周没跑过 -> 逐群生成。"""
    cfg = _cfg()
    if not bool(cfg.get("enable", False)):
        return
    from datetime import datetime
    now = datetime.now()
    if now.weekday() != _DAYS.get(str(cfg.get("day", "sun")).lower(), 6):
        return
    if now.strftime("%H:%M") != str(cfg.get("time", "20:00")):
        return
    week = now.strftime("%G-W%V")
    state = _read_state()
    for chat_id in _target_chats():
        if state.get(chat_id) == week:
            continue
        state[chat_id] = week
        _write_state(state)
        logger.info(f"每周群报开跑: {chat_id} {week}")
        try:
            await run_for_chat(chat_id)
        except Exception as e:
            logger.warning(f"每周群报失败 [{chat_id}]: {type(e).__name__}: {e}")


@register_command("weekly_report", aliases=["周报"], plugin="weekly_report",
                  description="立即生成本周群吃瓜周报（管理员，走审批）")
async def weekly_report_cmd(ctx) -> str:
    from junjun_core.security import is_admin, current_user_id
    if not is_admin(current_user_id.get()):
        return "周报只有管理员能手动催更。"
    week = time.strftime("%G-W%V")
    state = _read_state()
    if state.get(ctx.session.chat_id) == week:
        return "本周的周报已经跑过了，下周再来。"
    state[ctx.session.chat_id] = week
    _write_state(state)
    return await run_for_chat(ctx.session.chat_id)


TOOLS = []

scheduler.add(ScheduledTask("weekly_report", weekly_report_tick, interval=60,
                            plugin="weekly_report"))
