"""Identity Core 自我模型（P6-3）：从日记蒸馏「我是个什么样的猫娘」。

人设目前完全靠 prompt 静态描述，长跑会漂移；本模块给她第二个锚——
从自己的经历里长出来的自我认知：

- 每周（且期间 ≥3 篇新日记）用 utils 槽把近期日记蒸馏成自我认知条目，
  prompt 写死「只沉淀反复出现的模式」（防一次 emo 固化成永远丧）
- 条目带权重：本轮再次确认 +0.2（上限 1.5），未确认 ×0.85 衰减，
  <0.5 折叠归档（不删，可复盘）
- 注入 persona role 层，话术写死优先级：静态人设是宪法，自我模型是生活
- /重置自我（管理员）兜底：全部归档重来
"""

import json
import re
import time
from pathlib import Path
from typing import List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.identity")

_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "identity_state.json"

_CATEGORIES = ("我喜欢", "我看不惯", "我们的梗", "最近在乎")

_DISTILL_PROMPT = """你是 QQ 机器人「{nickname}」。下面是你最近的私人日记，和你目前已有的自我认知条目。

【最近的日记】
{diaries}

【已有的自我认知】
{existing}

任务：从日记里蒸馏「你是个什么样的猫娘」。铁律：**只沉淀反复出现的模式**
（至少在两篇日记里都体现出来的才算），一次偶然的情绪/事件绝不固化成自我认知。
和已有条目意思相同的不要重复造。
输出 JSON 数组，最多 {max_entries} 条，每条格式：
{{"category": "我喜欢|我看不惯|我们的梗|最近在乎", "content": "一句话，第一人称口语"}}
只输出 JSON 数组，不要任何别的文字。"""


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("identity", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------- 条目读写

def get_entries(*, limit: int = 0) -> list:
    """活跃自我认知条目（weight 降序）。limit=0 走配置 max_entries。"""
    from junjun_core.database.models import SelfIdentity, _bot_id
    if limit <= 0:
        limit = int(_cfg().get("max_entries", 8))
    rows = list(SelfIdentity.select()
                .where((SelfIdentity.bot_id == _bot_id())
                       & (SelfIdentity.archived == False))  # noqa: E712
                .order_by(SelfIdentity.weight.desc()).limit(limit))
    return rows


def build_identity_block() -> str:
    """persona role 层注入块；无条目返回空串。"""
    try:
        rows = get_entries()
    except Exception:
        return ""
    if not rows:
        return ""
    lines = "\n".join(f"- {r.category}：{r.content}" for r in rows)
    return (f"你从这些日子的经历里认识到的自己：\n{lines}\n"
            "（这是你活出来的自我，上面的人设描述是宪法——冲突时听宪法。）")


def reset_identity() -> int:
    """/重置自我：全部活跃条目归档。返回归档数。"""
    from junjun_core.database.models import SelfIdentity, _bot_id
    n = (SelfIdentity.update(archived=True, updated_at=time.time())
         .where((SelfIdentity.bot_id == _bot_id())
                & (SelfIdentity.archived == False))  # noqa: E712
         .execute())
    logger.info(f"自我认知已重置（归档 {n} 条）")
    return n


# ---------------------------------------------------------------- 蒸馏

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
        logger.debug(f"identity 状态落盘失败（忽略）: {e}")


def _recent_diaries(since_ts: float, *, limit: int = 14) -> list:
    from junjun_core.database.models import DiaryEntry, _bot_id
    return list(DiaryEntry.select()
                .where((DiaryEntry.bot_id == _bot_id())
                       & (DiaryEntry.created_at >= since_ts))
                .order_by(DiaryEntry.date.desc()).limit(limit))


def _parse_entries(text: str) -> List[dict]:
    """从 LLM 产出抠 JSON 数组；坏输出返回 []。"""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for e in data if isinstance(data, list) else []:
        if not isinstance(e, dict):
            continue
        cat = str(e.get("category") or "").strip()
        content = str(e.get("content") or "").strip()
        if cat in _CATEGORIES and content:
            out.append({"category": cat, "content": content[:60]})
    return out


def _similar(a: str, b: str) -> bool:
    """粗略同义判重：互为子串即视为同一条（蒸馏输出本身很短）。"""
    return a in b or b in a


def _upsert_entries(new_entries: List[dict]) -> tuple:
    """合并新条目：确认旧条目加权，新条目入库；未确认旧条目衰减/归档。
    返回 (新增数, 确认数, 归档数)。"""
    from junjun_core.database.models import SelfIdentity, _bot_id
    now = time.time()
    active = list(SelfIdentity.select()
                  .where((SelfIdentity.bot_id == _bot_id())
                         & (SelfIdentity.archived == False)))  # noqa: E712
    matched_ids = set()
    added = confirmed = 0
    for e in new_entries:
        row = next((r for r in active
                    if r.category == e["category"]
                    and _similar(r.content, e["content"])), None)
        if row is not None:
            row.seen_count += 1
            row.weight = min(1.5, row.weight + 0.2)
            row.updated_at = now
            row.save()
            matched_ids.add(row.id)
            confirmed += 1
        else:
            SelfIdentity.create(category=e["category"], content=e["content"],
                                weight=1.0, seen_count=1,
                                created_at=now, updated_at=now)
            added += 1
    archived = 0
    for row in active:
        if row.id in matched_ids:
            continue
        row.weight *= 0.85
        row.updated_at = now
        if row.weight < 0.5:
            row.archived = True
            archived += 1
        row.save()
    return added, confirmed, archived


async def distill(*, model=None, force: bool = False, callbacks=None) -> int:
    """蒸馏一轮自我认知。返回新增条目数（0 = 无产出/失败）。"""
    cfg = _cfg()
    if not force and not bool(cfg.get("enable", True)):
        return 0
    state = _load_state()
    last = float(state.get("last_distill", 0.0))
    diaries = _recent_diaries(last or (time.time() - 14 * 86400))
    if len(diaries) < int(cfg.get("min_diaries", 3)) and not force:
        return 0
    try:
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        existing = get_entries(limit=20)
        existing_text = ("\n".join(f"- {r.category}：{r.content}" for r in existing)
                         or "（还没有）")
        diary_text = "\n\n".join(f"[{d.date}] {d.content[:400]}" for d in diaries)
        from langchain_core.messages import HumanMessage
        prompt = _DISTILL_PROMPT.format(
            nickname=get_global_config().bot.nickname,
            diaries=diary_text or "（最近没有日记）",
            existing=existing_text,
            max_entries=int(cfg.get("max_entries", 8)),
        )
        resp = await model.ainvoke([HumanMessage(content=prompt)],
                                   config={"callbacks": callbacks or []})
        entries = _parse_entries(str(resp.content))
        if not entries:
            logger.warning("自我认知蒸馏：LLM 产出无法解析，保留旧条目")
            return 0
        added, confirmed, archived = _upsert_entries(entries)
        _save_state({"last_distill": time.time()})
        logger.info(f"自我认知蒸馏：新增 {added} / 确认 {confirmed} / 归档 {archived}")
        return added
    except Exception as e:
        logger.warning(f"自我认知蒸馏失败: {type(e).__name__}: {e}")
        return 0


async def identity_tick() -> None:
    """定时 tick：到间隔且攒够新日记才蒸馏（scheduler 每小时一轮）。"""
    cfg = _cfg()
    if not bool(cfg.get("enable", True)):
        return
    last = float(_load_state().get("last_distill", 0.0))
    if time.time() - last < float(cfg.get("interval_days", 7)) * 86400:
        return
    await distill()
