"""跨场景用户档案（P6-4）：按 user_id 聚合的蒸馏事实，「知道，但分得清场合」。

私聊和群聊是两个 chat_id，没有这层她不知道「和你在群里聊游戏、私聊聊心事」
——真人社交的基本能力是割裂的。本模块把用户在各场景说的话蒸馏成事实条目
（不落原文），注入时按场合过滤：

- 私聊里：可以引用 ta 在群里的公开内容（群聊本来就是半公开的）
- 群聊里：只注入**本群**来源的事实 + 事实级熟度（「你们私聊也很熟」），
  私聊来源的条目**绝不注入**——隐私泄露 = 社死，单向不可逆
- 多群隔离：A 群的事不在 B 群说（source_chat_id 精确匹配）

蒸馏打「来源场景」标签是机械打标（按场景分别蒸馏），不靠 LLM 自觉。
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("memory.scene_profile")

_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "scene_profile_state.json"
_FACT_TTL = 14 * 86400          # 事实 14 天未被再确认则清理（事实会过期）
_PRIVATE_HINT_MIN = 20          # 私聊消息数阈值：群里才说「你们私聊也很熟」

_DISTILL_PROMPT = """这是用户「{nickname}」最近 {days} 天在{scene_desc}里说的话（节选）：
{lines}

任务：蒸馏关于 ta 的「值得长期记住的事实」，最多 {max_facts} 条
（在备考/要出差/喜好/最近状态/在意的东西等）。
铁律：
- 只写能从这些话里确定的事，不猜不编
- 不逐字引用原话，不出现 QQ 号，提到别人用昵称
- 没有值得记的就输出 []
输出 JSON 数组，每条 {{"fact": "一句话"}}，不要任何别的文字。"""


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("scene_profile", {}) or {}
    except Exception:
        return {}


def _person_id(platform: str, user_id: str) -> str:
    return f"{platform}:{user_id}"


# ---------------------------------------------------------------- 状态（每用户蒸馏节流）

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
        logger.debug(f"scene_profile 状态落盘失败（忽略）: {e}")


# ---------------------------------------------------------------- 素材收集

def _user_scenes(platform: str, user_id: str, *, days: int,
                 max_per_scene: int = 40) -> Dict[str, dict]:
    """用户最近 N 天发言按场景分组 -> {chat_id: {scene, lines}}。"""
    from junjun_core.database.models import Messages
    since = time.time() - days * 86400
    try:
        rows = list(Messages.select()
                    .where((Messages.user_id == user_id)
                           & (Messages.is_bot == False)  # noqa: E712
                           & (Messages.time >= since))
                    .order_by(Messages.time.desc()).limit(300))
    except Exception as e:
        logger.debug(f"scene_profile 素材读取失败: {e}")
        return {}
    scenes: Dict[str, dict] = {}
    for r in rows:
        text = (r.processed_plain_text or "").strip()
        if not text:
            continue
        entry = scenes.setdefault(r.chat_id, {
            "scene": "group" if r.chat_id.endswith(":group") else "private",
            "lines": []})
        if len(entry["lines"]) < max_per_scene:
            entry["lines"].append(text[:60])
    return scenes


def _user_nickname(user_id: str) -> str:
    from junjun_core.database.models import Messages
    row = (Messages.select()
           .where((Messages.user_id == user_id) & (Messages.user_nickname != ""))
           .order_by(Messages.time.desc()).first())
    return row.user_nickname if row else user_id


# ---------------------------------------------------------------- 蒸馏

def _parse_facts(text: str) -> List[str]:
    import re
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for e in data if isinstance(data, list) else []:
        raw = e.get("fact") if isinstance(e, dict) else None
        fact = str(raw).strip() if raw else ""
        if fact:
            out.append(fact[:80])
    return out


async def _distill_scene(nickname: str, scene_desc: str, lines: List[str], *,
                         model, days: int, max_facts: int) -> List[str]:
    from langchain_core.messages import HumanMessage
    prompt = _DISTILL_PROMPT.format(
        nickname=nickname, days=days, scene_desc=scene_desc,
        lines="\n".join(f"- {l}" for l in lines), max_facts=max_facts)
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    return _parse_facts(str(resp.content))


def _merge_facts(person_id: str, platform: str, user_id: str,
                 chat_id: str, scene: str, facts: List[str]) -> int:
    """合并一场景的蒸馏结果：已有（含子串）保持并刷新时间，新的插入。
    返回新增数。"""
    from junjun_core.database.models import UserSceneProfile, _bot_id
    now = time.time()
    existing = list(UserSceneProfile.select().where(
        (UserSceneProfile.bot_id == _bot_id())
        & (UserSceneProfile.person_id == person_id)
        & (UserSceneProfile.source_chat_id == chat_id)))
    added = 0
    for fact in facts:
        row = next((r for r in existing
                    if r.content == fact or r.content in fact or fact in r.content), None)
        if row is not None:
            row.updated_at = now
            row.weight = min(1.5, row.weight + 0.1)
            row.save()
        else:
            UserSceneProfile.create(
                person_id=person_id, platform=platform, user_id=user_id,
                content=fact, source_scene=scene, source_chat_id=chat_id,
                weight=1.0, created_at=now, updated_at=now)
            added += 1
    return added


def _gc_stale() -> int:
    """清理过期未确认的事实（14 天没再被蒸馏确认的 = 旧事，不再注入）。"""
    from junjun_core.database.models import UserSceneProfile, _bot_id
    n = (UserSceneProfile.delete()
         .where((UserSceneProfile.bot_id == _bot_id())
                & (UserSceneProfile.updated_at < time.time() - _FACT_TTL))
         .execute())
    if n:
        logger.info(f"scene_profile 清理过期事实 {n} 条")
    return n


async def distill_user(platform: str, user_id: str, *, model=None, days: int = 3) -> int:
    """蒸馏一个用户的跨场景档案。返回新增事实数。"""
    cfg = _cfg()
    scenes = _user_scenes(platform, user_id, days=days)
    if not scenes:
        return 0
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
    nickname = _user_nickname(user_id)
    pid = _person_id(platform, user_id)
    max_facts = int(cfg.get("max_facts", 3))
    added = 0
    for chat_id, entry in scenes.items():
        if len(entry["lines"]) < 3:   # 一个场景没说几句话，蒸不出可靠事实
            continue
        scene_desc = (f"QQ 群（chat_id={chat_id}）" if entry["scene"] == "group"
                      else "和你的私聊")
        try:
            facts = await _distill_scene(nickname, scene_desc, entry["lines"],
                                         model=model, days=days, max_facts=max_facts)
        except Exception as e:
            logger.debug(f"scene_profile 单场景蒸馏失败（跳过）: {type(e).__name__}: {e}")
            continue
        added += _merge_facts(pid, platform, user_id, chat_id, entry["scene"], facts)
    if added:
        logger.info(f"scene_profile {pid}: 新增 {added} 条事实")
    return added


async def profile_tick() -> None:
    """定时 tick：近 24h 活跃（≥min_messages 条）且 12h 未蒸馏的用户，每轮最多 5 个。"""
    cfg = _cfg()
    if not bool(cfg.get("enable", True)):
        return
    from junjun_core.database.models import Messages
    from peewee import fn
    min_msgs = int(cfg.get("min_messages", 10))
    since = time.time() - 86400
    try:
        rows = list(Messages.select(Messages.user_id, fn.COUNT(Messages.id).alias("n"))
                    .where((Messages.is_bot == False)  # noqa: E712
                           & (Messages.user_id != "")
                           & (Messages.time >= since))
                    .group_by(Messages.user_id)
                    .having(fn.COUNT(Messages.id) >= min_msgs)
                    .order_by(fn.COUNT(Messages.id).desc()).limit(20))
    except Exception as e:
        logger.debug(f"scene_profile tick 查询失败: {e}")
        return
    state = _load_state()
    now = time.time()
    done = 0
    for r in rows:
        if done >= 5:
            break
        pid = _person_id("qq", r.user_id)
        if now - float(state.get(pid, 0.0)) < 12 * 3600:
            continue
        try:
            await distill_user("qq", r.user_id)
        except Exception as e:
            logger.debug(f"scene_profile 蒸馏失败 {pid}: {type(e).__name__}: {e}")
        state[pid] = now
        done += 1
    _save_state(state)
    _gc_stale()


# ---------------------------------------------------------------- 注入（隐私边界在这层强制）

def build_scene_block(platform: str, user_id: str, current_chat_id: str,
                      is_group: bool) -> str:
    """按场合过滤的用户跨场景档案块。

    隐私生命线（违反 = 社死级事故）：
    - 群聊：只注入 source_scene=="group" 且 source_chat_id==本群 的事实
      （私聊来源、别群来源绝不出现），外加事实级熟度「你们私聊也很熟」
    - 私聊：可以看到 ta 自己的全部场景事实（都是 ta 自己的言行）
    """
    if not bool(_cfg().get("enable", True)):
        return ""
    from junjun_core.database.models import Messages, UserSceneProfile, _bot_id
    pid = _person_id(platform, user_id)
    try:
        rows = list(UserSceneProfile.select().where(
            (UserSceneProfile.bot_id == _bot_id())
            & (UserSceneProfile.person_id == pid)
            & (UserSceneProfile.updated_at >= time.time() - _FACT_TTL))
            .order_by(UserSceneProfile.weight.desc()))
    except Exception:
        return ""
    parts = []
    if is_group:
        facts = [r.content for r in rows
                 if r.source_scene == "group" and r.source_chat_id == current_chat_id]
        if facts:
            parts.append("关于 ta 你还记得（本群里的事）："
                         + "；".join(facts[:5]))
        # 事实级熟度：只传达「熟」，不传达任何私聊内容
        try:
            private_n = (Messages.select()
                         .where((Messages.user_id == user_id)
                                & (Messages.chat_id.endswith(":private")))
                         .count())
        except Exception:
            private_n = 0
        if private_n >= _PRIVATE_HINT_MIN:
            parts.append("你们私聊也很熟（私下的事绝不外说，这里只表示你们关系近）")
    else:
        group_facts = [r.content for r in rows if r.source_scene == "group"]
        private_facts = [r.content for r in rows if r.source_scene == "private"]
        facts = private_facts + group_facts
        if facts:
            parts.append("关于 ta 你还记得（包括 ta 在群里的公开表现）："
                         + "；".join(facts[:6]))
    return "\n".join(parts)


def forget_user_facts(platform: str, user_id: str, kw: str, *,
                      admin: bool = False, current_chat_id: str = "") -> int:
    """/忘掉 集成：删本人档案里含关键词的事实。非管理员只删「当前会话来源」的。"""
    from junjun_core.database.models import UserSceneProfile, _bot_id
    cond = ((UserSceneProfile.bot_id == _bot_id())
            & (UserSceneProfile.person_id == _person_id(platform, user_id))
            & (UserSceneProfile.content.contains(kw)))
    if not admin:
        cond &= (UserSceneProfile.source_chat_id == current_chat_id)
    return UserSceneProfile.delete().where(cond).execute()
