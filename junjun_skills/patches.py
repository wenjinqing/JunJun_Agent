"""技能补丁（P8-2 经验回放，Memento 思路：基座冻结，补丁外挂）。

闭环：工具失败 -> JSONL 日志（registry 错误包装处挂钩）-> 每周 agent 槽
复盘 -> 补丁候选（candidate）-> 管理员启用（人工审查 = 门控一环）->
注入对应工具 description 后缀（【经验补丁 vN】…）-> pytest 回放注入链路。
回归即 /补丁 回滚；单工具补丁超上限时 agent 槽合并（防膨胀）。

[evolution] enable=false 时只记日志不复盘（默认关，灰度同意向系统）。
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("skills.patches")

_DIR = Path(__file__).resolve().parent.parent / "data"
_LOG_PATH = _DIR / "tool_failures.jsonl"
_STATE_PATH = _DIR / "patches_state.json"

_REVIEW_PROMPT = """你在复盘一个 QQ 机器人最近的工具失败记录，目标是写「经验补丁」——
附加在工具说明末尾的一句话提醒，让未来的调用避开同样的坑。

【工具 {tool} 的当前说明】
{description}

【最近 {days} 天的失败记录（{count} 次）】
{failures}

要求：
- 只写从记录里能确定的教训（如「调用前要先拿作者 UID 再查」「该接口上午常限流，失败建议对方稍后再试」）
- 补丁一句话（40 字以内），写给未来的调用者，不复述错误本身
- 失败若只是网络抖动/偶发，提炼不出教训，输出空补丁
输出 JSON：{{"patch": "...", "lesson": "依据的失败模式一句话"}}，不要别的文字。"""

_MERGE_PROMPT = """以下是一个工具的 {n} 条经验补丁，内容可能有重复。合并成一条
（保留全部有效教训，40 字以内），只输出合并后的文本：
{patches}"""


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("evolution", {}) or {}
    except Exception:
        return {}


def _enabled() -> bool:
    return bool(_cfg().get("enable", False))


# ---------------------------------------------------------------- 失败日志

def log_failure(tool: str, kind: str, error: str = "") -> None:
    """追加一条工具失败记录（registry 错误包装处调用）。永不抛异常。"""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "tool": tool, "kind": kind,
                                "error": (error or "")[:200]}, ensure_ascii=False) + "\n")
        # 防膨胀：超 5000 行截断留后 2000 行
        if _LOG_PATH.stat().st_size > 1024 * 1024:
            lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()[-2000:]
            _LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        logger.debug(f"失败日志写入失败（忽略）: {e}")


def aggregate_failures(days: int = 7) -> Dict[str, List[dict]]:
    """近 N 天失败按工具聚合（时间升序）。"""
    since = time.time() - days * 86400
    out: Dict[str, List[dict]] = {}
    try:
        if not _LOG_PATH.exists():
            return out
        for line in _LOG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("ts", 0) >= since and rec.get("tool"):
                out.setdefault(rec["tool"], []).append(rec)
    except Exception as e:
        logger.debug(f"失败日志读取失败: {e}")
    return out


# ---------------------------------------------------------------- 补丁 CRUD + 注入

def _active_patches() -> Dict[str, list]:
    from junjun_core.database.models import SkillPatch, _bot_id
    out: Dict[str, list] = {}
    try:
        rows = (SkillPatch.select()
                .where((SkillPatch.bot_id == _bot_id())
                       & (SkillPatch.status == "active"))
                .order_by(SkillPatch.version.asc()))
        for r in rows:
            out.setdefault(r.tool, []).append(r)
    except Exception:
        pass
    return out


def apply_to_registry(tool_name: str = "") -> None:
    """把活跃补丁注入工具 description（幂等：从原始描述重算）。永不抛异常。"""
    try:
        from junjun_skills import registry
        active = _active_patches()
        names = [tool_name] if tool_name else list(registry._registry.keys())
        for name in names:
            skill = registry._registry.get(name)
            if skill is None:
                continue
            try:
                meta = dict(skill.metadata or {})
                if "_orig_desc" not in meta:
                    meta["_orig_desc"] = skill.description or ""
                    skill.metadata = meta
                suffix = "".join(f"\n【经验补丁 v{p.version}】{p.patch}"
                                 for p in active.get(name, []))
                skill.description = meta["_orig_desc"] + suffix
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"补丁注入失败（忽略）: {e}")


def activate(patch_id: int) -> str:
    """启用候选补丁（管理员门控）。无 source_case 不许激活。"""
    from junjun_core.database.models import SkillPatch
    row = SkillPatch.get_or_none(SkillPatch.id == patch_id)
    if row is None:
        return f"补丁 #{patch_id} 不存在。"
    if row.status == "active":
        return f"补丁 #{patch_id} 已经是启用状态。"
    if not (row.source_case or "").strip():
        return f"补丁 #{patch_id} 缺少失败依据（source_case），按规矩不能启用。"
    row.status, row.updated_at = "active", time.time()
    row.save()
    apply_to_registry(row.tool)
    logger.info(f"技能补丁 #{patch_id}（{row.tool}）已启用")
    return f"补丁 #{patch_id} 已启用并注入 {row.tool} 的工具说明。"


def rollback(patch_id: int) -> str:
    """回滚（回归即回滚）。"""
    from junjun_core.database.models import SkillPatch
    row = SkillPatch.get_or_none(SkillPatch.id == patch_id)
    if row is None:
        return f"补丁 #{patch_id} 不存在。"
    row.status, row.updated_at = "rolled_back", time.time()
    row.save()
    apply_to_registry(row.tool)
    logger.info(f"技能补丁 #{patch_id}（{row.tool}）已回滚")
    return f"补丁 #{patch_id} 已回滚，{row.tool} 的说明已还原。"


def list_patches() -> List[dict]:
    from junjun_core.database.models import SkillPatch, _bot_id
    try:
        rows = (SkillPatch.select()
                .where(SkillPatch.bot_id == _bot_id())
                .order_by(SkillPatch.updated_at.desc()).limit(30))
        return [{"id": r.id, "tool": r.tool, "patch": r.patch, "status": r.status,
                 "version": r.version, "source_case": r.source_case} for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------- 复盘

def _similar(a: str, b: str) -> bool:
    return a in b or b in a


def _existing_patch_texts(tool: str) -> List[str]:
    from junjun_core.database.models import SkillPatch, _bot_id
    return [r.patch for r in SkillPatch.select().where(
        (SkillPatch.bot_id == _bot_id()) & (SkillPatch.tool == tool)
        & (SkillPatch.status.in_(["candidate", "active"])))]


def _tool_description(tool: str) -> str:
    try:
        from junjun_skills import registry
        skill = registry._registry.get(tool)
        if skill:
            meta = skill.metadata or {}
            return meta.get("_orig_desc") or (skill.description or "")
    except Exception:
        pass
    return ""


async def _review_one(tool: str, records: List[dict], *, model, days: int) -> Optional[dict]:
    """单工具复盘 -> 补丁候选 dict 或 None。"""
    from langchain_core.messages import HumanMessage
    failures_text = "\n".join(
        f"- [{time.strftime('%m-%d %H:%M', time.localtime(r['ts']))}] "
        f"{r.get('kind', '?')}：{r.get('error', '')[:80]}" for r in records[-10:])
    resp = await model.ainvoke([HumanMessage(content=_REVIEW_PROMPT.format(
        tool=tool, description=_tool_description(tool)[:500] or "（无说明）",
        days=days, count=len(records), failures=failures_text))])
    import re
    m = re.search(r"\{.*\}", str(resp.content), re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    patch = str(data.get("patch") or "").strip()[:80]
    lesson = str(data.get("lesson") or "").strip()[:120]
    if not patch:
        return None
    return {"tool": tool, "patch": patch,
            "source_case": lesson or f"{len(records)} 次失败：{records[-1].get('kind', '?')}"}


async def _merge_overflow(tool: str, *, model) -> None:
    """单工具活跃补丁超上限：agent 槽合并成一条（防膨胀），旧标 merged。"""
    from junjun_core.database.models import SkillPatch
    max_per = int(_cfg().get("max_patches_per_tool", 3))
    active = _active_patches().get(tool, [])
    if len(active) <= max_per:
        return
    from langchain_core.messages import HumanMessage
    try:
        resp = await model.ainvoke([HumanMessage(content=_MERGE_PROMPT.format(
            n=len(active),
            patches="\n".join(f"{i + 1}. {p.patch}" for i, p in enumerate(active))))])
        merged = str(resp.content).strip()[:80]
    except Exception:
        merged = ""
    now = time.time()
    if merged:
        max_ver = max(p.version for p in active)
        SkillPatch.create(tool=tool, patch=merged,
                          source_case=f"由 {len(active)} 条补丁合并",
                          version=max_ver + 1, status="active",
                          created_at=now, updated_at=now)
    keep = active[-max_per:] if not merged else []
    for p in active:
        if p in keep:
            continue
        p.status, p.updated_at = "merged", now
        p.save()
    apply_to_registry(tool)
    logger.info(f"工具 {tool} 的 {len(active)} 条补丁已合并清理")


async def review(*, model=None, days: int = 7) -> int:
    """复盘一轮：聚合失败 -> 逐工具蒸馏补丁候选。返回新增候选数。"""
    min_fails = int(_cfg().get("min_failures", 3))
    agg = aggregate_failures(days)
    targets = {t: rs for t, rs in agg.items() if len(rs) >= min_fails}
    if not targets:
        return 0
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("agent")
    from junjun_core.database.models import SkillPatch
    now = time.time()
    added = 0
    for tool, records in sorted(targets.items(), key=lambda kv: -len(kv[1]))[:5]:
        try:
            cand = await _review_one(tool, records, model=model, days=days)
        except Exception as e:
            logger.debug(f"复盘 {tool} 失败（跳过）: {type(e).__name__}: {e}")
            continue
        if not cand:
            continue
        if any(_similar(cand["patch"], old) for old in _existing_patch_texts(tool)):
            continue  # 已有相似候选/活跃补丁，不重复造
        SkillPatch.create(tool=tool, patch=cand["patch"],
                          source_case=cand["source_case"], version=1,
                          status="candidate", created_at=now, updated_at=now)
        added += 1
        logger.info(f"技能补丁候选（{tool}）: {cand['patch'][:40]}")
        try:
            await _merge_overflow(tool, model=model)
        except Exception:
            pass
    return added


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
    except Exception:
        pass


async def review_tick(*, model=None) -> None:
    """调度入口（每小时一轮，内部按周节流）。enable=false 只记日志不复盘。"""
    if not _enabled():
        return
    interval = float(_cfg().get("review_interval_days", 7)) * 86400
    st = _load_state()
    if time.time() - float(st.get("last_review", 0.0)) < interval:
        return
    n = await review(model=model)
    st["last_review"] = time.time()
    _save_state(st)
    if n:
        logger.info(f"经验复盘完成：新增 {n} 条补丁候选（等管理员启用）")
