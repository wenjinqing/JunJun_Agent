"""工具健康度：跨时间的工具故障聚合认知（P5-4，2026-08-02）。

单次失败已有 P0-13 结构化反喂；本模块解决「接口被风控三天，她每次都当场
撞墙当场道歉，下次还主动提」——把连续失败聚合成降级状态注入上下文，
让 Agent 有「我这个工具最近在修」的持续自我意识；成功自动恢复。

存储：data/tool_health.json（原子写，重启不丢；DB 迁移都省了）。
降级规则：连续失败 >=3 次且最近一次失败在 24h 内；成功即清零。
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List

from junjun_core.observability import get_logger

logger = get_logger("skills.health")

_DEGRADE_THRESHOLD = 3          # 连续失败几次降级
_DEGRADE_TTL = 24 * 3600.0      # 最近失败超过该时长视为「已过去」，不再宣称故障
_STATE: Dict[str, dict] = {}    # tool -> {consec_fails,last_kind,last_error,last_fail_at,degraded_since}
_loaded = False

_STATE_PATH = Path(os.environ.get(
    "TOOL_HEALTH_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "tool_health.json")))


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        if _STATE_PATH.exists():
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _STATE.update(data)
    except Exception as e:
        logger.warning(f"工具健康状态读取失败（忽略）: {e}")


def _save() -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_STATE, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_STATE_PATH)
    except Exception as e:
        logger.debug(f"工具健康状态写入失败（忽略）: {e}")


def record_fail(tool: str, kind: str, error: str = "") -> None:
    """工具硬失败（逃逸异常）时记录；达到阈值标降级。"""
    _load()
    st = _STATE.get(tool) or {}
    fails = int(st.get("consec_fails", 0)) + 1
    st = {"consec_fails": fails, "last_kind": kind,
          "last_error": (error or "")[:150], "last_fail_at": time.time()}
    if fails >= _DEGRADE_THRESHOLD and "degraded_since" not in st:
        st["degraded_since"] = time.time()
        logger.warning(f"工具 {tool} 连续失败 {fails} 次，标记降级[{kind}]")
    _STATE[tool] = st
    _save()


def record_ok(tool: str) -> None:
    """成功即清零（自动恢复）。"""
    _load()
    if tool in _STATE:
        if _STATE[tool].get("consec_fails", 0) >= _DEGRADE_THRESHOLD:
            logger.info(f"工具 {tool} 恢复健康")
        _STATE.pop(tool, None)
        _save()


def degraded_tools() -> List[dict]:
    """当前处于降级状态的工具列表（含故障信息）；过期的自动清理。"""
    _load()
    now = time.time()
    out, expired = [], []
    for tool, st in _STATE.items():
        if st.get("consec_fails", 0) < _DEGRADE_THRESHOLD:
            continue
        if now - st.get("last_fail_at", 0) > _DEGRADE_TTL:
            expired.append(tool)  # 失败已是 24h 前的事，不再宣称故障
            continue
        out.append({"tool": tool, "kind": st.get("last_kind", "未知"),
                    "fails": st["consec_fails"],
                    "since": st.get("degraded_since", st.get("last_fail_at", now)),
                    "error": st.get("last_error", "")})
    for tool in expired:
        _STATE.pop(tool, None)
    if expired:
        _save()
    return out


def _tool_purpose(tool: str) -> str:
    """工具的中文用途（取 description 首行；注册表未加载时退化用名字）。"""
    try:
        from junjun_skills import registry
        skill = registry._registry.get(tool)
        if skill and skill.description:
            first = skill.description.strip().splitlines()[0]
            return first.split("。")[0][:30]
    except Exception:
        pass
    return tool


def health_block() -> str:
    """上下文注入块：降级工具清单 + 行为指引；无降级返回 ""。"""
    degraded = degraded_tools()
    if not degraded:
        return ""
    items = []
    for d in degraded:
        since = time.strftime("%m-%d %H:%M", time.localtime(d["since"]))
        items.append(f"{_tool_purpose(d['tool'])}（{d['tool']}，{d['kind']}类故障，"
                     f"从 {since} 起连续失败 {d['fails']} 次）")
    return ("系统状态：以下功能最近持续故障——" + "；".join(items) +
            "。在恢复前：不要主动提议使用这些功能；被问起时诚实说明"
            "「最近出问题了在修」，并给出可行的替代方案或让对方稍后再试。")


def _reset_for_test() -> None:
    """仅供测试。"""
    global _loaded
    _STATE.clear()
    _loaded = False
