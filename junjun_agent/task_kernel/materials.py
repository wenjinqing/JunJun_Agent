"""TaskKernel 步骤材料库：大产出全文落盘，上下文只带摘要+引用指针。

思想来源 DeepSeek Harness 的 PTC（程序化工具调用，2026-08-13 开源）：
「中间数据留在运行环境，只有最终结果进模型上下文」。我们的低成本移植——
不进城不等于丢掉：全文存 data/task_materials/<plan_id>/<step_id>.md，
step.result 只留摘要+材料 id；llm_synthesize/终态汇报按需读回全文
（带单项与总量预算），重规划轮只看摘要（reviser 本就只用 result[:100]）。

落地前的实况（plan.py Step.result 注释早就写了「大产出的全文不落这里」，
但全文从没落过别处）：step.result = result[:500]，全文直接丢弃——
合成步骤拿 500 字残桩写报告，内容饥荒（与 synth 断料同族不同因）。

落盘失败一律降级回旧行为（截断 inline），材料库是增强不是硬依赖。
"""

import re
from pathlib import Path
from typing import Optional

from junjun_core.observability import get_logger

logger = get_logger("task_kernel.materials")

# 默认阈值与预算（[task_kernel] 可覆盖）
_INLINE_CHARS = 500          # 超过则外置（与旧截断长度对齐，短结果行为不变）
_STUB_CHARS = 200            # 外置后留在 step.result 的摘要长度
_SYNTH_PER_CHARS = 4000      # 合成步骤单项材料预算
_SYNTH_TOTAL_CHARS = 12000   # 合成步骤材料总预算
_REPORT_PER_CHARS = 1500     # 终态汇报单项材料预算（QQ 消息不宜过长）


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("task_kernel", {})
    except Exception:
        return {}


def _dir() -> Path:
    return Path(str(_cfg().get("material_dir") or "data/task_materials"))


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_cfg().get(key, default))
    except Exception:
        return default


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:64] or "x"


def store(plan_id: str, step_id: str, text: str) -> str:
    """全文落盘，返回材料 id（"<plan_id>/<step_id>"）；失败返回 ""。"""
    try:
        d = _dir() / _safe(plan_id)
        d.mkdir(parents=True, exist_ok=True)
        (_dir() / _safe(plan_id) / f"{_safe(step_id)}.md").write_text(
            text, encoding="utf-8")
        return f"{_safe(plan_id)}/{_safe(step_id)}"
    except Exception as e:
        logger.warning(f"材料落盘失败（降级 inline 截断）: {type(e).__name__}: {e}")
        return ""


def read(material_id: str, max_chars: int) -> str:
    """按材料 id 读回全文（截到 max_chars）；不存在/失败返回 ""。"""
    if not material_id or "/" not in material_id:
        return ""
    plan_id, step_id = material_id.split("/", 1)
    try:
        p = _dir() / _safe(plan_id) / f"{_safe(step_id)}.md"
        if not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n……（材料截断，全文 {len(text)} 字）"
        return text
    except Exception:
        return ""


def store_result(plan_id: str, step_id: str, result: str) -> tuple:
    """步骤产出入库的决策点。返回 (step.result 摘要, material_id)。

    短产出原样 inline（material_id=""，行为与旧版一致）；长产出外置，
    摘要带指针——「全文 N 字已存材料」让合成/汇报知道有货可取。
    落盘失败降级旧行为：截断 inline、无指针。
    """
    inline = _cfg_int("material_inline_chars", _INLINE_CHARS)
    if len(result) <= inline:
        return result, ""
    mid = store(plan_id, step_id, result)
    if not mid:
        return result[:inline], ""
    stub = result[:_STUB_CHARS]
    return f"{stub}\n……（全文 {len(result)} 字已存材料 {mid}）", mid


def material_text(step, max_chars: int) -> str:
    """取步骤产出的可用全文：有材料读材料，否则 inline 摘要。"""
    mid = getattr(step, "material_id", "")
    if mid:
        text = read(mid, max_chars)
        if text:
            return text
    return getattr(step, "result", "") or ""


def synth_budget() -> tuple:
    """(单项预算, 总量预算)——_synthesize 组装材料用。"""
    return (_cfg_int("synth_material_chars", _SYNTH_PER_CHARS),
            _cfg_int("synth_material_total_chars", _SYNTH_TOTAL_CHARS))


def report_budget() -> int:
    return _cfg_int("report_material_chars", _REPORT_PER_CHARS)
