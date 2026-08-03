"""坏 case 规则检测（P8-1）：Langfuse trace -> 可规则判定的坏 case 类别。

只做规则可判的（主观判定留人工标注）：
- percept_denial：输出说了「看不到/没收到」但输入明明有图/语音/在途感知
  （「我看不到图片」事故类，P4 已修，回归防线）
- tool_error_storm：单条 trace 里 [TOOL_ERROR ≥3 次（参数校验连错烧穿迭代类）
- intent_missed:<tool>：输入含强意图词但 trace 里对应工具从未出现
  （口头答应没调用类，意图自检的漏网之鱼）

检测器与 scripts/export_bad_traces.py、tests/regression_corpus 三方共用——
语料回放保证检测器本身不退化。
"""

import json

_DENIAL_WORDS = ("看不到", "没收到", "没发吧", "看不见")
_PERCEPT_MARKS = ("[图片]", "[语音]", "[视频]", "还在看", "image", "voice")


def _text(v) -> str:
    """trace 的 input/output 字段（dict/list/str 皆可）-> 纯文本。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def detect_bad_case(trace: dict) -> list:
    """单条 trace（Langfuse REST API 返回的 dict）-> 命中的规则列表（可空）。"""
    kinds = []
    input_text = _text(trace.get("input"))
    output_text = _text(trace.get("output"))
    blob = input_text + output_text

    # R1 感知否认：输出否认看到 + 输入有感知标记
    if (any(kw in output_text for kw in _DENIAL_WORDS)
            and any(mk in input_text for mk in _PERCEPT_MARKS)):
        kinds.append("percept_denial")

    # R2 工具错误风暴
    if blob.count("[TOOL_ERROR") >= 3:
        kinds.append("tool_error_storm")

    # R3 意图漏调（元数据单一数据源：注册表 INTENT 层）
    try:
        from junjun_skills.registry import intent_groups
        for keywords, _tools, primary in intent_groups():
            if primary and any(kw in input_text for kw in keywords) \
                    and primary not in blob:
                kinds.append(f"intent_missed:{primary}")
    except Exception:
        pass
    return kinds
