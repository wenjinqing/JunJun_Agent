"""HonestyGuard：代码层诚实校验（架构重写 Phase 3）。

把 persona 里「没调工具不许说发了」这类诚实规则从 prompt 搬到代码层：
- agent 执行阶段记录本轮真实调用的工具；
- 发送前扫描回复文本，若发现「已发送/已画好/已设置」等行为声称但缺少对应
  工具成功记录，则拦截并替换为诚实说明。

设计要点：
- 只拦截【明确的行为完成声称】，不拦截口语化暧昧表达；
- 工具记录按决策轮次隔离，不跨轮翻旧账；
- 校验失败不沉默，而是坦白「我刚才那句话没兑现，不能发」。
"""

import re
import time
from typing import Dict, List, Set, Tuple

from junjun_core.observability import get_logger

logger = get_logger("honesty_guard")


# 声称完成 ↔ 必需工具名（任一匹配即视为有证据）
_CLAIM_RULES: List[Dict] = [
    {
        "name": "media_drawn",
        "patterns": [r"画[好了|完了|出来]", r"正在画", r"等[一下|会儿]?就发", r"画完[了]?"],
        "tools": {"ai_draw"},
    },
    {
        "name": "voice_sent",
        "patterns": [r"语音[发好了|发给你|发过去]", r"唱[好了|完了]", r"唱了[一首|一段]"],
        "tools": {"unified_tts", "ja_tts"},
    },
    {
        "name": "feed_sent",
        "patterns": [r"说说[发好了|已发|发了]", r"空间[发好了|已发|发了]"],
        "tools": {"send_feed"},
    },
    {
        "name": "message_sent",
        "patterns": [r"消息[发好了|发过去|发给你]", r"已经发[了|过去]"],
        "tools": {"send_message"},
    },
    {
        "name": "reminder_set",
        "patterns": [r"提醒[设好了|定好了|设置成功]", r"闹钟[设好了|定好了]"],
        "tools": {"set_reminder"},
    },
    {
        "name": "subscribed",
        "patterns": [r"订阅[好了|成功]", r"已经盯[上|着]", r"关注[好了|成功]"],
        "tools": {"subscribe_updates"},
    },
    {
        "name": "unsubscribed",
        "patterns": [r"取消[好了|成功]", r"已经取消"],
        "tools": {"unsubscribe"},
    },
    {
        "name": "music_played",
        "patterns": [r"歌[放好了|开始放了|已经放]", r"音乐[放好了|开始放了]"],
        "tools": {"play_music"},
    },
]


def start_decision(session) -> None:
    """标记新一轮决策开始，工具记录从此时间点起算。"""
    session._hg_decision_ts = time.time()
    # 清理过旧的工具记录（保留本轮及上一轮，防止内存无限增长）
    cutoff = session._hg_decision_ts - 3600
    log = getattr(session, "_tool_log", [])
    session._tool_log = [c for c in log if c.get("ts", 0) > cutoff]


def record_tool_call(session, tool_name: str, *, args: dict = None, result: str = "") -> None:
    """记录一次工具调用。由 agent 工具循环调用。"""
    if not hasattr(session, "_tool_log"):
        session._tool_log = []
    session._tool_log.append({
        "name": tool_name,
        "args": args or {},
        "result": (result or "")[:500],
        "ts": time.time(),
    })


def _recent_tool_names(session) -> Set[str]:
    """本轮决策以来调用的工具名集合。"""
    start_ts = getattr(session, "_hg_decision_ts", 0)
    return {c["name"] for c in getattr(session, "_tool_log", []) if c.get("ts", 0) >= start_ts}


def verify(session, text: str) -> Tuple[bool, str, List[str]]:
    """发送前诚实校验。

    返回 (is_honest, text_or_correction, issues)。
    is_honest=True 时 text_or_correction 等于原文；False 时返回修正后的诚实说明。
    """
    if not text:
        return True, text, []
    recent = _recent_tool_names(session)
    issues: List[str] = []
    for rule in _CLAIM_RULES:
        for pat in rule["patterns"]:
            if re.search(pat, text):
                if not recent.intersection(rule["tools"]):
                    issues.append(
                        f"声称「{pat.strip('/')}」但未调用 {'/'.join(rule['tools'])}"
                    )
                break  # 同 rule 多 pattern 只报一次

    if not issues:
        return True, text, []

    # 拦截：坦白没办成。避免模型继续编细节，用固定模板。
    correction = (
        "（系统拦住我了：我刚才说『{}』，但其实还没真的调工具办成，"
        "不能骗你。我重新来。）".format("；".join(i.split("但未调用")[0].replace("声称「", "").replace("」", "")
                                       for i in issues))
    )
    logger.warning(f"[{getattr(session, 'chat_id', '?')}] HonestyGuard 拦截: {issues}")
    return False, correction, issues


def enabled() -> bool:
    """开关：默认关闭，通过 [honesty_guard] enable=true 开启。"""
    try:
        from junjun_core.config import get_global_config
        return bool(get_global_config().raw.get("honesty_guard", {}).get("enable", False))
    except Exception:
        return False
