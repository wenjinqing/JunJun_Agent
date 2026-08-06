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
# 注意：pattern 必须写成完整词组或 (?:a|b) 交替——曾经写成 [a|b] 字符组
# （2026-08-06 实锤），单字就命中：「说说发生了什么」被当成发说说声称、
# 「画出这道题」被当成画图声称，误拦满天飞。
_CLAIM_RULES: List[Dict] = [
    {
        "name": "media_drawn",
        # 完成与进行中声称都要证据：「在画了」意味着真有画图任务在跑（异步队列）
        "patterns": [r"画好[了啦]", r"画完[了啦]", r"画出来", r"在画了", r"正在画",
                     r"等(?:一下|一会儿|会儿|下)?就发"],
        "tools": {"ai_draw"},
    },
    {
        "name": "voice_sent",
        "patterns": [r"语音发好", r"语音发给", r"语音发过", r"语音已经发",
                     r"唱好[了啦]", r"唱完[了啦]", r"唱了[一首一段]"],
        "tools": {"unified_tts", "ja_tts"},
    },
    {
        "name": "feed_sent",
        "patterns": [r"说说发好", r"说说发了", r"说说已发", r"说说已经发",
                     r"空间发好", r"空间发了", r"空间已发", r"空间已经发"],
        "tools": {"send_feed"},
    },
    {
        "name": "message_sent",
        "patterns": [r"消息发好", r"消息发过", r"消息发给", r"已经发了", r"已经发过去"],
        "tools": {"send_message"},
    },
    {
        "name": "reminder_set",
        "patterns": [r"提醒设好", r"提醒定好", r"提醒已经设", r"提醒设置成功",
                     r"闹钟设好", r"闹钟定好"],
        "tools": {"set_reminder"},
    },
    {
        "name": "subscribed",
        "patterns": [r"订阅好[了啦]", r"订阅成功", r"已经盯[上着]", r"关注好[了啦]",
                     r"关注成功"],
        "tools": {"subscribe_updates"},
    },
    {
        "name": "unsubscribed",
        "patterns": [r"取消好[了啦]", r"取消成功", r"已经取消[了啦]?订阅"],
        "tools": {"unsubscribe"},
    },
    {
        "name": "music_played",
        "patterns": [r"歌放好[了啦]", r"歌开始放", r"歌已经放", r"音乐放好[了啦]",
                     r"音乐开始放"],
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
            m = re.search(pat, text)
            if m:
                if not recent.intersection(rule["tools"]):
                    issues.append(
                        f"声称「{m.group(0)}」但未调用 {'/'.join(rule['tools'])}"
                    )
                break  # 同 rule 多 pattern 只报一次

    if not issues:
        return True, text, []

    # 拦截：坦白没办成。引用模型实际说出口的短语（m.group(0)），别把 regex
    # 源码糊用户脸上（2026-08-06 实锤：用户收到『画[好了|完了|出来]』一脸懵）。
    # 也不许承诺「我重新来」——守卫自己不重试，空头承诺恰是它在防的不诚实。
    claims = [i.split("」")[0].replace("声称「", "") for i in issues]
    correction = (
        "（刚才那句当我没说：我说『{}』，但其实还没真的去办，"
        "不能骗你。想要的话再喊我一次，这次老老实实办。）".format("；".join(claims))
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
