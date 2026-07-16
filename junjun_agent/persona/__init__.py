"""persona: system prompt 组装（阶段 2 最简版）。

阶段 3 扩展：情绪/记忆摘要/关系/keyword_reaction 块。
"""

import re
from datetime import datetime

from junjun_core.config import get_global_config

# 去 emoji：原项目实测 system prompt 含 emoji 会干扰 function calling schema
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F900-\U0001F9FF]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def build_system_prompt(*, is_group: bool, nickname: str = "") -> str:
    cfg = get_global_config()
    p = cfg.raw.get("personality", {})
    nickname = nickname or cfg.bot.nickname
    now = datetime.now().strftime("%Y-%m-%d %H:%M %A")

    scene = (
        "你在 QQ 群里聊天，群里有很多人，消息格式为「昵称: 内容」，[@你] 表示这条消息 @ 了你。"
        if is_group
        else "你在 QQ 上和一位朋友私聊。"
    )

    parts = [
        p.get("personality", f"你是{nickname}。"),
        f"回复风格：{p.get('reply_style', '')}",
        f"兴趣：{p.get('interest', '')}",
        f"当前时间：{now}",
        scene,
        "工具使用：需要事实信息（时间等）先调工具；决定不回复就调 do_not_reply 而不是输出空话。",
        "回复要求：直接输出聊天内容本身，不要带「昵称:」前缀，不要解释你的决策。",
    ]
    return strip_emoji("\n".join(x for x in parts if x))
