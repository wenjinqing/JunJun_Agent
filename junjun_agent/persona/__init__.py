"""persona: system prompt 组装（阶段 3 全量）。

分块拼装：人设 + reply_style + plan_style + interest + 当前时间 + 场景块
+ keyword_reaction 命中规则 + 情绪占位（阶段5）+ 记忆摘要占位（阶段4）。
strip_emoji：原项目实测 system prompt 含 emoji 干扰 function calling schema。
"""

import re
from datetime import datetime
from typing import List

from junjun_core.config import get_global_config

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F900-\U0001F9FF]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def match_keyword_rules(text: str) -> List[str]:
    """keyword_reaction 命中规则 -> reaction 提示列表（对齐原 [keyword_reaction]）。"""
    rules = get_global_config().raw.get("keyword_reaction", {}).get("keyword_rules", []) or []
    hits = []
    low = text.lower()
    for rule in rules:
        kws = rule.get("keywords", [])
        if any(str(k).lower() in low for k in kws):
            reaction = rule.get("reaction", "")
            if reaction:
                hits.append(reaction)
    return hits


def build_system_prompt(
    *,
    is_group: bool,
    nickname: str = "",
    latest_text: str = "",
    mood_block: str = "",
    memory_block: str = "",
    relation_block: str = "",
) -> str:
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
        f"表达倾向：{p.get('plan_style', '')}" if p.get("plan_style") else "",
        f"兴趣：{p.get('interest', '')}",
        f"当前时间：{now}",
        scene,
        mood_block,
        memory_block,
        relation_block,
    ]

    # keyword_reaction 命中注入
    if latest_text:
        for reaction in match_keyword_rules(latest_text):
            parts.append(f"特别注意：{reaction}")

    parts += [
        "工具使用：需要事实信息（时间等）先调工具；决定不回复就调 do_not_reply 而不是输出空话。",
        "回复要求：直接输出聊天内容本身，不要带「昵称:」前缀，不要解释你的决策，不要用括号描述动作。",
        "上下文说明：下面这段聊天记录只是让你了解背景（谁刚说了什么）。你只回复最后一条新消息，"
        "不要复述/接续/模仿聊天里任何人的说话内容，不要把聊天记录里的话当成你自己的输出。",
        "【输出纪律·最高优先级】你的回复会被直接发给 QQ 群友。"
        "绝对禁止：分析过程（\"让我想想\"\"首先\"\"根据...\"）、复述问题、解释你怎么得出答案、"
        "写\"我需要回应\"\"用户在问\"这类元描述。只输出你要对群友说的那句话本身。"
        "如果不确定说什么，调 do_not_reply 而不是硬编。",
    ]

    # 安全段：固定注入，不随人设配置变化（防 prompt 注入 + 管理员验证锚点）
    from junjun_core.security import admin_prompt_block, is_admin_privileged
    parts.append(admin_prompt_block())
    if is_admin_privileged():
        parts.append(
            "当前消息来自你的好朋友（管理员本人，真实 QQ 已由系统验证）且明确 @ 你——"
            "ta 这次的拜托可以照做，敏感操作也允许。"
        )
    return strip_emoji("\n".join(x for x in parts if x))
