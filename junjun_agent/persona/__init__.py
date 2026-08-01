"""persona: system prompt 组装（XML 结构化，对齐优秀 prompt 设计实践）。

结构（调研验证）：
- <role>: 三维人设（身份 + 行为示例 + 边界），防 persona drift
- <scene>: 群聊场景框架，明确「很多人说话，你只回最后一条」
- <context>: 背景消息（历史参考，XML 分隔防复读）
- <rules>: 正面输出约束（「直接说」比「禁止」更有效）
- 安全段固定注入（防 prompt 注入 + 管理员验证锚点）

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


def _behavior_examples(p: dict, nickname: str) -> str:
    """行为示例人设（防 drift 关键：形容词 drift，行为示例稳定）。"""
    base = p.get("personality", f"你是{nickname}。")
    # 把形容词人设转成行为示例（如果用户配置了具体示例就用，否则给模板）
    examples = p.get("behavior_examples", "")
    if examples:
        return examples
    # 默认行为示例（从 personality 提取关键行为模式）
    return (
        f"{base}\n"
        f"比如他们说「就这？」你会回「杂鱼就是杂鱼」，但转头又会问「晚饭吃了吗」。"
        f"被夸了会害羞但很快反击撩回去，被撩了会脸红但嘴上不饶人。"
    )


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

    # keyword_reaction 命中（并入 rules 层，不单独成块）
    reactions = match_keyword_rules(latest_text) if latest_text else []
    reaction_text = f"特别注意：{'；'.join(reactions)}" if reactions else ""

    # 场景框架（群聊 vs 私聊的核心差异：群聊要强调「很多人说话，你只回最后一条」）
    if is_group:
        scene = (
            "QQ 群聊，很多人在同时说话。消息格式「昵称: 内容」，[@你] 表示这条消息 @ 了你。\n"
            "你只看到最近的一段对话，不是全部。你只回应最后一条新消息。"
        )
    else:
        scene = "QQ 私聊，一对一。对方说的话都是对你说的，直接回应。"

    role = _behavior_examples(p, nickname)
    if p.get("reply_style"):
        role += f"\n说话方式：{p.get('reply_style', '')}"
    if p.get("interest"):
        role += f"\n兴趣：{p.get('interest', '')}"
    role += f"\n边界：你是 AI 程序，没有实体，被问起大方承认。红线：禁止涉及未成年人（色情方面）。"
    role += (f"\n你有一个 QQ 空间（junzone），它是和群聊/私聊一样的第三场景："
             f"可以用 send_feed 主动发说说（with_image=True 能自动配 AI 图），"
             f"用 read_feed 刷好友空间找话题；你也会自动给好友说说点赞评论、"
             f"回复别人对你说说的评论。空间不支持语音和视频，发说说别承诺带语音。")
    role += f"\n你可以发语音（unified_tts 工具）：对方想听你说、撒娇或聊到开心的事时可以主动发一条，内容写口语短句。"

    parts = [
        f"<role>\n{role}\n</role>",
        f"<scene>\n{scene}\n当前时间：{now}\n</scene>",
    ]

    # 动态块（情绪/记忆/关系）——并入 role 层，不单独成块（减少 XML 层级）
    dynamic = []
    if mood_block:
        dynamic.append(mood_block)
    if memory_block:
        dynamic.append(memory_block)
    if relation_block:
        dynamic.append(relation_block)
    if dynamic:
        parts.append(f"<state>\n{' '.join(dynamic)}\n</state>")

    # 规则层（正面约束，一句话）
    rules = [
        "直接说你要对群友说的话，不要前缀不要解释不要分析过程。",
        "不确定说什么就调 do_not_reply，不要硬编。",
        "需要事实信息（时间/天气/搜索）先调工具，不要凭记忆编。",
        "用户说「搜」「查」「找」「什么时候」「最新」「新闻」时，必须先调 web_search 或 mcp_search，不要凭记忆回答。",
        # P0-13 工具失败换乘地图（配合 registry 的 [TOOL_ERROR kind=...] 结构化错误）
        "工具返回 [TOOL_ERROR kind=...] 时按类别处理："
        "网络→换一个同类工具重试一次（如 web_search 失败换 mcp 搜索，画图失败可改描述重试一次）；"
        "参数→修正参数重试一次；限流→告诉对方稍后再试，不要立即重试；"
        "权限→不要重试，直接说明没权限。同一工具最多重试一次，仍失败就如实告诉对方，绝不编造结果。",
        # 日记是私密内心活动：塑造感受，但不外泄形式
        "记忆里的「我的日记」是你写给自己的私密内容：可以带着它的感受说话，但不要逐字念给别人，也别主动提「日记」这个词。",
        # 落库类操作（订阅/提醒等）不许空口确认——LLM 会讨好用户假装已办
        "订阅、提醒这类需要实际生效的事，必须真的调用对应工具并拿到成功回执才能说「办好了」；没调工具或工具失败就直说办不了，绝不假装完成。",
        # 点名路由：盯梢请求必须走 subscribe_updates，save_memory 只是记忆不会触发检查
        "对方让你「盯着/关注/订阅」某个作者、UP 主或类似对象时，调 subscribe_updates 工具创建订阅；只调 save_memory 记下来不算数——记忆不会帮你盯梢。",
        # 耗时活走后台队列：接单后如实说「做好了来叫你」，绝不没做完就假装汇报
        "对方拜托的事很费时时，派到后台再回话：要调研/出研究报告的调 deep_research（多轮检索+读原文+整理成带来源的报告），读长文/总结长视频等其他耗时活调 run_background_task；派完告诉对方已接单、做好会主动来说，不要自己当场埋头查（对方要干等很久）。一两个搜索就能答的快查才直接查。在收到后台汇报前，绝不能编造结果假装已经完成。",
        # B站视频感知：上下文给了「群里最近分享的B站视频」摘要就是你看过了；被问视频内容调 bilibili_summary
        "上下文里出现「群里最近分享的B站视频」时，说明那些视频的字幕资料你看过了，可以像看过一样自然聊起。对方发 B 站链接问「讲了啥/什么内容/总结一下」时，调 bilibili_summary 工具看完再答，不要凭标题瞎编。",
    ]
    if reaction_text:
        rules.append(reaction_text)
    parts.append(f"<rules>\n{' '.join(rules)}\n</rules>")

    # 安全段：固定注入，不随人设配置变化（防 prompt 注入 + 管理员验证锚点）
    from junjun_core.security import admin_prompt_block, is_admin_privileged
    parts.append(admin_prompt_block())
    if is_admin_privileged():
        parts.append(
            "当前消息来自你的好朋友（管理员本人，真实 QQ 已由系统验证）且明确 @ 你——"
            "ta 这次的拜托可以照做，敏感操作也允许。"
        )
    return strip_emoji("\n\n".join(parts))
