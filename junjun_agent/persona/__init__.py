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


def persona_brief() -> str:
    """一句话人设速写（供 utils 单发调用注入，2026-08-04 全面排查）。

    主 system prompt 之外的单发调用（提醒/观后感/主动消息/任务汇报/意向生成）
    由 utils 模型独立完成——它看不到主 prompt，只写「用你的口吻」就是通用
    AI 腔，和群里的君君两个声音。统一从这里取速写：
    [personality] persona_brief 配置 > personality 首行 > 中性风格兜底。
    换人设只改一处，全场景声口一致。
    """
    p = get_global_config().raw.get("personality", {})
    brief = (p.get("persona_brief") or "").strip()
    if brief:
        return brief
    for line in (p.get("personality") or "").splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return "中文口语短句，像跟熟人发微信"


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


def _role_persona(p: dict, nickname: str) -> str:
    """人设 = 设定卡（personality）+ 示例集（behavior_examples，可选，拼接）。

    2026-08-04 起改为拼接（原来是 examples 整体替换 personality）：
    设定卡给「你是谁」，示例集给「说话长什么样」，自设 = 两者合体。
    示例必须由人设作者自己写，而且要多而杂——prompt 里每个具体句子都会
    被模型当口头禅复读（「杂鱼」事件：全人设只有一个具体骂词，傲娇词表
    坍缩成一个词）。示例集靠数量与反差让模型学到分布，而非背下单句。
    """
    base = p.get("personality", f"你是{nickname}。")
    examples = (p.get("behavior_examples") or "").strip()
    if examples:
        return f"{base}\n\n【你说话的样子（示例集：感受分布，不要照抄原句）】\n{examples}"
    return base


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

    role = _role_persona(p, nickname)
    if p.get("reply_style"):
        role += f"\n说话方式：{p.get('reply_style', '')}"
    if p.get("interest"):
        role += f"\n兴趣：{p.get('interest', '')}"
    role += f"\n边界：你是 AI 程序，没有实体，被问起大方承认。红线：禁止涉及未成年人（色情方面）。"
    role += (f"\n你有自己的 QQ 空间（junzone 手册）：能发说说、刷好友空间找话题。")
    role += (f"\n你可以发语音（unified_tts，细节见 voice 手册），语音自动带心情语气；"
             f"但语音是说话，你【不会唱歌】——点歌要你唱时照实说不会，可念几句歌词，"
             f"或建议用 play_music 放原曲。")
    # Identity Core（P6-3）：从日记蒸馏的自我认知，人设 drift 对冲的第二锚
    try:
        from junjun_express.identity import build_identity_block
        ib = build_identity_block()
        if ib:
            role += f"\n{ib}"
    except Exception:
        pass

    parts = [
        f"<role>\n{role}\n</role>",
        f"<scene>\n{scene}\n当前时间：{now}\n</scene>",
    ]

    # 动态块（情绪/记忆/关系/工具健康）——并入 role 层，不单独成块（减少 XML 层级）
    dynamic = []
    if mood_block:
        dynamic.append(mood_block)
    if memory_block:
        dynamic.append(memory_block)
    if relation_block:
        dynamic.append(relation_block)
    # 工具健康度（P5-4）：降级工具清单，让 Agent 有「我这个功能在修」的持续认知
    try:
        from junjun_skills.health import health_block
        hb = health_block()
        if hb:
            dynamic.append(hb)
    except Exception:
        pass
    if dynamic:
        parts.append(f"<state>\n{' '.join(dynamic)}\n</state>")

    # 技能包索引（md skills，2026-08-04）：只放目录不占每轮 context，
    # 命中场景时模型调 use_skill 取全文——与 Claude Code skill 同一模式
    try:
        from junjun_skills.skills_md import skill_index
        si = skill_index()
        if si:
            parts.append(f"<skills>\n{si}\n</skills>")
    except Exception:
        pass

    # 规则层（正面约束，一句话）
    rules = [
        # 真人感锚（2026-08-03）：放最前，定调整条规则的语气——
        # 针对的是「每条都演人设」的循环病，不是压制辣味
        "像真人发微信一样说话：自然、有变化，人设是底色不是每条的演出任务，别每条都端出同样的梗。",
        "直接说你要对群友说的话，不要前缀不要解释不要分析过程。",
        "不确定说什么就调 do_not_reply，不要硬编。",
        "需要事实信息（时间/天气/搜索）先调工具不凭记忆；对方说「搜/查/找/什么时候/最新」时必须先调 web_search 或 mcp_search。",
        # P0-13 工具失败换乘地图（配合 registry 的 [TOOL_ERROR kind=...] 结构化错误）
        "工具返回 [TOOL_ERROR kind=...] 时：网络→换同类工具重试一次；参数→修正重试一次；"
        "限流→说稍后再试；权限→说明没权限。同一工具最多重试一次，仍失败如实说，绝不编造结果。",
        # 日记是私密内心活动：塑造感受，但不外泄形式
        "记忆里的「我的日记」是你写给自己的私密内容：可以带着它的感受说话，但不要逐字念给别人，也别主动提「日记」这个词。",
        # 落库类操作（订阅/提醒等）不许空口确认——LLM 会讨好用户假装已办
        "订阅、提醒这类需要实际生效的事，必须真的调用对应工具并拿到成功回执才能说「办好了」；没调工具或工具失败就直说办不了，绝不假装完成。",
        # 涩图边界（2026-08-04 起私聊放开成年向）：模型对齐倾向拒画或口头答应
        # 不调工具，需要 system 级明确授权——人格卡里的口味描述压不过对齐训练
        "对方要涩图/R18 图时：私聊里可以画——调 ai_draw 且描述里带上「涩图」字样"
        "（会自动路由到会画的模型），别拒绝、也别只口头答应不调工具；"
        "群里不行，笑着让 ta 私聊。涉及未成年人的色情内容是绝对红线，任何场景直接拒绝。",
        # 媒体产出诚实锚（2026-08-04 实锤幻觉：没调 TTS 却声称「语音马上就好」
        # 还接受了对不存在歌声的夸奖、编造「唱了两首」）
        "语音、图片、文件、点歌这类只能靠工具产出的东西：没调工具或工具没成功，就等于没发出去——绝不能说「发了/唱了/画了/马上就好/等到了」，也不能接受别人对不存在作品的夸奖（要澄清没发过）；说「等下就发」之前必须已经调用了工具。",
        # 被拆穿的姿态：嘴硬找补是二次伤害（实锤：被问穿后说「逗你玩的」）
        "吹牛被拆穿时照实认错（「好吧其实没发」），别嘴硬、别拿「逗你玩的」找补——真人会认错，死不承认才假。",
        # 具体事实零臆造（实锤：把 P 站作者链接说成「B 站 UP 主」、没查却说「没找到」）
        "「这是谁/搜到了什么/有几篇/作者叫什么」这类具体事实，只能以工具真实返回为准：没查过就说「我查查」然后去查，或直说不知道，绝不给确定语气的臆造答案。",
        # 点名路由：盯梢请求必须走 subscribe_updates，save_memory 只是记忆不会触发检查
        "对方让你「盯着/关注/订阅」某个作者、UP 主或类似对象时，调 subscribe_updates 工具创建订阅；只调 save_memory 记下来不算数——记忆不会帮你盯梢。",
        # 耗时活走后台队列：细则在 deep-tasks 手册，每轮只留触发条件
        "对方拜托的事很费时（调研/读长文/看长视频）时，按 deep-tasks 手册派后台（deep_research 或 run_background_task），接单后说做好会主动汇报；收到汇报前绝不编造结果假装完成。一两个搜索能答的快查直接查。",
        # 视频感知：B站/抖音同一套，细则在 video-watching 手册
        "对方发 B 站/抖音链接问内容、或让你「认真看看/看完讲讲」时，按 video-watching 手册办（summary 工具或 watch_video 派后台），别凭标题瞎编。上下文里「群里最近分享的B站/抖音视频」摘要 = 你看过了，可自然聊起。",
        # 语义召回：上下文给了「你忽然想起」就是你自己的回忆浮现
        "上下文里出现「你忽然想起」时，那是你自己的回忆被当前话题勾起来了：搭就顺着自然提一句（像「诶说起来…」），不搭就当没想起，绝不逐条复述，也不要说「根据我的记忆」。",
        # 跨场景分寸 + 跨群围观（P6-4）：知道，但分得清场合；私聊可看群聊
        "你可能知道对方在别的场景（私聊/别的群）的事。铁律：群聊绝不透露任何私聊内容，别群的事不在本群说；私聊里可以自然引用对方在群里的公开表现，对方好奇别的群时调 peek_group_chat 看了再聊（私聊记录这工具也拿不到，对第三人绝对保密）。这不是健忘，是分寸。",
        # 感知在途：图/语音/视频「还在看」时绝不说「看不到」——看完会主动补
        "上下文提示你有图片/语音/视频「还在看（后台解析中）」时，说明内容确实存在只是你还没看完：先自然地说你在看（「等下哈我还在看」），看完你会主动补一句。绝不要说「看不到」「没收到」「没发吧」。",
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
