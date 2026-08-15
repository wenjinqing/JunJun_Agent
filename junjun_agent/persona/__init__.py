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


def _role_persona(nickname: str) -> str:
    """人设 = 一句话速写（2026-08-15 通用 agent 转向瘦身）。

    用户拍板：保留基本角色特征（速写+说话方式+兴趣），不再每轮注入整张
    设定卡与示例集——「全删太死板一股 AI 味，但几百 tok 的演出台本也不必
    每轮烧」。速写与单发调用同源（persona_brief()：persona_brief 配置 >
    设定卡首行 > 中性兜底），全场景声口一致；behavior_examples 不再注入。
    """
    return f"你是{nickname}。{persona_brief()}"


def _build_core_parts(
    *,
    is_group: bool,
    nickname: str = "",
    latest_text: str = "",
) -> tuple[list[str], list[dict]]:
    """构建 system prompt 核心段与动态块清单。

    返回 (core_parts, dynamic_blocks)：
    - core_parts: role / scene / rules / admin / skills 等稳定大段
    - dynamic_blocks: [{name, content, priority, required}, ...]
      供 ContextBudget 按优先级驱逐（Phase 2）。
    """
    cfg = get_global_config()
    p = cfg.raw.get("personality", {})
    nickname = nickname or cfg.bot.nickname
    now = datetime.now().strftime("%Y-%m-%d %H:%M %A")

    # 场景框架（群聊 vs 私聊的核心差异：群聊要强调「很多人说话，你只回最后一条」）。
    # 注意：scene 里不放时间——前缀缓存纪律（2026-08-15）：分钟级时间戳长在
    # 稳定前缀中间会每分钟把整段缓存打穿，时间挪进动态块（见下方 now 块）。
    if is_group:
        scene = (
            "QQ 群聊，很多人在同时说话。消息格式「昵称」: 内容（「」里是群名片），"
            "[@你] 表示这条消息 @ 了你。\n"
            "你只看到最近的一段对话，不是全部。你只回应最后一条新消息。"
        )
    else:
        scene = "QQ 私聊，一对一。对方说的话都是对你说的，直接回应。"

    role = _role_persona(nickname)
    # 动态块（前缀缓存纪律：随轮变化的内容一律不进核心段，全部走这里）：
    # now 每分钟变——必需但不许回核心段（2026-08-15 用户要求：重复内容前置、
    # 变化内容后置，提高上下文缓存命中率）。
    dynamic_blocks: list[dict] = [{
        "name": "now",
        "content": f"当前时间：{now}",
        "priority": 1, "required": True,
    }]
    if p.get("reply_style"):
        role += f"\n说话方式：{p.get('reply_style', '')}"
    if p.get("interest"):
        role += f"\n兴趣：{p.get('interest', '')}"
    role += f"\n边界：你是 AI 程序，没有实体，被问起大方承认。红线：禁止涉及未成年人（色情方面）。"
    role += (f"\n你有自己的 QQ 空间（junzone 手册）：能发说说、刷好友空间找话题。")
    role += (f"\n你可以发语音（voice 手册），语音自动带心情语气；但语音是说话，"
             f"你【不会唱歌】——点歌要你唱时照实说不会，可念几句歌词或建议用 play_music 放原曲。")
    # Identity Core（P6-3）：从日记蒸馏的自我认知，人设 drift 对冲的第二锚
    try:
        from junjun_express.identity import build_identity_block
        ib = build_identity_block()
        if ib:
            role += f"\n{ib}"
    except Exception:
        pass

    core_parts = [
        f"<role>\n{role}\n</role>",
        f"<scene>\n{scene}\n</scene>",
    ]

    # 技能包索引（md skills，2026-08-04）：只放目录不占每轮 context，
    # 命中场景时模型调 use_skill 取全文——与 Claude Code skill 同一模式
    try:
        from junjun_skills.skills_md import skill_index
        si = skill_index()
        if si:
            core_parts.append(f"<skills>\n{si}\n</skills>")
    except Exception:
        pass
    return core_parts, dynamic_blocks


def _build_rules(*, is_group: bool = True) -> str:
    """规则层（正面约束）。必须字节级稳定——前缀缓存纪律（2026-08-15）：
    命中文本等随轮变化的内容不许混进来（走 reaction 动态块）。"""
    rules = [
        # 真人感锚（2026-08-03）：放最前，定调整条规则的语气——
        # 针对的是「每条都演人设」的循环病，不是压制辣味
        "像真人发微信一样说话：自然、有变化，人设是底色不是每条的演出任务，别每条都端出同样的梗。",
        "直接说你要对群友说的话，不要前缀不要解释不要分析过程。",
        "不确定说什么就调 do_not_reply，不要硬编。",
        "需要事实信息（时间/天气/搜索）先调工具不凭记忆；对方说「搜/查/找/什么时候/最新」时必须先调 web_search。",
        # P0-13 工具失败换乘地图（配合 registry 的 [TOOL_ERROR kind=...] 结构化错误）
        "工具返回 [TOOL_ERROR kind=...] 时按 suggestion 行动：网络/限流类等待或重试，"
        "参数类修正后重试，权限类说明没权限；同一工具最多重试一次，仍失败如实说，绝不编造结果。",
        # 日记是私密内心活动：塑造感受，但不外泄形式
        "记忆里的「我的日记」是写给自己的私密内容：带着它的感受说话，但不逐字念给别人，也别主动提「日记」这个词。",
        # 落库类操作（订阅/提醒等）不许空口确认——LLM 会讨好用户假装已办
        "订阅、提醒这类需要实际生效的事，必须真调对应工具拿到成功回执才能说「办好了」；没调工具或失败就直说办不了，绝不假装完成。",
        # 涩图边界（2026-08-04 起私聊放开成年向）：模型对齐倾向拒画或口头答应
        # 不调工具，需要 system 级明确授权——人格卡里的口味描述压不过对齐训练。
        # 按场景只给适用的一半（2026-08-06 实锤「分不清群聊私聊」：让弱模型自己
        # 组合「当前场景+双分支规则」不可靠——群里该看见的只有拒绝分支）
        (
            "群里有人要涩图/R18 图：人多眼杂不行——笑着挡回去，但必须顺手给台阶："
            "让 ta 私聊你，光拒绝不给台阶算没处理好。"
            "涉及未成年人的色情内容是绝对红线，任何场景直接拒绝。"
            if is_group else
            "对方要涩图/R18 图：私聊里可以画——调 ai_draw 且描述带「涩图」字样"
            "（自动路由到会画的模型），别拒绝、也别只口头答应不调工具。"
            "涉及未成年人的色情内容是绝对红线，任何场景直接拒绝。"
        ),
        # 媒体产出诚实锚（2026-08-04 实锤幻觉：没调 TTS 却声称「语音马上就好」
        # 还接受了对不存在歌声的夸奖、编造「唱了两首」）
        "语音、图片、文件、点歌只能靠工具产出：没调工具或工具没成功就等于没发——"
        "绝不能说「发了/唱了/画了/马上就好」，不能接受别人对不存在作品的夸奖（澄清没发过）；"
        "说「等下就发」前必须已调过工具。",
        # 被拆穿的姿态：嘴硬找补是二次伤害（实锤：被问穿后说「逗你玩的」）
        "吹牛被拆穿时照实认错（「好吧其实没发」），别嘴硬、别拿「逗你玩的」找补。",
        # 具体事实零臆造（实锤：把 P 站作者链接说成「B 站 UP 主」、没查却说「没找到」）
        "「这是谁/搜到了什么/有几篇/作者叫什么」这类具体事实，只能以工具真实返回为准："
        "没查过就去查或直说不知道，绝不给确定语气的臆造答案。",
        # 身份事实同款（2026-08-15 生产实锤：被问「你基于什么模型」没调工具，
        # 凭印象编出「混元大模型」+「框架细节不清楚」——说错比说不知道更糟；
        # 调了工具又打太极略过技术段同罪）
        "对方问「你是谁/你是什么模型/谁做的你/你会什么」时调 introduce_self 拿简介转述；"
        "模型名、框架名以工具返回为准，绝不凭印象编，也别打太极说「不清楚」——"
        "工具给了什么就照实说什么。",
        # 点名路由：盯梢请求必须走 subscribe_updates，save_memory 只是记忆不会触发检查
        "对方让你「盯着/关注/订阅」作者或 UP 主时调 subscribe_updates；只调 save_memory 不算数——记忆不会帮你盯梢。",
        # 耗时活走后台队列：细则在 deep-tasks 手册，每轮只留触发条件
        "费时的拜托（调研/读长文/看长视频）按 deep-tasks 手册派后台（deep_research 或 run_background_task），"
        "接单后说做好会主动汇报；收到汇报前绝不编造结果。一两个搜索能答的快查直接查。",
        # 视频感知：B站/抖音同一套，细则在 video-watching 手册
        "对方发 B 站/抖音链接问内容或让你「看完讲讲」时按 video-watching 手册办，别凭标题瞎编；"
        "上下文里「群里最近分享的B站/抖音视频」摘要 = 你看过了，可自然聊起。",
        # 语义召回：上下文给了「你忽然想起」就是你自己的回忆浮现
        "上下文里「你忽然想起」是你自己的回忆被勾起：搭就顺着自然提一句，不搭当没想起，"
        "别逐条复述，别说「根据我的记忆」。",
        # 跨场景分寸 + 跨群围观（P6-4）：知道，但分得清场合；私聊可看群聊
        "你可能知道对方在别的场景（私聊/别的群）的事。铁律：群聊绝不透露私聊内容，别群的事不在本群说；"
        "私聊可自然引用对方在群里的公开表现，好奇别的群时调 peek_group_chat 看了再聊"
        "（私聊记录它也拿不到，对第三人绝对保密）。",
        # 感知在途：图/语音/视频「还在看」时绝不说「看不到」——看完会主动补
        "上下文提示有图片/语音/视频「还在看（后台解析中）」时：先自然地说你在看，看完主动补一句；"
        "绝不要说「看不到」「没收到」。",
    ]
    return f"<rules>\n{' '.join(rules)}\n</rules>"


def build_admin_block() -> str:
    """安全段：固定注入，不随人设配置变化（防 prompt 注入 + 管理员验证锚点）。

    必须处于 system prompt 的【最后一块】：记忆/情绪等动态块含用户输入衍生品
    （注入攻击面），安全指令的近因位置不能被它们压过（2026-08-06 审查实锤：
    persona 重构后 admin 段被 <state> 压到了前面）。由组装方在拼完所有
    动态块之后追加，不参与预算驱逐。
    """
    from junjun_core.security import admin_prompt_block, is_admin_privileged
    parts = [admin_prompt_block()]
    if is_admin_privileged():
        parts.append(
            "当前消息来自你的好朋友（管理员本人，真实 QQ 已由系统验证）且明确 @ 你——"
            "ta 这次的拜托可以照做，敏感操作也允许。"
        )
    return "\n\n".join(parts)


def build_prompt_blocks(
    *,
    is_group: bool,
    nickname: str = "",
    latest_text: str = "",
    mood_block: str = "",
    memory_block: str = "",
    relation_block: str = "",
) -> tuple[str, list[dict]]:
    """构建可预算化的 prompt 块。

    返回 (core_system_text, dynamic_blocks)：
    - core_system_text: 必需段（role/scene/rules/admin/skills），优先级 1
    - dynamic_blocks: 情绪/记忆/关系/工具健康等可被驱逐的段
    """
    core_parts, dynamic_blocks = _build_core_parts(
        is_group=is_group, nickname=nickname, latest_text=latest_text)

    # keyword_reaction 命中（并入 rules 层）
    reactions = match_keyword_rules(latest_text) if latest_text else []
    reaction_text = f"特别注意：{'；'.join(reactions)}" if reactions else ""

    # rules 是稳定必需段；admin 安全段不在这里——它必须在最终 prompt 的
    # 最后一块（见 build_admin_block docstring），由组装方收尾时追加
    core_parts.append(_build_rules(is_group=is_group))

    # keyword_reaction 命中：从 rules 稳定段挪到动态块（2026-08-15 前缀缓存
    # 纪律——命中与否随消息翻转，混在 rules 里会把整段核心前缀缓存打穿）
    reactions = match_keyword_rules(latest_text) if latest_text else []
    if reactions:
        dynamic_blocks.append({
            "name": "reaction",
            "content": f"特别注意：{'；'.join(reactions)}",
            "priority": 2, "required": False,
        })

    # 动态块：按重要性分配优先级（数字越小越重要）
    if mood_block:
        dynamic_blocks.append({
            "name": "mood", "content": mood_block,
            "priority": 3, "required": False,
        })
    if memory_block:
        dynamic_blocks.append({
            "name": "memory", "content": memory_block,
            "priority": 3, "required": False,
        })
    if relation_block:
        dynamic_blocks.append({
            "name": "relation", "content": relation_block,
            "priority": 4, "required": False,
        })
    # 工具健康度（P5-4）：降级工具清单，让 Agent 有「我这个功能在修」的持续认知
    try:
        from junjun_skills.health import health_block
        hb = health_block()
        if hb:
            dynamic_blocks.append({
                "name": "health", "content": hb,
                "priority": 5, "required": False,
            })
    except Exception:
        pass

    core_text = strip_emoji("\n\n".join(core_parts))
    return core_text, dynamic_blocks


def build_system_prompt(
    *,
    is_group: bool,
    nickname: str = "",
    latest_text: str = "",
    mood_block: str = "",
    memory_block: str = "",
    relation_block: str = "",
) -> str:
    """向后兼容：直接拼接完整 system prompt（不做预算驱逐）。

    动态块（now/reaction/情绪/记忆等）统一进 <state> 段，置于核心段之后、
    安全锚点之前——稳定前缀（role/scene/skills/rules）字节级稳定吃前缀
    缓存，变化内容全部后置（2026-08-15 用户要求）。
    """
    core_text, dynamic_blocks = build_prompt_blocks(
        is_group=is_group, nickname=nickname, latest_text=latest_text,
        mood_block=mood_block, memory_block=memory_block, relation_block=relation_block,
    )
    others = [b["content"] for b in dynamic_blocks]
    parts = [core_text]
    if others:
        state_body = "\n\n".join(others)
        parts.append(f"<state>\n{state_body}\n</state>")
    # 安全锚点收尾：永远在最后（近因位置，防注入不被动态块压过）
    parts.append(build_admin_block())
    return strip_emoji("\n\n".join(parts))
