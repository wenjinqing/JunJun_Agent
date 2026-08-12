"""提示词工作室：写手 + 评审 + 验收的轻量多 Agent 协作（2026-08-04）。

背景：用户实测出图「太粗糙」。根因是旧扩写只要 60-100 个英文单词——
细度比优秀实践（docs/prompt_skills/ 四份参考 + 用户提供的范例）差一个量级。
Z-Image / Qwen-Image 都是国产新模型，吃**详细中文描述**，不吃 SD 时代的
英文玄学质量词。

协作拓扑（都在本进程内，不引入编排框架）：
- 写手（utils 槽）：按模型家族把口语描述编译成结构化提示词
  - zimage：全中文短语串，黄金词序（参考 zimage-prompt/SKILL.md）
  - qwen：详细中文自然语言段落（写实/文字渲染优势域）
  - anime：不走这里——Danbooru 标签路径在 tools.py 保留
- 评审（utils_small 槽）：对照检查清单挑毛病，直接给修订版（一轮）
- 验收（vlm 槽，可选默认关）：出图后看图对照描述，严重不符带意见重画一次

设计纪律：任何一环失败都降级到旧路径，绝不让「提质量」把「能画」搞坏。
"""

from langchain_core.messages import HumanMessage

from junjun_core.observability import get_logger

logger = get_logger("plugin.ai_draw.studio")

# ---------------------------------------------------------------- 写手模板
# Z-Image：参考 docs/prompt_skills/zimage-prompt——全中文、结构化、禁玄学词、正向限定
_WRITER_ZIMAGE = (
    "你是 Z-Image 文生图模型的中文提示词架构师。把用户的口语化画面描述，编译成"
    "高权重、结构化的全中文提示词。规则：\n"
    "1. 主体绝对不丢失、不改变、不替换；用户没说的元素不要硬加\n"
    "2. 黄金词序，逗号分隔的短语（不写长难句）："
    "[摄影机位/媒介], [主体核心描述（年龄感/外貌/表情）], [服装/动作细节], "
    "[环境/背景], [光影与氛围/艺术风格]\n"
    "3. 禁止使用「杰作、最高画质、8k、超精细」等无物理意义的玄学质量词；"
    "所有约束写成正向限定（不写「背景不杂乱」，写「纯色干净背景」）\n"
    "4. 细节要具体到物理属性：发型发质、五官、表情、服装面料剪裁、光线方向质感、"
    "镜头感（特写/全身/景深/手机随拍感）——细度决定出图质量\n"
    "5. 画面需要文字时用规范语法：text \"具体内容\"\n"
    "6. 60-150 字，只输出提示词本身，不解释、不引号、不换行\n"
    "用户描述：{prompt}"
)

# Qwen-Image：写实/文字渲染优势域，吃详细自然语言段落（用户范例风格）
_WRITER_QWEN = (
    "你是 Qwen-Image 文生图模型的提示词专家（该模型擅长写实摄影感与图中文字渲染）。"
    "把用户的口语化画面描述，扩写成一段详细生动的全中文自然语言画面描述。规则：\n"
    "1. 主体绝对不丢失、不改变、不替换；用户没说的关键元素不要硬加\n"
    "2. 依次写足：主体（年龄感/五官/发型/表情/身材）→ 服装（款式/颜色/面料）→ "
    "姿势动作 → 场景背景（具体物件与空间关系）→ 光线（光源方向/质感/时间感）→ "
    "镜头与构图（特写/全身/视角/景深）→ 整体氛围\n"
    "3. 写实感诀窍：写明拍摄媒介与光感，如「智能手机随手拍摄、柔和均匀的环境光、"
    "色调自然、清晰度高」——像描述一张真实照片，不像画画订单\n"
    "4. 画面需要文字时用引号括起具体文字内容，并说明字体气质与位置\n"
    "5. 禁止使用「杰作、最高画质、8k」等玄学质量词\n"
    "6. 120-250 字一段成文，只输出描述本身，不解释、不换行\n"
    "用户描述：{prompt}"
)

# ---------------------------------------------------------------- 评审模板
_CRITIC = (
    "你是 AI 绘画提示词评审。下面是【用户原始需求】和一封【候选提示词】。"
    "对照检查清单逐项核对，然后直接输出修订后的最终提示词（只输出提示词本身，"
    "保持原语言与格式，不解释、不点评）：\n"
    "- 主体与用户明确要求的元素是否全部保留、没有被替换或遗漏？\n"
    "- 是否有「杰作/最高画质/8k」等玄学质量词？（删掉）\n"
    "- 是否写了光线？是否写了视角/构图？（缺则补上，但别发明用户没说的主体元素）\n"
    "- 是否有与用户意图冲突的机位/构图？（用户意图最高优先，改掉）\n"
    "- 画面文字是否用了规范语法、内容准确？\n"
    "- 长度是否合适（过短补细节，冗长删减）？\n"
    "【用户原始需求】{origin}\n【候选提示词】{draft}"
)

# ---------------------------------------------------------------- 验收模板
_REVIEW = (
    "你是出图验收员。对照【需求描述】检查这张生成的图，只挑严重问题"
    "（主体错了/关键元素缺失/画面崩坏/文字写错）；风格喜好类的小差异不算问题。"
    "没有严重问题就只回复两个字：通过。有问题用一句话说明哪里不对（不超过 50 字）。\n"
    "【需求描述】{prompt}"
)


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("ai_draw", {}) or {}
    except Exception:
        return {}


async def _call(slot: str, text: str) -> str:
    from junjun_llm import get_chat_model, get_callbacks
    resp = await get_chat_model(slot).ainvoke(
        [HumanMessage(content=text)],
        # 接 Langfuse：后台任务里没有 agent trace 的 callbacks，
        # 不传的话写手/评审在 Langfuse 里完全不可见（2026-08-04 用户实测
        # 「trace 里看不到工作室效果」——它们会以独立 trace 出现）
        config={"callbacks": get_callbacks(),
                "metadata": {"langfuse_tags": ["junjun", "ai_draw", "prompt-studio"],
                             "langfuse_session_id": "ai_draw"}},
    )
    return (resp.content or "").strip().strip('"').replace("\n", " ")


async def craft_prompt(prompt: str, family: str) -> str:
    """写手 (+ 评审) 协作产出最终提示词。失败返回 ""（调用方降级旧路径）。

    family: "zimage" | "qwen"（anime 不走这里）。
    """
    template = {"zimage": _WRITER_ZIMAGE, "qwen": _WRITER_QWEN}.get(family)
    if template is None:
        return ""
    draft = await _call("utils", template.format(prompt=prompt))
    if not draft:
        return ""
    if not bool(_cfg().get("prompt_critic", True)):
        return draft[:600]
    try:
        revised = await _call("utils_small", _CRITIC.format(origin=prompt, draft=draft))
        if revised:
            return revised[:600]
    except Exception as e:
        logger.warning(f"提示词评审失败（用写手稿）: {type(e).__name__}: {e}")
    return draft[:600]


async def review_image(url: str, origin_prompt: str) -> str | None:
    """VLM 验收：None=通过（或验收不可用），str=严重问题描述（供重画修订）。

    url 可以是远程 URL 或本地路径（2026-08-12 起 AI Ping 成品落盘本地）。"""
    try:
        import base64
        import os.path
        if os.path.exists(str(url)):
            with open(url, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        else:
            import httpx
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode()
        from junjun_llm import get_chat_model, get_callbacks
        vlm = get_chat_model("vlm")
        resp = await vlm.ainvoke([HumanMessage(content=[
            {"type": "text", "text": _REVIEW.format(prompt=origin_prompt)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])], config={"callbacks": get_callbacks(),
                     "metadata": {"langfuse_tags": ["junjun", "ai_draw", "review"],
                                  "langfuse_session_id": "ai_draw"}})
        verdict = (resp.content or "").strip()
        if not verdict or verdict.startswith("通过"):
            return None
        return verdict[:100]
    except Exception as e:
        logger.warning(f"出图验收失败（视为通过）: {type(e).__name__}: {e}")
        return None
