"""ai_draw 插件：AI 文生图（迁移自旧 ai_draw_plugin，新架构重写）。

命令：/draw <描述>（/绘图 /画图 同）
工具：ai_draw（LLM 自动配图）
API：ModelScope 异步文生图（api-inference.modelscope.cn）
  - POST /v1/images/generations（头 X-ModelScope-Async-Mode: true）提交任务 -> task_id
  - GET  /v1/tasks/{task_id} 轮询（间隔 5s，总超时 120s）-> output_images[0]
模型路由：描述含 动漫/二次元/anime 等词时用二次元模型，否则默认模型；
  env AI_DRAW_MODEL / AI_DRAW_MODEL_ANIME 可覆盖默认值。
提示词工程（按模型家族定制）：
  - 默认（Z-Image-Turbo）/写实（Qwen-Image）：prompt_studio 提示词工作室——
    中文结构化写手 + 评审修订一轮（[ai_draw] prompt_critic 可关）；
    可选 VLM 出图验收重画一次（[ai_draw] review_enable，默认关）；
    工作室失败降级旧英文扩写（expand_prompt）
  - 二次元（WAI-illustrious-SDXL）：Danbooru 标签串 + 质量词后缀 + 负面提示词
    （防烂手/多余肢体/水印），模型不接受 negative_prompt 时自动降级重试
安全：描述命中「未成年词 + 性词」组合直接拒绝；未配置 MODELSCOPE_API_KEY 降级文本。
限流：每会话 20 秒最小间隔（内存 dict）。
异步：工具/命令均为「提交即返回」——后台轮询完成由 task_manager 直发图片，
  不阻塞会话，也不依赖 LLM 复述任何标记。
"""

import asyncio
import os
import time

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_agent.interceptors import register_interceptor
from junjun_agent.tasks import task_manager
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.ai_draw")

_API_BASE = os.environ.get("AI_DRAW_API_BASE", "https://api-inference.modelscope.cn")
_HTTP_TIMEOUT = 30.0
_POLL_INTERVAL = 5.0    # 轮询间隔（秒）
_POLL_TIMEOUT = 120.0   # 轮询总超时（秒）
_COOLDOWN = 20.0        # 每会话最小间隔（秒）
_EXPAND_MAX_LEN = 200   # 描述长于该长度时不扩写（已经够详细，避免稀释）

# 默认生图模型（取自旧插件 config.toml，可用 env 覆盖）
_DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"
_DEFAULT_ANIME_MODEL = "QWQ114514123/WAI-illustrious-SDXL-v16"
_DEFAULT_QWEN_MODEL = "Qwen/Qwen-Image-2512"
# AI Ping 生图模型（2026-08-12 平台迁移：同步 OpenAI 风格 /images/generations，
# 返回 data[0].url；与 ModelScope 异步任务协议不同，generate() 按模型分流）：
#   Kolors ¥0（可图，快手中文生图——默认 SFW）、GLM-Image ¥0.10、Doubao-Seedream-4.0 ¥0.20
_DEFAULT_KOLORS_MODEL = "Kolors"
_DEFAULT_GLM_IMAGE_MODEL = "GLM-Image"
_DEFAULT_SEEDREAM_MODEL = "Doubao-Seedream-4.0"

# 模型别名 -> 实际 Model-Id（env 可覆盖）：显式指定或关键词路由用
def _model_registry() -> dict:
    return {
        "kolors": os.environ.get("AI_DRAW_MODEL_KOLORS", "") or _DEFAULT_KOLORS_MODEL,
        "glm-image": os.environ.get("AI_DRAW_MODEL_GLM_IMAGE", "") or _DEFAULT_GLM_IMAGE_MODEL,
        "seedream": os.environ.get("AI_DRAW_MODEL_SEEDREAM", "") or _DEFAULT_SEEDREAM_MODEL,
        "zimage": os.environ.get("AI_DRAW_MODEL", "") or _DEFAULT_MODEL,
        "anime": os.environ.get("AI_DRAW_MODEL_ANIME", "") or _DEFAULT_ANIME_MODEL,
        "qwen": os.environ.get("AI_DRAW_MODEL_QWEN", "") or _DEFAULT_QWEN_MODEL,
    }


def _aiping_model_ids() -> set:
    """当前走 AI Ping 网关的 Model-Id 集合（随 env 覆盖动态变化）。"""
    reg = _model_registry()
    return {reg["kolors"], reg["glm-image"], reg["seedream"]}

# ---------------- 提示词工程（按模型家族定制，两套风格不可混用） ----------------
# Z-Image-Turbo：中英双语自然语言完整描述效果最好（光照/色彩/构图/质感）
_EXPAND_PROMPT_DEFAULT = (
    "你是顶级 AI 绘画提示词专家。把用户的画面描述扩写成一段生动细腻的英文画面描述"
    "（供 Z-Image 文生图模型使用）。规则：\n"
    "1. 主体绝对不丢失、不改变、不替换，放在句首；用户没说的元素不要硬加\n"
    "2. 用自然语言完整句子（不是标签堆砌），依次补充：环境细节、光照"
    "（如 soft rim light / golden hour / volumetric lighting）、色彩基调、"
    "材质质感、构图与镜头感（如 close-up / wide shot / depth of field / bokeh）、氛围\n"
    "3. 60-100 个英文单词，只输出描述本身，不要解释、不要引号、不要换行\n"
    "用户描述：{prompt}"
)
# WAI-illustrious-SDXL：Illustrious/SDXL 系吃 Danbooru 标签 + 质量词
_EXPAND_PROMPT_ANIME = (
    "你是顶级 AI 绘画提示词专家，精通 Danbooru 标签体系（Illustrious/SDXL 系模型）。"
    "把用户的中文画面描述转写为英文 Danbooru 标签串。规则：\n"
    "1. 主体绝对不丢失、不改变（角色/物体翻译为准确英文标签，如 猫娘 -> catgirl, cat ears, cat tail）\n"
    "2. 标签顺序：主体（1girl/1boy/solo 等 -> 发型发色 -> 瞳色 -> 服饰 -> 表情 -> 姿势动作）"
    "-> 场景背景 -> 光照氛围 -> 构图视角\n"
    "3. 全英文小写、逗号分隔、25-45 个标签；只用标签不写句子，不要序号不要解释\n"
    "4. 不要输出 masterpiece / best quality 等质量标签（质量后缀由系统统一追加，重复会稀释权重）\n"
    "5. 可补充提升画面完成度的标签（如 detailed background, soft lighting, depth of field），"
    "绝不添加用户没说的 NSFW/未成年元素\n"
    "用户描述：{prompt}"
)
# 质量词后缀（Illustrious 系惯例，显著提升出图质量）
_ANIME_QUALITY_SUFFIX = "masterpiece, best quality, very aesthetic, absurdres"
# 负面提示词（防崩坏：烂手/多余肢体/水印文字等）
_ANIME_NEGATIVE = (
    "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra fingers, "
    "fused fingers, extra limbs, mutated limbs, bad proportions, blurry, lowres, "
    "jpeg artifacts, watermark, signature, text, logo, username"
)
_DEFAULT_NEGATIVE = "低质量，模糊，过曝，变形，错误解剖，多余手指，多余肢体，水印，文字，签名"

# 内容红线：未成年词 与 性词 同时命中 -> 直接拒绝
_MINOR_WORDS = ("萝莉", "幼女", "小学生", "儿童", "幼童", "女童", "男童", "未成年",
                "underage", "preteen", "child", "loli")
_NSFW_WORDS = ("色情", "裸", "sex", "涩情", "裸体", "nsfw", "porn",
               "涩图", "色图", "r18")  # 放开成年向后这三个也是性词，红线组合要认得

# 二次元/动漫画风词：命中则路由到二次元特化模型
# （含涩图词：WAI-illustrious 是唯一能出 R18 的模型，
#  ModelScope 的 zimage/qwen 走平台过滤必失败——2026-08-04 起私聊放开成年向）
_ANIME_WORDS = ("动漫", "二次元", "anime", "漫画", "番", "manga",
                "涩图", "色图", "涩涩", "r18", "nsfw")

# Qwen-Image 优势域：写实/摄影 + 图中文字渲染（Qwen-Image 的中英文写字能力最强）
_QWEN_WORDS = ("写实", "照片", "真人", "摄影", "海报", "带字", "文字", "招牌",
               "贺卡", "封面", "photorealistic")

# 「画自己」触发词：命中则把人设词附加到 prompt
_SELF_WORDS = ("你", "自己", "自画像", "自拍")

# 每会话上次画图时间戳（chat_id -> ts）
_last_use: dict = {}


def _api_key() -> str:
    """每次调用实时读 env（便于测试与热更）。"""
    return os.environ.get("MODELSCOPE_API_KEY", "")


def _aiping_key() -> str:
    return os.environ.get("AIPING_API_KEY", "").strip()


def _aiping_base() -> str:
    return os.environ.get("AIPING_BASE_URL", "").strip().rstrip("/")


def _any_provider_key() -> str:
    """任一生图平台凭据（门控用：默认模型走 AI Ping，ModelScope 系别名才要 MS key）。"""
    return _api_key() or _aiping_key()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def is_minor_nsfw(prompt: str) -> bool:
    """命中未成年红线（未成年词 + 性词组合）-> True（必须拒绝生成）。"""
    low = (prompt or "").lower()
    has_minor = any(w.lower() in low for w in _MINOR_WORDS)
    has_nsfw = any(w.lower() in low for w in _NSFW_WORDS)
    return has_minor and has_nsfw


# R18 路由标记（_ANIME_WORDS 的成年向子集）：带这些词会被路由到唯一能出
# R18 的 anime 模型——群聊硬门认这套。未带标记的露骨描述走默认模型，
# ModelScope 平台过滤层会杀，不需要也不该在这里拦（误伤正常二次元请求）。
# 同步纪律：junjun_agent/agent.py 的 _NUDGE_NSFW_WORDS 是这套词的镜像，改一处必须改另一处
_R18_MARKERS = ("涩图", "色图", "涩涩", "r18", "nsfw")


def has_r18_marker(prompt: str) -> bool:
    """描述带 R18 路由标记 -> True（群聊硬门的「真会出 R18」判定）。"""
    low = (prompt or "").lower()
    return any(w in low for w in _R18_MARKERS)


def is_anime(prompt: str) -> bool:
    """命中二次元/动漫画风词 -> True（路由到二次元特化模型）。"""
    low = (prompt or "").lower()
    return any(w.lower() in low for w in _ANIME_WORDS)


def is_qwen_domain(prompt: str) -> bool:
    """命中写实/文字渲染词 -> True（路由到 Qwen-Image）。"""
    low = (prompt or "").lower()
    return any(w.lower() in low for w in _QWEN_WORDS)


def route_model(prompt: str, explicit: str = "") -> str:
    """根据描述选择生图模型：显式别名 > 写实/文字词(qwen) > 二次元词(anime) > 默认(kolors)。"""
    reg = _model_registry()
    if explicit and explicit.lower() in reg:
        return reg[explicit.lower()]
    if is_qwen_domain(prompt):
        return reg["qwen"]
    if is_anime(prompt):
        return reg["anime"]
    return reg["kolors"]


def model_style(model: str) -> str:
    """模型 -> 提示词风格：anime 家族走 Danbooru 标签，其余走自然语言细描。"""
    return "anime" if model == _model_registry()["anime"] else "default"


def _get_persona() -> str:
    """取「画自己」用的视觉形象描述。

    优先 [personality] appearance（专门的视觉形象卡：发型/眼睛/服装风格——
    画图要的是画面不是性格）。没有则退回 personality 首行（legacy 行为，
    但性格首行大多是抽象词，对画图帮助有限，建议配 appearance）。
    2026-08-04 实战：学姐设定卡首行是「谁愿意找你说心事的那种」，
    80 字性格描述前置进画面 prompt 只会稀释主体。
    """
    try:
        from junjun_core.config import get_global_config
        raw = get_global_config().raw or {}
        p = raw.get("personality") or {}
        appearance = str(p.get("appearance") or "").strip()
        if appearance:
            return appearance[:150]
        text = str(p.get("personality") or "")
        # 取第一段（首行或前 80 字），避免整段人设过长稀释画面主体
        first = text.split("\n", 1)[0].strip()
        return first[:80]
    except Exception as e:
        logger.warning(f"读取人设配置失败: {type(e).__name__}: {e}")
        return ""


def apply_self_prompt(prompt: str) -> str:
    """「画自己」类描述：含 你/自己/自画像/自拍 时把人设词附加到 prompt 前。"""
    if any(w in (prompt or "") for w in _SELF_WORDS):
        persona = _get_persona()
        if persona:
            return f"{persona}，{prompt}"
    return prompt


async def expand_prompt(prompt: str, *, anime: bool = False) -> str:
    """按模型家族转写高质量提示词：二次元走 Danbooru 标签 + 质量词，默认走自然语言描述。
    失败降级：原文 + 质量后缀（二次元）/ 原文（默认）。"""
    if not prompt or len(prompt) > _EXPAND_MAX_LEN:
        return prompt
    try:
        from junjun_llm import get_chat_model
        model = get_chat_model("utils_small")
        template = _EXPAND_PROMPT_ANIME if anime else _EXPAND_PROMPT_DEFAULT
        resp = await model.ainvoke([HumanMessage(content=template.format(prompt=prompt))])
        expanded = (resp.content or "").strip().strip('"').replace("\n", " ")
        if expanded:
            if anime:
                # Danbooru 标签串 + 质量后缀（不混入中文原文，保持标签纯净；
                # 防御性去重：LLM 偶尔仍会带出质量词，与后缀重复的剔除）
                suffix_tags = {t.strip() for t in _ANIME_QUALITY_SUFFIX.split(",")}
                tags = [t.strip() for t in expanded[:600].split(",") if t.strip()]
                tags = [t for t in dict.fromkeys(tags) if t not in suffix_tags]
                return ", ".join(tags) + f", {_ANIME_QUALITY_SUFFIX}"
            # Z-Image 中英双语：中文原文前置保主体，英文细描随后
            return f"{prompt}，{expanded[:600]}"
    except Exception as e:
        logger.warning(f"prompt 扩写失败（降级）: {type(e).__name__}: {e}")
    if anime:
        return f"{prompt}, {_ANIME_QUALITY_SUFFIX}"
    return prompt


async def submit_task(prompt: str, model: str, negative: str = "") -> str | None:
    """提交 ModelScope 异步生图任务，返回 task_id；任何失败返回 None。
    negative：负面提示词；模型不支持该参数（HTTP 400）时自动去掉重试。"""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            payload = {"model": model, "prompt": prompt}
            if negative:
                payload["negative_prompt"] = negative
            resp = await client.post(
                f"{_API_BASE}/v1/images/generations",
                headers={**_headers(), "X-ModelScope-Async-Mode": "true"},
                json=payload,
            )
            if resp.status_code == 400 and negative:
                logger.info("模型不接受 negative_prompt，去掉后重试")
                payload.pop("negative_prompt")
                resp = await client.post(
                    f"{_API_BASE}/v1/images/generations",
                    headers={**_headers(), "X-ModelScope-Async-Mode": "true"},
                    json=payload,
                )
            if resp.status_code != 200:
                logger.warning(f"ModelScope 提交任务失败 HTTP {resp.status_code}")
                return None
            task_id = resp.json().get("task_id")
            if not task_id:
                logger.warning("ModelScope 未返回 task_id")
                return None
            return str(task_id)
    except Exception as e:
        logger.warning(f"ModelScope 提交任务异常: {type(e).__name__}: {e}")
        return None


async def poll_task(task_id: str) -> str | None:
    """轮询任务状态（间隔 5s，总超时 120s），成功返回图片 URL；失败/超时返回 None。"""
    deadline = time.monotonic() + _POLL_TIMEOUT
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            while time.monotonic() < deadline:
                await asyncio.sleep(_POLL_INTERVAL)
                resp = await client.get(
                    f"{_API_BASE}/v1/tasks/{task_id}",
                    headers={**_headers(), "X-ModelScope-Task-Type": "image_generation"},
                )
                data = resp.json()
                status = data.get("task_status")
                if status == "SUCCEED":
                    urls = data.get("output_images") or []
                    if urls:
                        return str(urls[0])
                    logger.warning("ModelScope 任务成功但未返回图片 URL")
                    return None
                if status == "FAILED":
                    logger.warning(f"ModelScope 生成失败: {str(data)[:200]}")
                    return None
    except Exception as e:
        logger.warning(f"ModelScope 轮询异常: {type(e).__name__}: {e}")
    return None


async def _generate_aiping(prompt: str, model: str) -> str | None:
    """AI Ping 同步生图：POST /images/generations，返回 data[0].url；失败 None。

    OpenAI 风格同步协议（与 ModelScope 异步任务不同）；negative_prompt 不支持。
    """
    if not _aiping_key() or not _aiping_base():
        logger.warning("AI Ping 生图未配置 AIPING_API_KEY/AIPING_BASE_URL")
        return None
    payload = {
        "model": model,
        "prompt": prompt,
        "size": os.environ.get("AI_DRAW_AIPING_SIZE", "").strip() or "1024x1024",
    }
    headers = {"Authorization": f"Bearer {_aiping_key()}",
               "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=_POLL_TIMEOUT) as client:
            resp = await client.post(f"{_aiping_base()}/images/generations",
                                     json=payload, headers=headers)
        if resp.status_code != 200:
            logger.warning(f"AI Ping 生图失败 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = (resp.json().get("data") or [{}])
        url = data[0].get("url") if data else None
        if not url:
            logger.warning(f"AI Ping 生图未返回 url: {resp.text[:200]}")
            return None
        return str(url)
    except Exception as e:
        logger.warning(f"AI Ping 生图异常: {type(e).__name__}: {e}")
        return None


async def generate(prompt: str, model: str, negative: str = "") -> str | None:
    """完整生图链路：AI Ping 模型走同步直连，ModelScope 模型走提交->轮询；失败 None。"""
    if model in _aiping_model_ids():
        return await _generate_aiping(prompt, model)
    task_id = await submit_task(prompt, model, negative)
    if not task_id:
        return None
    return await poll_task(task_id)


async def _draw_pipeline(prompt: str, model_alias: str = "",
                         _reviewed: bool = False) -> tuple[str | None, str]:
    """通用链路：人设注入 -> 提示词工作室（写手+评审）-> 生图 -> (可选)VLM验收重画。

    提示词策略按模型家族：anime 走 Danbooru 标签（expand_prompt），
    zimage/qwen 走 prompt_studio 的中文结构化写手；工作室失败降级旧扩写。
    """
    from .prompt_studio import craft_prompt, review_image, _cfg
    final_prompt = apply_self_prompt(prompt)
    model = route_model(final_prompt, model_alias)
    anime = model_style(model) == "anime"
    if anime:
        final_prompt = await expand_prompt(final_prompt, anime=True)
    else:
        family = "qwen" if model == _model_registry()["qwen"] else "zimage"
        try:
            crafted = await craft_prompt(final_prompt, family)
        except Exception as e:
            logger.warning(f"提示词工作室异常（降级旧扩写）: {type(e).__name__}: {e}")
            crafted = ""
        final_prompt = crafted or await expand_prompt(final_prompt, anime=False)
    negative = _ANIME_NEGATIVE if anime else _DEFAULT_NEGATIVE
    url = await generate(final_prompt, model, negative)
    # VLM 出图验收（[ai_draw] review_enable，默认关）：严重不符带意见重画一次
    if url and not _reviewed and bool(_cfg().get("review_enable", False)):
        issue = await review_image(url, prompt)
        if issue:
            logger.info(f"出图验收不通过，带意见重画一次: {issue[:50]}")
            return await _draw_pipeline(
                f"{prompt}。上一稿的问题：{issue}，这次修正", model_alias,
                _reviewed=True)
    return url, final_prompt


def _parse_model_alias(args: str) -> tuple[str, str]:
    """从命令参数尾部解析显式模型别名：/draw 猫娘 qwen -> ("猫娘", "qwen")。"""
    tokens = (args or "").split()
    if tokens and tokens[-1].lower() in _model_registry():
        return " ".join(tokens[:-1]).strip(), tokens[-1].lower()
    return (args or "").strip(), ""


@register_command("draw", aliases=["绘图", "画图"], plugin="ai_draw",
                  description="AI画图：/draw <描述> [模型]，含动漫/二次元自动切换二次元模型")
async def draw_cmd(ctx):
    """手动画图命令：提交即回「在画了」，后台画完直发图片，绝不抛异常。"""
    prompt, model_alias = _parse_model_alias(ctx.args)
    if not prompt:
        return ("要画什么呀？用法：/draw <描述> [模型]，比如 /draw 猫娘少女\n"
                "模型可选：kolors（默认，免费）/ anime（二次元）/ qwen（写实/带字图最强）/ "
                "glm-image、seedream（AI Ping 付费高质量）/ zimage（旧默认），不填按描述自动路由。")
    if is_minor_nsfw(prompt):
        return "这种不行哦，涉及未成年人的色色内容君君绝对不画！换个描述吧。"
    # 群聊 R18 硬门（2026-08-06 实锤「分不清群聊私聊」：群场景此前只靠模型
    # 自觉，命令/工具层无兜底——/draw 涩图 xxx 在群里真的会派单出图）
    if ctx.session.is_group and has_r18_marker(prompt):
        return "这种图群里不画哦，人多眼杂 + 账号风控。私聊我，悄悄给你画。"

    chat_id = ctx.session.chat_id
    now = time.time()
    left = _COOLDOWN - (now - _last_use.get(chat_id, 0))
    if left > 0:
        return f"画得太勤啦，{int(left) + 1} 秒后再来吧。"

    if not _any_provider_key():
        return "画图功能还没配置密钥喵，让主人设置 AIPING_API_KEY 或 MODELSCOPE_API_KEY 吧。"

    _last_use[chat_id] = now
    fut = _begin_pending_draw(chat_id)
    ack = await task_manager.submit(
        kind="ai_draw",
        work=lambda: _draw_work(prompt, chat_id, fut, model_alias),
        done_text=f"画好啦！{prompt}",
        fail_text="画图失败了，稍后再试试吧。",
        timeout=_POLL_TIMEOUT + 60,
        chat_id=chat_id,
        context=prompt,
    )
    return ack


# 「画图发空间」防重复接力：ai_draw 是异步的（提交即返回，后台画 1~2 分钟），
# 若 Agent 紧接着调 send_feed(with_image=True)，此时图多半还在画。
# 所以 send_feed 侧复用顺序 = ① 等本会话【进行中】的画图（_PENDING）
# ② 30 秒短窗口内刚完成的图（_LAST_DRAWN，防止画图刚好完成 pending 已弹出）。
# 窗口刻意只有 30 秒：更早的图属于上一轮对话，绝不复用（避免张冠李戴）。
_LAST_DRAWN: dict[str, tuple[float, str]] = {}   # chat_id -> (完成时间戳, url)
_PENDING: dict[str, asyncio.Future] = {}          # chat_id -> 进行中的画图 Future
_LAST_DRAWN_TTL = 30.0    # 完成图的复用窗口（秒）
_WAIT_DRAWN_TIMEOUT = 150.0  # 等待进行中画图的上限（秒）


def _begin_pending_draw(chat_id: str) -> "asyncio.Future | None":
    """登记一次进行中的画图（提交任务前调用），返回 Future 由 _draw_work 回填。"""
    if not chat_id:
        return None
    fut = asyncio.get_running_loop().create_future()
    _PENDING[chat_id] = fut
    return fut


async def wait_recent_drawn_url(chat_id: str, timeout: float = _WAIT_DRAWN_TIMEOUT) -> str | None:
    """给发说说等「二次用图」方：等本会话正在画的图 / 拿 30s 内刚画好的图；没有返回 None。"""
    if not chat_id:
        return None
    fut = _PENDING.get(chat_id)
    if fut is not None:
        try:
            url = await asyncio.wait_for(asyncio.shield(fut), timeout)
            if url:
                return url
        except Exception:
            pass
    item = _LAST_DRAWN.get(chat_id)
    if item and time.time() - item[0] <= _LAST_DRAWN_TTL:
        return item[1]
    return None


async def _draw_work(prompt: str, chat_id: str = "",
                     fut: "asyncio.Future | None" = None,
                     model_alias: str = "") -> list | None:
    """后台生图：成功返回 [image 段]，失败返回 None（由任务管理器发降级文案）。"""
    try:
        url, _ = await _draw_pipeline(prompt, model_alias)
        if url and chat_id:
            _LAST_DRAWN[chat_id] = (time.time(), url)
        if fut is not None and not fut.done():
            fut.set_result(url)
        return [ReplySegment(type="image", data=url)] if url else None
    finally:
        if fut is not None and not fut.done():
            fut.set_result(None)
        if chat_id:
            _PENDING.pop(chat_id, None)


@tool
async def ai_draw(prompt: str, model: str = "") -> str:
    """根据描述 AI 生成图片并发到当前聊天。当用户要求画图、画个xxx、帮我画、来张图时使用。

    【重要】如果用户要求把图发到 QQ 空间说说（「画好发到空间」「画一张发说说」），
    绝对不要调本工具——直接调 send_feed(with_image=True)，它会自己画图一起发。
    两个都调会画出两张不同的图，双倍成本还货不对板（2026-08-09 eval 实锤：
    警告藏在描述中段时弱模型照犯，故置顶）。

    prompt 为画面描述（如「猫娘少女」「星空下的城市」）。
    model 为可选模型别名：kolors（默认通用，免费）/ anime（二次元）/ qwen（写实照片、
    海报贺卡等带文字的图最强）/ glm-image、seedream（AI Ping 付费高质量）——不填按描述自动路由。

    本工具是异步的：调用后立即返回，图片画好会自动发到当前聊天，
    不要在回复里编造图片 URL，也不要说「无法发送图片」——图片会随后发出。"""
    prompt = (prompt or "").strip()
    if not prompt:
        return "没有描述词，画不了。"
    if is_minor_nsfw(prompt):
        return "拒绝：描述涉及未成年人性内容，不会生成。"
    if not _any_provider_key():
        return "画图功能未配置生图密钥（AIPING_API_KEY 或 MODELSCOPE_API_KEY），暂时画不了。"
    model_alias = (model or "").strip().lower()
    if model_alias and model_alias not in _model_registry():
        return f"不认识模型「{model}」，可选：kolors / anime / qwen / glm-image / seedream / zimage。"
    from junjun_skills.builtin.memory_skills import current_chat_id
    chat_id = current_chat_id.get("")
    # 群聊 R18 硬门：模型在群里违规调工具也不能真出图（最后一道防线）。
    # 返回引导文案让模型照着婉拒——比空拒绝更像人，也不会占派单位。
    # 宁可误拒：路由缺失（chat_id 空）的边缘场景按群聊处理（2026-08-06 审查：
    # 空路由会跳过群门落到同步出图分支）——只有显式私聊才放行。
    if has_r18_marker(prompt) and not chat_id.endswith(":private"):
        return ("群里画不了这种（公共场合 + 账号风控）。"
                "笑着让对方私聊你——照这个意思回他，别派单。")
    if not chat_id:
        # 无会话路由（边缘场景）：同步生成 + [IMAGE:] 标记，由 processor 提取发图
        url, _ = await _draw_pipeline(prompt, model_alias)
        return f"[IMAGE:{url}]" if url else "画图失败了，稍后再试。"
    # 在途防重（2026-08-04 实战：用户问「图呢」，模型又调一次 ai_draw 重复派画，
    # 然后用「在画了」搪塞——/draw 命令有 20s 冷却，工具路径此前没有）
    if chat_id in _PENDING:
        return "上一张还在画，画好会自动发到当前聊天，不要重复派单。" \
               "如果对方在催，告诉 ta 还在画；超过两三分钟还没收到再来找你。"
    fut = _begin_pending_draw(chat_id)
    return await task_manager.submit(
        kind="ai_draw",
        work=lambda: _draw_work(prompt, chat_id, fut, model_alias),
        fail_text="这次画失败了，再试一次？",
        timeout=_POLL_TIMEOUT + 60,
        context=prompt,
    )


TOOLS = [ai_draw]


# ---------------- 私聊涩图直通道（2026-08-04） ----------------
# 背景：agent 槽模型（Qwen3.5 + 硅基合规层）面对 NSFW 请求会输出空内容自我
# 审查——实测四种死法（ playful 答应不调工具 / 文本拒绝 / 调 do_not_reply /
# 空内容），rules 授权 + 手册 + 意图自检追问全试过，模型就是不从。
# 无歧义的私聊涩图请求是内容判断问题不是智能问题：绕过 LLM 直接派画，
# 与 /draw 命令同链路（红线/冷却/提交全复用）。群聊不拦截，交给 LLM 婉拒。
# 「色的」只匹配句尾（「搞点色的」「来张色的」）——「蓝色的天空」「角色的立绘」
# 这类颜色/角色词不再被劫持（严厉审查 P1-6 假阳性）
_NSFW_DRAW_RE = r"涩图|色图|r18|nsfw|(画|生成|来[一张幅个])[^\n]{0,30}(涩|色色|色的?[。!~？\s]*$)"
_NSFW_NEGATIONS = ("别画", "别发", "不要画", "不许画", "不准画", "不能画", "不要发",
                   "先别画", "这次算了", "不用画")


@register_interceptor(_NSFW_DRAW_RE, name="nsfw_draw", plugin="ai_draw", priority=-10)
async def nsfw_draw_hit(ctx) -> bool:
    """私聊涩图请求直通：绕过 LLM 内容审查直接走 /draw 同链路。"""
    if ctx.session.is_group:
        return False  # 群里交给 LLM 按手册婉拒（公共场合 + 风控）
    text = (ctx.meta.text or "").strip()
    if any(neg in text for neg in _NSFW_NEGATIONS):
        return False  # 「别画涩图了」是制止不是请求，交给 LLM
    from junjun_agent.commands import CommandContext
    reply_text = await draw_cmd(CommandContext(
        session=ctx.session, meta=ctx.meta, args=text))
    await ctx.reply(reply_text)
    return True
