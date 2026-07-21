"""ai_draw 插件：AI 文生图（迁移自旧 ai_draw_plugin，新架构重写）。

命令：/draw <描述>（/绘图 /画图 同）
工具：ai_draw（LLM 自动配图）
API：ModelScope 异步文生图（api-inference.modelscope.cn）
  - POST /v1/images/generations（头 X-ModelScope-Async-Mode: true）提交任务 -> task_id
  - GET  /v1/tasks/{task_id} 轮询（间隔 5s，总超时 120s）-> output_images[0]
模型路由：描述含 动漫/二次元/anime 等词时用二次元模型，否则默认模型；
  env AI_DRAW_MODEL / AI_DRAW_MODEL_ANIME 可覆盖默认值。
安全：描述命中「未成年词 + 性词」组合直接拒绝；未配置 MODELSCOPE_API_KEY 降级文本。
限流：每会话 20 秒最小间隔（内存 dict）。
"""

import asyncio
import os
import time

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.ai_draw")

_API_BASE = os.environ.get("AI_DRAW_API_BASE", "https://api-inference.modelscope.cn")
_HTTP_TIMEOUT = 30.0
_POLL_INTERVAL = 5.0    # 轮询间隔（秒）
_POLL_TIMEOUT = 120.0   # 轮询总超时（秒）
_COOLDOWN = 20.0        # 每会话最小间隔（秒）
_EXPAND_MIN_LEN = 20    # 描述短于该长度时用 LLM 扩写

# 默认生图模型（取自旧插件 config.toml，可用 env 覆盖）
_DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"
_DEFAULT_ANIME_MODEL = "QWQ114514123/WAI-illustrious-SDXL-v16"

# 内容红线：未成年词 与 性词 同时命中 -> 直接拒绝
_MINOR_WORDS = ("萝莉", "幼女", "小学生", "儿童", "幼童", "女童", "男童", "未成年",
                "underage", "preteen", "child", "loli")
_NSFW_WORDS = ("色情", "裸", "sex", "涩情", "裸体", "nsfw", "porn")

# 二次元/动漫画风词：命中则路由到二次元特化模型
_ANIME_WORDS = ("动漫", "二次元", "anime", "漫画", "番", "manga")

# 「画自己」触发词：命中则把人设词附加到 prompt
_SELF_WORDS = ("你", "自己", "自画像", "自拍")

# 每会话上次画图时间戳（chat_id -> ts）
_last_use: dict = {}


def _api_key() -> str:
    """每次调用实时读 env（便于测试与热更）。"""
    return os.environ.get("MODELSCOPE_API_KEY", "")


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


def is_anime(prompt: str) -> bool:
    """命中二次元/动漫画风词 -> True（路由到二次元特化模型）。"""
    low = (prompt or "").lower()
    return any(w.lower() in low for w in _ANIME_WORDS)


def route_model(prompt: str) -> str:
    """根据描述选择生图模型：二次元词 -> 二次元模型，否则默认模型。"""
    if is_anime(prompt):
        return os.environ.get("AI_DRAW_MODEL_ANIME", "") or _DEFAULT_ANIME_MODEL
    return os.environ.get("AI_DRAW_MODEL", "") or _DEFAULT_MODEL


def _get_persona() -> str:
    """从全局配置取人设词（personality 段），取一段作为「画自己」附加描述。"""
    try:
        from junjun_core.config import get_global_config
        raw = get_global_config().raw or {}
        text = str((raw.get("personality") or {}).get("personality") or "")
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


async def expand_prompt(prompt: str) -> str:
    """短描述（<20 字）用 utils_small 扩写为英文生图提示词；失败降级用原文。"""
    if len(prompt or "") >= _EXPAND_MIN_LEN:
        return prompt
    try:
        from junjun_llm import get_chat_model
        model = get_chat_model("utils_small")
        ask = (
            "你是AI绘画提示词助手。把用户给的中文主体扩写成高质量英文生图提示词："
            "绝不改变主体，只补充场景/光照/氛围/构图/画质标签（如 masterpiece, best quality），"
            "英文逗号分隔标签，15-35 个词，只输出提示词本身，不要解释。\n"
            f"主体：{prompt}"
        )
        resp = await model.ainvoke([HumanMessage(content=ask)])
        expanded = (resp.content or "").strip()
        if expanded:
            # 主体强制前置，确保主体绝不丢失/被替换
            return f"{prompt}, {expanded[:400]}"
    except Exception as e:
        logger.warning(f"prompt 扩写失败（降级用原文）: {type(e).__name__}: {e}")
    return prompt


async def submit_task(prompt: str, model: str) -> str | None:
    """提交 ModelScope 异步生图任务，返回 task_id；任何失败返回 None。"""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{_API_BASE}/v1/images/generations",
                headers={**_headers(), "X-ModelScope-Async-Mode": "true"},
                json={"model": model, "prompt": prompt},
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


async def generate(prompt: str, model: str) -> str | None:
    """完整生图链路：提交 -> 轮询 -> 图片 URL；任何失败返回 None。"""
    task_id = await submit_task(prompt, model)
    if not task_id:
        return None
    return await poll_task(task_id)


async def _draw_pipeline(prompt: str) -> tuple[str | None, str]:
    """通用链路：人设注入 -> 扩写 -> 路由 -> 生图。返回 (图片 URL 或 None, 最终 prompt)。"""
    final_prompt = apply_self_prompt(prompt)
    final_prompt = await expand_prompt(final_prompt)
    url = await generate(final_prompt, route_model(final_prompt))
    return url, final_prompt


@register_command("draw", aliases=["绘图", "画图"], plugin="ai_draw",
                  description="AI画图：/draw <描述>，含动漫/二次元自动切换二次元模型")
async def draw_cmd(ctx):
    """手动画图命令；所有失败路径降级为友好中文文本，绝不抛异常。"""
    prompt = (ctx.args or "").strip()
    if not prompt:
        return "要画什么呀？用法：/draw <描述>，比如 /draw 猫娘少女"
    if is_minor_nsfw(prompt):
        return "这种不行哦，涉及未成年人的色色内容君君绝对不画！换个描述吧。"

    chat_id = ctx.session.chat_id
    now = time.time()
    left = _COOLDOWN - (now - _last_use.get(chat_id, 0))
    if left > 0:
        return f"画得太勤啦，{int(left) + 1} 秒后再来吧。"

    if not _api_key():
        return "画图功能还没配置 ModelScope 密钥喵，让主人设置 MODELSCOPE_API_KEY 吧。"

    _last_use[chat_id] = now
    url, _ = await _draw_pipeline(prompt)
    if not url:
        return "画图失败了，稍后再试试吧。"

    await ctx.send([ReplySegment(type="text", data=f"画好啦！{prompt}"),
                    ReplySegment(type="image", data=url)])
    return None


@tool
async def ai_draw(prompt: str) -> str:
    """根据描述 AI 生成图片。当用户要求画图、画个xxx、帮我画、来张图、需要配图时使用。
    prompt 为画面描述（如「猫娘少女」「星空下的城市」）。"""
    prompt = (prompt or "").strip()
    if not prompt:
        return "没有描述词，画不了。"
    if is_minor_nsfw(prompt):
        return "拒绝：描述涉及未成年人性内容，不会生成。"
    if not _api_key():
        return "画图功能未配置 MODELSCOPE_API_KEY，暂时画不了。"
    url, _ = await _draw_pipeline(prompt)
    if not url:
        return "画图失败了，稍后再试。"
    return url


TOOLS = [ai_draw]
