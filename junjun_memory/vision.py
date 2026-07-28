"""VLM 入站识图（阶段 3）：图片 -> 描述 -> Images 表（hash 去重）-> 上下文注入。

对齐原 utils_image.py 语义：
- 图片下载 -> md5 查 Images 表，命中直接用缓存描述（省 VLM 调用）
- 未命中调 task.vlm 槽模型描述，结果入库
- 15s 超时 + 失败降级 "[图片]"，不阻塞回复
- task.vlm 未配置（VLM_* env 缺）时全链路静默降级
"""

import base64
import hashlib
import time
from typing import Dict, List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.vision")

_DESCRIBE_PROMPT = "用一句中文口语描述这张图片的内容（20字以内，像跟朋友转述一样）。"
_STICKER_PROMPT = "这是一张 QQ 聊天表情包。用一句中文口语描述画面和它表达的情绪（20字以内，如「猫咪竖大拇指表示赞同」）。"
_TIMEOUT = 15.0


def _get_vlm():
    try:
        from junjun_llm import get_chat_model
        return get_chat_model("vlm")
    except Exception:
        return None


async def _download(url: str) -> Optional[bytes]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        logger.debug(f"图片下载失败: {e}")
        return None


async def _describe(data: bytes, *, model, prompt: str = _DESCRIBE_PROMPT) -> Optional[str]:
    import asyncio
    from langchain_core.messages import HumanMessage
    b64 = base64.b64encode(data).decode()
    try:
        resp = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ])]),
            timeout=_TIMEOUT,
        )
        return str(resp.content).strip() or None
    except Exception as e:
        logger.warning(f"VLM 识图失败（降级占位）: {e}")
        return None


async def _describe_one(url: str, *, model, prompt: str, placeholder: str) -> str:
    """单张图：下载 -> md5 查缓存 -> VLM -> 入库。失败返回占位。"""
    data = await _download(url)
    if data is None:
        return placeholder
    h = hashlib.md5(data).hexdigest()
    from junjun_core.database import Images
    row = Images.get_or_none(Images.image_hash == h)
    if row is not None and row.description:
        return row.description
    if model is None:
        return placeholder
    desc = await _describe(data, model=model, prompt=prompt)
    if desc:
        try:
            Images.create(image_hash=h, description=desc, timestamp=time.time())
        except Exception as e:
            logger.debug(f"图片描述入库失败（忽略）: {e}")
    return desc or placeholder


async def describe_images(image_urls: List[str], *, model=None) -> Dict[str, str]:
    """批量识图：url -> 描述。失败/未配置时该 url 映射为 "[图片]" 占位。"""
    if not image_urls:
        return {}
    if model is None:
        model = _get_vlm()
    return {url: await _describe_one(url, model=model, prompt=_DESCRIBE_PROMPT,
                                     placeholder="[图片]") for url in image_urls}


async def describe_stickers(sticker_urls: List[str], *, model=None) -> Dict[str, str]:
    """批量识表情包：url -> 「画面+情绪」描述。与普通图片共用 Images hash 缓存
    （同一张表情包只花一次 VLM 调用）。失败映射为 "[表情]" 占位。"""
    if not sticker_urls:
        return {}
    if model is None:
        model = _get_vlm()
    return {url: await _describe_one(url, model=model, prompt=_STICKER_PROMPT,
                                     placeholder="[表情]") for url in sticker_urls}


def render_image_block(descriptions: Dict[str, str]) -> str:
    """渲染进上下文：对方发了一张图片：描述。"""
    descs = [d for d in descriptions.values() if d and d != "[图片]"]
    if not descs:
        return ""
    if len(descs) == 1:
        return f"对方发了一张图片：{descs[0]}"
    return "对方发了图片：\n" + "\n".join(f"- {d}" for d in descs)


def render_sticker_block(descriptions: Dict[str, str]) -> str:
    """渲染进上下文：对方发了一个表情包：画面+情绪。"""
    descs = [d for d in descriptions.values() if d and d != "[表情]"]
    if not descs:
        return ""
    if len(descs) == 1:
        return f"对方发了一个表情包：{descs[0]}"
    return "对方发了表情包：\n" + "\n".join(f"- {d}" for d in descs)
