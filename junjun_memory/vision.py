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


async def _describe(data: bytes, *, model) -> Optional[str]:
    import asyncio
    from langchain_core.messages import HumanMessage
    b64 = base64.b64encode(data).decode()
    try:
        resp = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=[
                {"type": "text", "text": _DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ])]),
            timeout=_TIMEOUT,
        )
        return str(resp.content).strip() or None
    except Exception as e:
        logger.warning(f"VLM 识图失败（降级占位）: {e}")
        return None


async def describe_images(image_urls: List[str], *, model=None) -> Dict[str, str]:
    """批量识图：url -> 描述。失败/未配置时该 url 映射为 "[图片]" 占位。"""
    out: Dict[str, str] = {}
    if not image_urls:
        return out
    if model is None:
        model = _get_vlm()
    from junjun_core.database import Images
    for url in image_urls:
        desc: Optional[str] = None
        data = await _download(url)
        if data is not None:
            h = hashlib.md5(data).hexdigest()
            row = Images.get_or_none(Images.image_hash == h)
            if row is not None:
                desc = row.description or None
            elif model is not None:
                desc = await _describe(data, model=model)
                if desc:
                    try:
                        Images.create(image_hash=h, description=desc, timestamp=time.time())
                    except Exception as e:
                        logger.debug(f"图片描述入库失败（忽略）: {e}")
        out[url] = desc or "[图片]"
    return out


def render_image_block(descriptions: Dict[str, str]) -> str:
    """渲染进上下文：对方发了一张图片：描述。"""
    descs = [d for d in descriptions.values() if d and d != "[图片]"]
    if not descs:
        return ""
    if len(descs) == 1:
        return f"对方发了一张图片：{descs[0]}"
    return "对方发了图片：\n" + "\n".join(f"- {d}" for d in descs)
