"""VLM 入站识图（阶段 3）：图片 -> 描述 -> Images 表（hash 去重）-> 上下文注入。

对齐原 utils_image.py 语义：
- 图片下载 -> md5 查 Images 表，命中直接用缓存描述（省 VLM 调用）
- 未命中调 task.vlm 槽模型描述，结果入库
- 下载 20s / 描述 30s 超时 + 失败降级 "[图片]"，不阻塞回复
- task.vlm 未配置（VLM_* env 缺）时全链路静默降级

2026-07-29 竞态修复（用户反馈「图还没识别 Agent 就回复了」）：
- prewarm_images：消息入站（不管是否 @bot）即后台启动识图，
  回复路径命中缓存/共享在途任务——「发图 → 再 @君君看」场景不再看不到图
- describe_image_shared：同一 url 的 in-flight 任务全局共享，不重复调 VLM
- 多张图并行识图（原串行）；VLM 调用并发限流（semaphore 3）

2026-07-31 P0-14 感知就绪等待：
- describe_images/describe_stickers 带就绪上限（[perception] ready_wait_seconds，
  默认 3s）：到点没描述完的图降级占位，决策不被 VLM 慢调用拖住；
  在途任务不取消——结果照常入 Images 缓存，下一条消息直接命中。
"""

import asyncio
import base64
import hashlib
import time
from collections import deque
from typing import Dict, List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.vision")

_DESCRIBE_PROMPT = "用一句中文口语描述这张图片的内容（20字以内，像跟朋友转述一样）。"
_STICKER_PROMPT = "这是一张 QQ 聊天表情包。用一句中文口语描述画面和它表达的情绪（20字以内，如「猫咪竖大拇指表示赞同」）。"
_DOWNLOAD_TIMEOUT = 20.0
_DESCRIBE_TIMEOUT = 30.0

# 同一 url 的 in-flight 识图任务（预热与回复路径共享，防重复 VLM 调用）
_PENDING: Dict[str, asyncio.Task] = {}
# VLM 并发限流（群图密集时防打爆槽位）
_VLM_SEM: Optional[asyncio.Semaphore] = None

# 每会话最近收到的图片（chat_id -> deque[(ts, kind, url)]）：回复时补充描述
_RECENT: Dict[str, deque] = {}
_RECENT_TTL = 600.0     # 最近 10 分钟内的图才算「刚发的」
_RECENT_MAX = 5         # 注入上限（防上下文膨胀）


def _get_vlm():
    try:
        from junjun_llm import get_chat_model
        return get_chat_model("vlm")
    except Exception:
        return None


async def _download(url: str) -> Optional[bytes]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        logger.debug(f"图片下载失败: {e}")
        return None


def _vlm_sem() -> asyncio.Semaphore:
    global _VLM_SEM
    if _VLM_SEM is None:
        _VLM_SEM = asyncio.Semaphore(3)
    return _VLM_SEM


async def _describe(data: bytes, *, model, prompt: str = _DESCRIBE_PROMPT) -> Optional[str]:
    from langchain_core.messages import HumanMessage
    from junjun_core.retry import retry_async
    b64 = base64.b64encode(data).decode()

    async def _call():
        async with _vlm_sem():
            return await asyncio.wait_for(
                model.ainvoke([HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ])]),
                timeout=_DESCRIBE_TIMEOUT,
            )

    try:
        # 瞬态失败（限流/网络抖动）重试 3 次再降级占位
        resp = await retry_async(_call, attempts=3, base_delay=1.0, label="vlm.describe")
        return str(resp.content).strip() or None
    except Exception as e:
        logger.warning(f"VLM 识图重试 3 次均失败（降级占位）: {e}")
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


def describe_image_shared(url: str, *, model, prompt: str, placeholder: str) -> asyncio.Task:
    """同一 url 的 in-flight 识图任务全局共享：预热和回复路径只算一次。"""
    task = _PENDING.get(url)
    if task is None or task.done():
        task = asyncio.create_task(
            _describe_one(url, model=model, prompt=prompt, placeholder=placeholder))
        _PENDING[url] = task
        task.add_done_callback(lambda _t, u=url: _PENDING.pop(u, None))
    return task


def prewarm_images(chat_id: str, image_urls: List[str], sticker_urls: List[str]) -> None:
    """消息入站即后台识图（不管是否 @bot）：「发图 -> 再 @君君看」场景
    等 Agent 被叫到时描述已就绪（或至少已在途，回复路径 await 同一任务）。"""
    urls = [(u, "image") for u in (image_urls or [])] + \
           [(u, "sticker") for u in (sticker_urls or [])]
    if not urls:
        return
    try:
        model = _get_vlm()
        dq = _RECENT.setdefault(chat_id, deque(maxlen=_RECENT_MAX * 4))
        now = time.time()
        for url, kind in urls:
            dq.append((now, kind, url))
            try:
                describe_image_shared(
                    url, model=model,
                    prompt=_STICKER_PROMPT if kind == "sticker" else _DESCRIBE_PROMPT,
                    placeholder="[表情]" if kind == "sticker" else "[图片]")
            except Exception as e:  # 无事件循环等场景：记录已入队，单张失败不影响其他
                logger.debug(f"图片预热任务创建失败（忽略）: {e}")
    except Exception as e:
        logger.debug(f"图片预热失败（忽略）: {e}")


def recent_image_urls(chat_id: str, ttl: float = _RECENT_TTL) -> List[tuple]:
    """该会话最近 ttl 秒内收到过的图片 [(kind, url)]，新在前，上限 _RECENT_MAX。"""
    now = time.time()
    out = [(kind, u) for ts, kind, u in _RECENT.get(chat_id, ()) if now - ts <= ttl]
    out.reverse()
    return out[:_RECENT_MAX]


def _perception_wait() -> float:
    """决策前等待在途识图的上限秒数（[perception] ready_wait_seconds，默认 3）。"""
    try:
        from junjun_core.config import get_global_config
        return float(get_global_config().raw.get("perception", {}).get("ready_wait_seconds", 3.0))
    except Exception:
        return 3.0


async def _gather_bounded(urls: List[str], tasks: List[asyncio.Task],
                          placeholder: str, wait: float) -> Dict[str, str]:
    """有界等待在途识图：到 wait 秒未完成的 url 降级占位。

    任务【不取消】——识图结果照常写 Images 缓存，下一条消息直接命中，
    本次只是「决策等不起」而不是「放弃这张图」。
    wait <= 0 表示全部等完（旧行为）。
    """
    if wait > 0:
        await asyncio.wait(tasks, timeout=wait)
    else:
        await asyncio.gather(*tasks, return_exceptions=True)
    out = {}
    for url, t in zip(urls, tasks):
        if t.done() and not t.cancelled() and t.exception() is None:
            out[url] = t.result()
        else:
            out[url] = placeholder
    return out


async def describe_images_full(image_urls: List[str], *, model=None,
                               wait: Optional[float] = None) -> tuple:
    """完整版：返回 (url->描述, 仍在途的任务列表)。

    「还在看」与「没有图」必须可区分——决策窗内没看完的图，Agent 应该
    说「我在看」而不是「看不到」（2026-08-02 生产反馈）。在途任务交给
    调用方决定要不要等完成后主动补一句。
    """
    if not image_urls:
        return {}, []
    if model is None:
        model = _get_vlm()
    tasks = [describe_image_shared(u, model=model, prompt=_DESCRIBE_PROMPT,
                                   placeholder="[图片]") for u in image_urls]
    if wait is None:
        wait = _perception_wait()
    out = await _gather_bounded(image_urls, tasks, "[图片]", wait)
    pending = [t for t in tasks if not t.done()]
    return out, pending


async def describe_images(image_urls: List[str], *, model=None,
                          wait: Optional[float] = None) -> Dict[str, str]:
    """批量识图（并行 + 在途共享）：url -> 描述。失败/未配置映射为 "[图片]" 占位。

    wait：就绪等待上限秒数；None 走配置 [perception] ready_wait_seconds（默认 3s），
    <=0 表示全部等完。超时未完成的图降级占位但在途任务继续跑（结果入缓存）。
    """
    out, _ = await describe_images_full(image_urls, model=model, wait=wait)
    return out


async def describe_stickers(sticker_urls: List[str], *, model=None,
                            wait: Optional[float] = None) -> Dict[str, str]:
    """批量识表情包（并行 + 在途共享）：url -> 「画面+情绪」描述。
    与普通图片共用 Images hash 缓存（同一张表情包只花一次 VLM 调用）。
    wait 语义同 describe_images。
    """
    if not sticker_urls:
        return {}
    if model is None:
        model = _get_vlm()
    tasks = [describe_image_shared(u, model=model, prompt=_STICKER_PROMPT,
                                   placeholder="[表情]") for u in sticker_urls]
    if wait is None:
        wait = _perception_wait()
    return await _gather_bounded(sticker_urls, tasks, "[表情]", wait)


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
