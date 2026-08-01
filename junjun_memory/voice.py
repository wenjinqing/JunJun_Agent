"""语音消息理解：record 段 -> NapCat get_record 转 mp3 -> SF ASR -> 转写注入。

与 vision 同构：入站即预热（prewarm_voice）+ 决策前有界等待（transcribe_voices），
同一语音的 in-flight 任务共享（预热与回复路径只转写一次）；
失败降级 "[语音]" 占位，不阻塞回复。

QQ 语音原始是 silk 编码，ASR 不吃——走 NapCat get_record（out_format=mp3）
服务端转码；NapCat 与 bot 同机部署，返回本地路径直接读。
"""

import asyncio
from typing import Dict, List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("memory.voice")

_FETCH_TIMEOUT = 20.0
_ASR_TIMEOUT = 60.0   # 语音消息最长也就一两分钟，转写很快

_PENDING: Dict[str, asyncio.Task] = {}  # 同一语音的 in-flight 转写任务（全局共享）


def _enabled() -> bool:
    try:
        return bool(get_global_config().raw.get("voice", {}).get("enable", True))
    except Exception:
        return True


# ---------------------------------------------------------------- 音频获取

async def _fetch_audio(ref: str) -> Optional[bytes]:
    """语音引用 -> mp3 字节。http 直链直接下；否则 NapCat get_record 转码。"""
    try:
        if ref.startswith("http"):
            import httpx
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT,
                                         follow_redirects=True) as client:
                resp = await client.get(ref)
                resp.raise_for_status()
                return resp.content
        # file id：NapCat get_record 服务端转 mp3（silk -> mp3）
        from junjun_adapter_napcat.send_handler.nc_sending import nc_message_sender
        resp = await nc_message_sender.send_message_to_napcat(
            "get_record", {"file": ref, "out_format": "mp3"})
        data = (resp or {}).get("data") or {}
        if data.get("base64"):
            import base64
            return base64.b64decode(data["base64"])
        path = data.get("file") or data.get("path") or ""
        if path:
            from pathlib import Path
            return Path(path).read_bytes()
    except Exception as e:
        logger.debug(f"语音获取失败 {ref[:40]}: {type(e).__name__}: {e}")
    return None


# ---------------------------------------------------------------- 转写（共享在途）

async def _transcribe_one(ref: str) -> str:
    audio = await _fetch_audio(ref)
    if not audio:
        return "[语音]"
    from junjun_llm.asr import transcribe_bytes
    text = await transcribe_bytes(audio, suffix=".mp3", timeout=_ASR_TIMEOUT)
    return text or "[语音]"


def _shared_task(ref: str) -> asyncio.Task:
    task = _PENDING.get(ref)
    if task is None or task.done():
        task = asyncio.create_task(_transcribe_one(ref))
        _PENDING[ref] = task
        task.add_done_callback(lambda _t, r=ref: _PENDING.pop(r, None))
    return task


def prewarm_voice(records: List[str]) -> None:
    """消息入站即后台转写（不管是否 @bot）：发语音 -> 再 @君君 的场景就绪。"""
    if not records or not _enabled():
        return
    try:
        for ref in records[:3]:  # 单条消息最多 3 条语音
            _shared_task(ref)
    except Exception as e:
        logger.debug(f"语音预热失败（忽略）: {e}")


def _perception_wait() -> float:
    try:
        return float(get_global_config().raw.get("perception", {}).get("ready_wait_seconds", 3.0))
    except Exception:
        return 3.0


async def transcribe_voices(records: List[str], *, wait: Optional[float] = None) -> List[str]:
    """批量转写（并行 + 在途共享 + 有界等待）。超时未完成的降级 "[语音]" 占位，
    在途任务不取消——结果留在 _PENDING 直到完成，同语音下次命中。
    """
    if not records or not _enabled():
        return []
    tasks = [_shared_task(ref) for ref in records[:3]]
    if wait is None:
        wait = _perception_wait()
    if wait > 0:
        await asyncio.wait(tasks, timeout=wait)
    else:
        await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for t in tasks:
        if t.done() and not t.cancelled() and t.exception() is None:
            out.append(t.result())
        else:
            out.append("[语音]")
    return out


def render_voice_block(texts: List[str]) -> str:
    """渲染进上下文：对方发来一条语音，说的是：…"""
    good = [t for t in texts if t and t != "[语音]"]
    if not good:
        return ""
    if len(good) == 1:
        return f"对方发来一条语音，说的是：「{good[0]}」"
    return "对方发来语音：\n" + "\n".join(f"- 「{t}」" for t in good)
