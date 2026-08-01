"""视频文件感知：群友直接发的视频 -> 抽帧(VLM) + 抽音(ASR) -> 描述注入。

与 vision/voice 同构：入站预热 + 决策前有界等待 + in-flight 共享 + 结果缓存
（md5，1 小时——同一条视频被转发/复读只感知一次）。

感知比识图重得多（下载+ffmpeg+ASR 10-30s），3s 有界等待内多半来不及——
主要靠预热：发视频 -> 过会儿 @君君，描述已就绪。首轮降级 "[视频]" 占位。
无 ffmpeg 时全链路降级（视频感知必须有 ffmpeg 抽帧/抽音）。
"""

import asyncio
import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("memory.video")

_DL_TIMEOUT = 60.0
_ASR_TIMEOUT = 90.0
_CACHE_TTL = 3600.0

_CACHE: Dict[str, tuple] = {}            # md5 -> (ts, desc)
_PENDING: Dict[str, asyncio.Task] = {}   # url -> in-flight 感知任务


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("video_percept", {}) or {}
    except Exception:
        return {}


def _enabled() -> bool:
    return bool(_cfg().get("enable", True)) and shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------- 单条视频感知

async def _download(url: str, max_bytes: int) -> Optional[bytes]:
    try:
        import httpx
        chunks, total = [], 0
        async with httpx.AsyncClient(timeout=_DL_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(1024 * 256):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        logger.info(f"视频超过 {max_bytes // 1024 // 1024}MB 上限，放弃感知")
                        return None
        return b"".join(chunks)
    except Exception as e:
        logger.debug(f"视频下载失败 {url[:50]}: {type(e).__name__}: {e}")
        return None


async def _run_ffmpeg(args: list) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", *args,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        return (await proc.wait()) == 0
    except Exception:
        return False


async def _perceive_one(url: str) -> str:
    """下载 -> 抽帧 VLM + 抽音 ASR -> 一行描述。失败 "[视频]"。"""
    data = await _download(url, int(_cfg().get("max_bytes", 30 * 1024 * 1024)))
    if not data:
        return "[视频]"
    h = hashlib.md5(data).hexdigest()
    hit = _CACHE.get(h)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]

    max_frames = int(_cfg().get("max_frames", 4))
    tmp = Path(tempfile.mkdtemp(prefix="junjun_vid_"))
    try:
        video_path = tmp / "v.mp4"
        video_path.write_bytes(data)

        frame_task = _frames_desc(video_path, tmp / "frames", max_frames)
        asr_task = _audio_transcript(video_path, tmp / "a.m4a")
        frame_desc, transcript = await asyncio.gather(frame_task, asr_task)

        parts = []
        if frame_desc:
            parts.append(f"画面：{frame_desc}")
        if transcript:
            parts.append(f"语音里说：「{transcript[:100]}」")
        desc = "；".join(parts) if parts else "[视频]"
        if desc != "[视频]":
            _CACHE[h] = (time.time(), desc)
            logger.info(f"视频感知完成: {desc[:50]}")
        return desc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _frames_desc(video_path: Path, out_dir: Path, max_frames: int) -> str:
    """等间隔抽帧 -> VLM 描述 -> 拼接（VLM 未配置返回 ""）。"""
    try:
        from junjun_llm import get_chat_model
        vlm = get_chat_model("vlm")
    except Exception:
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 短视频 fps=1/5 足够覆盖；max_frames 截断长视频（覆盖前段，名场面通常在前面）
    ok = await _run_ffmpeg(["-i", str(video_path), "-vf", "fps=1/5,scale=480:-1",
                            "-frames:v", str(max_frames), "-y",
                            str(out_dir / "f_%02d.jpg")])
    frames = sorted(out_dir.glob("f_*.jpg")) if ok else []
    if not frames:
        return ""
    from junjun_memory.vision import _describe

    async def _one(p: Path):
        try:
            return await _describe(p.read_bytes(), model=vlm,
                                   prompt="这是视频里的一帧画面，用一句中文口语描述画面（20字以内）。")
        except Exception:
            return None
    descs = [d for d in await asyncio.gather(*(_one(p) for p in frames)) if d]
    return "；".join(dict.fromkeys(descs))  # 相邻帧描述常重复，去重


async def _audio_transcript(video_path: Path, m4a: Path) -> str:
    """抽音 -> ASR（无声轨/失败返回 ""）。"""
    ok = await _run_ffmpeg(["-i", str(video_path), "-vn", "-c:a", "aac",
                            "-b:a", "96k", "-y", str(m4a)])
    if not ok or not m4a.exists() or m4a.stat().st_size < 1024:
        return ""
    from junjun_llm.asr import transcribe_file
    return await transcribe_file(m4a, timeout=_ASR_TIMEOUT)


# ---------------------------------------------------------------- 预热 / 决策注入

def _shared_task(url: str) -> asyncio.Task:
    task = _PENDING.get(url)
    if task is None or task.done():
        task = asyncio.create_task(_perceive_one(url))
        _PENDING[url] = task
        task.add_done_callback(lambda _t, u=url: _PENDING.pop(u, None))
    return task


def prewarm_videos(urls: List[str]) -> None:
    """入站即后台感知（视频感知重，预热几乎是唯一能及时用完的路径）。"""
    if not urls or not _enabled():
        return
    try:
        for url in urls[:2]:  # 单条消息最多感知 2 条
            _shared_task(url)
    except Exception as e:
        logger.debug(f"视频预热失败（忽略）: {e}")


def _perception_wait() -> float:
    try:
        return float(get_global_config().raw.get("perception", {}).get("ready_wait_seconds", 3.0))
    except Exception:
        return 3.0


async def describe_videos(urls: List[str], *, wait: Optional[float] = None) -> Dict[str, str]:
    """url -> 描述。超时未完成的降级 "[视频]"，在途任务不取消（结果入 md5 缓存）。"""
    if not urls or not _enabled():
        return {}
    tasks = [_shared_task(u) for u in urls[:2]]
    if wait is None:
        wait = _perception_wait()
    if wait > 0:
        await asyncio.wait(tasks, timeout=wait)
    else:
        await asyncio.gather(*tasks, return_exceptions=True)
    out = {}
    for u, t in zip(urls[:2], tasks):
        if t.done() and not t.cancelled() and t.exception() is None:
            out[u] = t.result()
        else:
            out[u] = "[视频]"
    return out


def render_video_block(descriptions: Dict[str, str]) -> str:
    good = [d for d in descriptions.values() if d and d != "[视频]"]
    if not good:
        return ""
    if len(good) == 1:
        return f"对方发来一段视频：{good[0]}"
    return "对方发来视频：\n" + "\n".join(f"- {d}" for d in good)
