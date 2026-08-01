"""video_watch job：认真看完一个 B 站视频（T2 真·看视频）。

与 bilibili_summary（T1）的分工：
- T1：字幕/弹幕/热评免费 API，2-3 秒「瞄一眼」，覆盖有字幕的 ~80%
- T2：字幕不够时真下载——DASH 音轨（或全文件抽音频）-> 硅基流动 ASR
  转写；VLM 槽可用时抽关键帧描述画面 -> 综述成观后报告。
  挂在异步队列上跑几分钟，完成后君君主动汇报「我看完啦」

流水线分支（每步可降级）：
  字幕命中 -> 直接综述（不下载，T2 白嫖 T1 缓存）
  无字幕   -> 下载 -> ffmpeg 抽音频 -> ASR 转写（失败降级弹幕热评材料）
             -> （有 VLM + 视频流）关键帧描述
             -> 综述

ASR 走硅基流动 OpenAI 兼容 /audio/transcriptions（SenseVoiceSmall），
key 复用号池（sf_pool），代金券直接烧。
"""

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.bilibili.watch")

_ASR_TIMEOUT = 300.0    # ASR 转写超时（10 分钟音频量级）
_FRAME_PROMPT = "这是一段视频里的一帧画面。用一句中文口语描述画面内容（30字以内）。"


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("video_watch", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------- 媒体抓取

async def _grab_media(info: dict, need_video: bool, tmp: Path):
    """下载音频（DASH 音轨优先，免抽）与可选视频流。返回 (video_path, audio_path)。"""
    from junjun_skills.plugins.bilibili.tools import _download, _fetch_playurl
    sources = await _fetch_playurl(info["aid"], info["cid"])
    if not sources:
        return None, None
    video_path = audio_path = None
    if sources["type"] == "dash":
        if sources.get("audio"):
            audio_path = tmp / "a.m4s"
            if not await _download(sources["audio"], audio_path):
                audio_path = None
        if need_video:
            video_path = tmp / "v.m4s"
            if not await _download(sources["video"], video_path):
                video_path = None
    else:  # durl 单文件：只有全文件，音频后面用 ffmpeg 抽
        video_path = tmp / "v.mp4"
        if not await _download(sources["url"], video_path):
            video_path = None
    return video_path, audio_path


async def _extract_audio(video: Optional[Path], audio: Optional[Path], out: Path) -> bool:
    """统一产出干净 m4a（m4s 是分片 mp4，转封装防 ASR 拒收）。"""
    from junjun_skills.plugins.bilibili.tools import _run_ffmpeg
    src = audio or video
    if src is None:
        return False
    return await _run_ffmpeg(["-i", str(src), "-vn", "-c:a", "aac",
                              "-b:a", "96k", "-y", str(out)])


async def _extract_frames(video: Path, out_dir: Path, interval: int, max_n: int) -> list:
    """等间隔抽关键帧（缩小到 480p 宽，省 VLM token）。"""
    from junjun_skills.plugins.bilibili.tools import _run_ffmpeg
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = await _run_ffmpeg([
        "-i", str(video), "-vf", f"fps=1/{interval},scale=480:-1",
        "-frames:v", str(max_n), "-y", str(out_dir / "f_%03d.jpg")])
    return sorted(out_dir.glob("f_*.jpg")) if ok else []


# ---------------------------------------------------------------- ASR / VLM

async def _asr_transcribe(audio_path: Path, *, client_factory=None) -> str:
    """硅基流动 ASR（OpenAI 兼容 /audio/transcriptions）。失败返回 ""。"""
    try:
        import httpx
        key = ""
        try:
            from junjun_llm.key_pool import sf_pool
            keys = sf_pool.healthy_keys()
            key = keys[0] if keys else ""
        except Exception:
            pass
        key = key or os.environ.get("SILICONFLOW_API_KEY", "")
        if not key:
            logger.warning("ASR 无可用 key（号池空且无 SILICONFLOW_API_KEY）")
            return ""
        base = os.environ.get("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        model = str(_cfg().get("asr_model", "FunAudioLLM/SenseVoiceSmall"))
        factory = client_factory or (lambda: httpx.AsyncClient(timeout=_ASR_TIMEOUT))
        async with factory() as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(
                    f"{base}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (audio_path.name, f, "audio/m4a")},
                    data={"model": model})
        text = str((resp.json() or {}).get("text") or "").strip()
        logger.info(f"ASR 转写完成: {len(text)} 字")
        return text
    except Exception as e:
        logger.warning(f"ASR 转写失败: {type(e).__name__}: {e}")
        return ""


async def _describe_frames(frames: list, *, vlm=None) -> list:
    """关键帧 -> VLM 口语描述（复用 vision 的 VLM 调用与限流）。失败帧跳过。"""
    if not frames:
        return []
    if vlm is None:
        try:
            from junjun_llm import get_chat_model
            vlm = get_chat_model("vlm")
        except Exception:
            return []
    from junjun_memory.vision import _describe

    async def _one(p: Path):
        try:
            return await _describe(p.read_bytes(), model=vlm, prompt=_FRAME_PROMPT)
        except Exception:
            return None
    descs = await asyncio.gather(*(_one(p) for p in frames))
    return [d for d in descs if d]


# ---------------------------------------------------------------- 综述

_SYNTH_PROMPT = """你在认真看一个 B 站视频，基于以下材料写一份观后报告（中文，{max_chars} 字以内）：
- 这个视频讲了什么（按脉络分两三段说清楚）
- 亮点/名场面/值得看的点
- 材料不足的部分（没字幕没语音的角度）如实承认，不许编
材料：
{material}"""


async def _synthesize(material: str, *, model=None) -> str:
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("agent")
    from langchain_core.messages import HumanMessage
    resp = await model.ainvoke([HumanMessage(content=_SYNTH_PROMPT.format(
        max_chars=int(_cfg().get("report_max_chars", 1200)),
        material=material[:14000]))])
    text = str(resp.content).strip()
    if not text:
        raise RuntimeError("观后报告生成失败（模型返回空）")
    return text


# ---------------------------------------------------------------- 主流程

async def video_watch_handler(job, payload: dict, *, synth_model=None,
                              asr=None, vlm=None) -> str:
    """看视频主流程（异步队列 handler）。抛异常 = job 失败。"""
    from junjun_skills.plugins.bilibili import content
    from junjun_skills.plugins.bilibili.tools import TMP_DIR, _ffmpeg_path

    url = str(payload.get("url") or "").strip()
    if not url:
        raise RuntimeError("没有视频链接")

    # 1) 先白嫖字幕（T1 缓存共享）：命中就不用下载了
    m = await content.get_material(url)
    if m is None:
        raise RuntimeError("视频信息都拿不到（链接失效或被风控）")
    info = m["info"]
    max_dur = int(_cfg().get("max_duration", 1800))
    if max_dur > 0 and (info.get("duration") or 0) > max_dur:
        raise RuntimeError(f"视频 {info['duration'] // 60} 分钟太长了，超过我能看的上限"
                           f"（{max_dur // 60} 分钟）")
    if m["source"] == "字幕":
        logger.info(f"视频 {info['bvid']} 字幕命中，直接综述（免下载）")
        return await _synthesize(m["material"], model=synth_model)

    if _ffmpeg_path() is None:
        # 无 ffmpeg 没法抽音频/帧——降级 T1 材料综述，不算失败
        logger.warning("无 ffmpeg，video_watch 降级为弹幕热评综述")
        return await _synthesize(m["material"], model=synth_model)

    # 2) 下载 -> 抽音频 -> ASR；有 VLM 时抽帧
    tmp = TMP_DIR / f"watch_{info['bvid']}_{int(time.time() * 1000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        need_frames = vlm is not None or _vlm_available()
        video, audio = await _grab_media(info, need_frames, tmp)
        parts = [m["material"]]

        m4a = tmp / "audio.m4a"
        if await _extract_audio(video, audio, m4a):
            transcribe = asr or _asr_transcribe
            transcript = await transcribe(m4a)
            if transcript:
                parts.append(f"语音转写：\n{transcript[:int(_cfg().get('asr_max_chars', 6000))]}")
        else:
            logger.warning(f"音频抽取失败 {info['bvid']}（用弹幕热评材料综述）")

        if need_frames and video is not None:
            frames = await _extract_frames(
                video, tmp / "frames",
                int(_cfg().get("keyframe_interval", 30)),
                int(_cfg().get("max_keyframes", 8)))
            descs = await _describe_frames(frames, vlm=vlm)
            if descs:
                parts.append("画面关键帧：\n" + "\n".join(f"- {d}" for d in descs))

        return await _synthesize("\n\n".join(parts), model=synth_model)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _vlm_available() -> bool:
    try:
        from junjun_llm import get_chat_model
        return get_chat_model("vlm") is not None
    except Exception:
        return False
