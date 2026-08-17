"""video_watch job：认真看完一个 B 站/抖音视频（T2 真·看视频）。

与 bilibili_summary（T1）的分工：
- T1：字幕/弹幕/热评免费 API，2-3 秒「瞄一眼」，覆盖有字幕的 ~80%
- T2：字幕不够时真下载——DASH 音轨（或全文件抽音频）-> 硅基流动 ASR
  转写；VLM 槽可用时抽关键帧描述画面 -> 综述成观后报告。
  挂在异步队列上跑几分钟，完成后君君主动汇报「我看完啦」
- 抖音：SSR 解析直接给 mp4 直链（无字幕体系），走同一条下载->ASR->抽帧
  流水线；直播/回放没有可下载流，直接婉拒

流水线分支（每步可降级）：
  字幕命中 -> 直接综述（不下载，T2 白嫖 T1 缓存）
  无字幕   -> 下载 -> ffmpeg 抽音频 -> ASR 转写（失败降级弹幕热评材料）
             -> （有 VLM + 视频流）关键帧描述
             -> 综述

ASR 走 OpenAI 兼容 /audio/transcriptions，
key 复用号池（sf_pool），代金券直接烧。
"""

import asyncio
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
    """视频音频转写（长音频，走共享 ASR 模块，模型取 [video_watch].asr_model）。"""
    from junjun_llm.asr import transcribe_file
    return await transcribe_file(audio_path,
                                 model=str(_cfg().get("asr_model", "")),
                                 timeout=_ASR_TIMEOUT)


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

_SYNTH_PROMPT = """你在认真看一个{platform}视频，基于以下材料写一份观后报告（中文，{max_chars} 字以内）：
- 这个视频讲了什么（按脉络分两三段说清楚）
- 亮点/名场面/值得看的点
- 材料不足的部分（没字幕没语音的角度）如实承认，不许编
材料：
{material}"""


async def _synthesize(material: str, *, model=None, platform: str = "B 站") -> str:
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("agent")
    from langchain_core.messages import HumanMessage
    resp = await model.ainvoke([HumanMessage(content=_SYNTH_PROMPT.format(
        platform=platform,
        max_chars=int(_cfg().get("report_max_chars", 1200)),
        material=material[:14000]))])
    text = str(resp.content).strip()
    if not text:
        raise RuntimeError("观后报告生成失败（模型返回空）")
    return text


# ---------------------------------------------------------------- 主流程

async def _finish_watch(material: str, video: Optional[Path], audio: Optional[Path],
                        tmp: Path, *, synth_model=None, asr=None, vlm=None,
                        platform: str, tag: str) -> str:
    """下载完成后的共享流水线：抽音频 -> ASR -> 抽帧 -> VLM -> 综述。"""
    parts = [material]

    m4a = tmp / "audio.m4a"
    if await _extract_audio(video, audio, m4a):
        transcribe = asr or _asr_transcribe
        transcript = await transcribe(m4a)
        if transcript:
            parts.append(f"语音转写：\n{transcript[:int(_cfg().get('asr_max_chars', 6000))]}")
    else:
        logger.warning(f"音频抽取失败 {tag}（用基础材料综述）")

    if video is not None and (vlm is not None or _vlm_available()):
        frames = await _extract_frames(
            video, tmp / "frames",
            int(_cfg().get("keyframe_interval", 30)),
            int(_cfg().get("max_keyframes", 8)))
        descs = await _describe_frames(frames, vlm=vlm)
        if descs:
            parts.append("画面关键帧：\n" + "\n".join(f"- {d}" for d in descs))

    return await _synthesize("\n\n".join(parts), model=synth_model, platform=platform)


async def video_watch_handler(job, payload: dict, *, synth_model=None,
                              asr=None, vlm=None) -> str:
    """看视频主流程（异步队列 handler）。抛异常 = job 失败。"""
    from junjun_skills.plugins.bilibili import content
    from junjun_skills.plugins.bilibili.tools import TMP_DIR, _ffmpeg_path

    url = str(payload.get("url") or "").strip()
    if not url:
        raise RuntimeError("没有视频链接")

    if "douyin.com" in url:
        return await _watch_douyin(url, synth_model=synth_model, asr=asr, vlm=vlm)

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
        return await _finish_watch(m["material"], video, audio, tmp,
                                   synth_model=synth_model, asr=asr, vlm=vlm,
                                   platform="B 站", tag=info["bvid"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 抖音分支

_DY_DL_MAX_BYTES = 400 * 1024 * 1024  # mp4 下载上限（防超大文件撑爆磁盘）


async def _download_douyin(url: str, path: Path) -> bool:
    """抖音 mp4 直链流式下载（移动端 UA + Referer 防盗链，带体积上限）。"""
    import httpx
    from junjun_skills.plugins.douyin.content import _UA_MOBILE
    headers = {"User-Agent": _UA_MOBILE, "Referer": "https://www.douyin.com/"}
    try:
        async with httpx.AsyncClient(headers=headers, timeout=120.0,
                                     follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                size = 0
                with open(path, "wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        size += len(chunk)
                        if size > _DY_DL_MAX_BYTES:
                            raise RuntimeError(f"文件超过 {_DY_DL_MAX_BYTES // 1024 // 1024}MB 上限")
                        f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"抖音视频下载失败: {type(e).__name__}: {e}")
        path.unlink(missing_ok=True)
        return False


async def _watch_douyin(url: str, *, synth_model=None, asr=None, vlm=None) -> str:
    """抖音深看：SSR 直链 mp4 -> 同一套 抽音频/ASR/抽帧/VLM 流水线。"""
    from junjun_skills.plugins.douyin import content as dy
    from junjun_skills.plugins.bilibili.tools import TMP_DIR, _ffmpeg_path

    m = await dy.get_material(url)
    if m and m.get("type") == "live":
        raise RuntimeError("这是抖音直播/直播回放，没有可下载的视频流，看不了")
    if not m or not m.get("info"):
        raise RuntimeError("抖音视频资料都拿不到（链接失效或被风控）")
    info = m["info"]
    max_dur = int(_cfg().get("max_duration", 1800))
    if max_dur > 0 and (info.get("duration_s") or 0) > max_dur:
        raise RuntimeError(f"视频 {info['duration_s'] // 60} 分钟太长了，超过我能看的上限"
                           f"（{max_dur // 60} 分钟）")

    play_url = info.get("play_url") or ""
    if not play_url or _ffmpeg_path() is None:
        # 没直链或没 ffmpeg：降级文案综述，不算失败
        logger.warning("抖音无直链或无 ffmpeg，video_watch 降级为文案综述")
        return await _synthesize(m["material"], model=synth_model, platform="抖音")

    tmp = TMP_DIR / f"watch_dy_{info['aweme_id']}_{int(time.time() * 1000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        video = tmp / "v.mp4"
        if not await _download_douyin(play_url, video):
            video = None  # 下载失败仍有文案兜底（_finish_watch 会跳过 ASR/抽帧）
        return await _finish_watch(m["material"], video, None, tmp,
                                   synth_model=synth_model, asr=asr, vlm=vlm,
                                   platform="抖音", tag=info["aweme_id"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _vlm_available() -> bool:
    try:
        from junjun_llm import get_chat_model
        return get_chat_model("vlm") is not None
    except Exception:
        return False
