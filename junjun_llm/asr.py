"""ASR 客户端（本地 sherpa-onnx SenseVoiceSmall，2026-08-18 起）。

历史：原走 SiliconFlow /audio/transcriptions（SenseVoiceSmall 云端），
2026-08-18 SF 欠费 402 退役；AI Ping 无 ASR 模型（/audio/transcriptions 404），
用户拍板本地化——sherpa-onnx + SenseVoiceSmall int8（~234MB），CPU 推理。

模型文件：data/models/sense-voice/{model.int8.onnx,tokens.txt}（不入库；
缺失时 ASR 静默降级返回 ""，与旧行为一致）。解码靠系统 ffmpeg（任意格式
-> 16k 单声道 f32 PCM 管道流，不落临时文件）。

Windows 坑（2026-08-18 实锤）：sherpa-onnx win 轮子不带 onnxruntime.dll，
按名解析撞上 System32 里 Windows 自带的 ORT 1.17（API 太老直接拒载）。
onnxruntime pip 包的 capi DLL 复制到 _sherpa_onnx.pyd 同目录即可优先命中
（LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 先于 System32）——_ensure_ort_dlls 每次
进程启动自检补齐，uv 重装 sherpa-onnx 后也自愈。

watch_video（长音频）与语音消息转写（短音频）共用本模块；任何失败返回 ""，
由调用方决定降级策略（弹幕热评材料 / [语音] 占位）。
"""

import asyncio
import shutil
from pathlib import Path

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("llm.asr")

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "sense-voice"
_MIN_SAMPLES = 1600          # <0.1s 的音频没有转写意义

_recognizer = None
_recognizer_failed = False   # 加载失败只告警一次，之后静默降级


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("asr", {}) or {}
    except Exception:
        return {}


def _ensure_ort_dlls() -> None:
    """把 onnxruntime pip 包的 capi DLL 同步到 _sherpa_onnx.pyd 同目录。

    sherpa-onnx win 轮子不带 onnxruntime.dll（dist-info RECORD 无 dll 条目），
    不补则按名撞上 System32 的旧 ORT（模块目录优先于 System32 命中）。
    幂等：目标存在且大小一致即跳过；uv 重装 sherpa-onnx 后自动重补。
    """
    try:
        import onnxruntime
        import sherpa_onnx
        src_dir = Path(onnxruntime.__file__).parent / "capi"
        dst_dir = Path(sherpa_onnx.__file__).parent / "lib"
        for dll in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
            src, dst = src_dir / dll, dst_dir / dll
            if src.exists() and (not dst.exists()
                                 or src.stat().st_size != dst.stat().st_size):
                shutil.copy2(src, dst)
                logger.info(f"已同步 {dll} -> sherpa_onnx/lib（避让 System32 旧 ORT）")
    except Exception as e:
        logger.warning(f"ORT DLL 同步失败（ASR 可能不可用）: {type(e).__name__}: {e}")


def _get_recognizer():
    """懒加载识别器（模型 ~234MB int8，首次加载数秒）；失败记一次性告警后静默降级。"""
    global _recognizer, _recognizer_failed
    if _recognizer is not None or _recognizer_failed:
        return _recognizer
    model_dir = Path(_cfg().get("model_dir", "") or _DEFAULT_MODEL_DIR)
    model, tokens = model_dir / "model.int8.onnx", model_dir / "tokens.txt"
    if not model.exists() or not tokens.exists():
        _recognizer_failed = True
        logger.warning(f"ASR 模型文件缺失（{model_dir}），语音转写降级；"
                       "下载 sherpa-onnx sense-voice int8 模型放入后重启恢复")
        return None
    try:
        _ensure_ort_dlls()
        import sherpa_onnx
        _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(tokens),
            num_threads=int(_cfg().get("num_threads", 2)),
            use_itn=True, debug=False)
        logger.info(f"ASR 识别器已加载（{model_dir.name}/model.int8.onnx, 本地 CPU）")
    except Exception as e:
        _recognizer_failed = True
        logger.warning(f"ASR 识别器加载失败（语音转写降级）: {type(e).__name__}: {e}")
    return _recognizer


async def _decode_to_16k(data: bytes) -> "object | None":
    """压缩音频字节 -> 16kHz 单声道 float32 PCM（ffmpeg 管道，不落临时文件）。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate(data)
        if proc.returncode != 0 or not out:
            logger.warning(f"ffmpeg 音频解码失败: {err.decode(errors='replace')[:120]}")
            return None
        import numpy as np
        return np.frombuffer(out, dtype=np.float32)
    except FileNotFoundError:
        logger.warning("ffmpeg 未安装，ASR 解码不可用")
    except Exception as e:
        logger.warning(f"音频解码失败: {type(e).__name__}: {e}")
    return None


def _recognize(samples) -> str:
    """同步识别（CPU 密集——调用方须 to_thread 卸载，别堵事件循环）。"""
    rec = _get_recognizer()
    if rec is None:
        return ""
    stream = rec.create_stream()
    stream.accept_waveform(16000, samples)
    rec.decode_stream(stream)
    return (stream.result.text or "").strip()


async def transcribe_bytes(data: bytes, *, suffix: str = ".mp3", model: str = "",
                           timeout: float = 120.0) -> str:
    """音频字节 -> 转写文本。失败/模型缺失/空结果返回 ""。

    suffix/model 参数仅为兼容旧调用方保留：本地管线 ffmpeg 自动嗅探格式，
    识别模型固定 sense-voice int8（云端时代按名选模型已不复存在）。
    """
    if not data:
        return ""
    max_bytes = int(_cfg().get("max_bytes", 20 * 1024 * 1024))
    if len(data) > max_bytes:
        logger.warning(f"音频过大（{len(data) // 1024}KB > 上限），拒绝转写")
        return ""
    samples = await _decode_to_16k(data)
    if samples is None or len(samples) < _MIN_SAMPLES:
        return ""
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_recognize, samples), timeout=timeout)
    except Exception as e:
        logger.warning(f"ASR 转写失败: {type(e).__name__}: {e}")
        return ""
    logger.info(f"ASR 转写完成: {len(text)} 字（{len(data) // 1024}KB）")
    return text


async def transcribe_file(path: Path, *, model: str = "", timeout: float = 300.0) -> str:
    """音频文件 -> 转写文本（path 不存在/读失败返回 ""）。"""
    try:
        data = Path(path).read_bytes()
    except Exception as e:
        logger.warning(f"音频文件读取失败 {path}: {e}")
        return ""
    return await transcribe_bytes(data, suffix=Path(path).suffix or ".m4a",
                                  model=model, timeout=timeout)
