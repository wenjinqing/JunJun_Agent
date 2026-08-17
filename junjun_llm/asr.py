"""ASR 客户端（OpenAI 兼容 /audio/transcriptions）。

key 复用号池（sf_pool，代金券直接烧），号池空退化 SILICONFLOW_API_KEY。
watch_video（长音频）与语音消息转写（短音频）共用本模块；任何失败返回 ""，
由调用方决定降级策略（弹幕热评材料 / [语音] 占位）。
"""

import os
from pathlib import Path

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("llm.asr")

_DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("asr", {}) or {}
    except Exception:
        return {}


def _pick_key() -> str:
    try:
        from junjun_llm.key_pool import sf_pool
        keys = sf_pool.healthy_keys()
        if keys:
            return keys[0]
    except Exception:
        pass
    return os.environ.get("SILICONFLOW_API_KEY", "")


async def transcribe_bytes(data: bytes, *, suffix: str = ".mp3", model: str = "",
                           timeout: float = 120.0) -> str:
    """音频字节 -> 转写文本。失败/无 key/空结果返回 ""。"""
    if not data:
        return ""
    max_bytes = int(_cfg().get("max_bytes", 20 * 1024 * 1024))
    if len(data) > max_bytes:
        logger.warning(f"音频过大（{len(data) // 1024}KB > 上限），拒绝转写")
        return ""
    key = _pick_key()
    if not key:
        logger.warning("ASR 无可用 key（号池空且无 SILICONFLOW_API_KEY）")
        return ""
    base = os.environ.get("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    model = model or str(_cfg().get("model", _DEFAULT_MODEL))
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (f"audio{suffix}", data, "application/octet-stream")},
                data={"model": model})
        text = str((resp.json() or {}).get("text") or "").strip()
        logger.info(f"ASR 转写完成: {len(text)} 字（{len(data) // 1024}KB）")
        return text
    except Exception as e:
        logger.warning(f"ASR 转写失败: {type(e).__name__}: {e}")
        return ""


async def transcribe_file(path: Path, *, model: str = "", timeout: float = 300.0) -> str:
    """音频文件 -> 转写文本（path 不存在/读失败返回 ""）。"""
    try:
        data = Path(path).read_bytes()
    except Exception as e:
        logger.warning(f"音频文件读取失败 {path}: {e}")
        return ""
    return await transcribe_bytes(data, suffix=Path(path).suffix or ".m4a",
                                  model=model, timeout=timeout)
