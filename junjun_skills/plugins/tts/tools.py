"""tts 插件：统一多后端语音合成（迁移自旧 MaiBot tts_voice_plugin，新架构重写）。

命令：/tts <文本> [后端]、/voice <文本> [后端]
工具：unified_tts（LLM 触发，语音发到当前会话）

后端（对齐旧插件 API 细节）：
- doubao       豆包 Seed-TTS 2.0 双向 WS（wss://openspeech.bytedance.com/api/v3/tts/bidirection，
               二进制帧协议复用 ja_tts 插件的内联实现，不重造）
- siliconflow  硅基流动 MOSS-TTSD：POST https://api.siliconflow.cn/v1/audio/speech
               {model,input,voice,response_format,stream,speed}，Bearer 鉴权
- gsv2p        GSV2P 云 API：POST /v1/audio/speech {model,input,voice,...,other_params}，Bearer 鉴权
- sovits       GPT-SoVITS 本机服务：POST http://127.0.0.1:9880/tts
               {text,text_lang,ref_audio_path,prompt_text,prompt_lang}

降级：默认/指定后端失败 -> 依次试其他已配置后端 -> 全灭回友好文本，不抛异常。
限制：文本上限 300 字（截断）；每后端超时 30s；每会话 15s 限流。
音频：存 data/tts/tts_<后端>_<毫秒时间戳>.<ext>（目录自动建），voice 段发本地路径。

偏差（相对旧插件）：旧插件的 ai_voice（MaiCore 内置）后端依赖 NapCat 侧命令转发，
新架构无对应通道，未迁移；GPT-SoVITS 的 styles 表简化为环境变量单风格；
音色/情绪参数按新需求只保留各后端默认音色常量（可用 env 覆盖）。
"""

import os
import re
import time
from pathlib import Path

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger
from junjun_skills.builtin.memory_skills import current_chat_id

logger = get_logger("plugin.tts")

_TIMEOUT = 30.0            # 每后端合成超时（秒）
_MAX_TEXT_LEN = 300        # 文本长度上限（超出截断）
_MIN_INTERVAL = 15.0       # 每会话最小间隔（秒）

# 音频输出目录（NapCat record 段支持本地绝对路径）；测试可 monkeypatch
OUTPUT_DIR = Path(os.environ.get("TTS_OUTPUT_DIR", r"E:\JunJun_Agent\data\tts"))

BACKENDS = ("doubao", "siliconflow", "gsv2p", "sovits")
_BACKEND_NAMES = {"doubao": "豆包", "siliconflow": "硅基流动",
                  "gsv2p": "GSV2P", "sovits": "GPT-SoVITS"}

# 各后端默认音色常量（可用 env 覆盖；豆包预设表见 ja_tts.VOICE_PRESETS）
_DOUBAO_DEFAULT_SPEAKER = "zh_female_vv_uranus_bigtts"   # 旧 config.toml [doubao].speaker
_SF_API_BASE = "https://api.siliconflow.cn/v1"
_SF_MODEL = "fnlp/MOSS-TTSD-v0.5"
_SF_DEFAULT_VOICE = "fnlp/MOSS-TTSD-v0.5:claire"          # 旧 config.toml [siliconflow]
_GSV2P_DEFAULT_URL = "https://gsv2p.acgnai.top/v1/audio/speech"
_GSV2P_DEFAULT_VOICE = "原神-中文-派蒙_ZH"                 # 旧 config.toml [gsv2p]
_SOVITS_DEFAULT_URL = "http://127.0.0.1:9880"

# 每会话上次合成时间戳（chat_id -> ts）
_last_use: dict = {}


# ========== 各后端合成 helper（独立 async，失败返回 None，绝不抛异常） ==========
async def synthesize_doubao(text: str) -> bytes | None:
    """豆包 Seed-TTS 双向 WS 合成（协议实现复用 ja_tts 插件）。返回 mp3 字节或 None。"""
    api_key = os.environ.get("DOUBAO_TTS_API_KEY", "").strip()
    if not api_key:
        return None
    speaker = os.environ.get("TTS_DOUBAO_SPEAKER", "").strip() or _DOUBAO_DEFAULT_SPEAKER
    try:
        # 懒加载避免 import 本模块时连带注册 ja_tts 命令
        from junjun_skills.plugins.ja_tts.tools import synthesize as _ja_synthesize
        return await _ja_synthesize(text, speaker)
    except Exception as e:
        logger.warning(f"tts 豆包合成失败: {type(e).__name__}: {e}")
        return None


async def synthesize_siliconflow(text: str) -> bytes | None:
    """硅基流动 MOSS-TTSD：POST /v1/audio/speech，返回音频字节或 None。"""
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        return None
    base = os.environ.get("TTS_SILICONFLOW_API_BASE", "").strip() or _SF_API_BASE
    voice = os.environ.get("TTS_SILICONFLOW_VOICE", "").strip() or _SF_DEFAULT_VOICE
    payload = {
        "model": _SF_MODEL,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "stream": True,
        "speed": 1.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(f"{base.rstrip('/')}/audio/speech",
                                     json=payload, headers=headers)
        if resp.status_code != 200:
            logger.warning(f"tts 硅基流动失败: HTTP {resp.status_code} {resp.text[:200]}")
            return None
        audio = resp.content
        if len(audio) < 100:
            logger.warning("tts 硅基流动返回音频过小")
            return None
        return audio
    except Exception as e:
        logger.warning(f"tts 硅基流动合成失败: {type(e).__name__}: {e}")
        return None


async def synthesize_gsv2p(text: str) -> bytes | None:
    """GSV2P 云 API：POST /v1/audio/speech，返回 mp3 字节或 None。"""
    token = os.environ.get("TTS_GSV2P_TOKEN", "").strip()
    if not token:
        return None
    url = os.environ.get("TTS_GSV2P_URL", "").strip() or _GSV2P_DEFAULT_URL
    voice = os.environ.get("TTS_GSV2P_VOICE", "").strip() or _GSV2P_DEFAULT_VOICE
    # other_params 默认值提取自旧 config.toml [gsv2p]
    payload = {
        "model": "tts-v4",
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "speed": 1.0,
        "other_params": {
            "text_lang": "中英混合", "prompt_lang": "中文", "emotion": "默认",
            "top_k": 10, "top_p": 1.0, "temperature": 1.0,
            "text_split_method": "按标点符号切", "batch_size": 1,
            "batch_threshold": 0.75, "split_bucket": True,
            "fragment_interval": 0.3, "parallel_infer": True,
            "repetition_penalty": 1.35, "sample_steps": 16,
            "if_sr": False, "seed": -1,
        },
    }
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.warning(f"tts GSV2P 失败: HTTP {resp.status_code} {resp.text[:200]}")
            return None
        audio = resp.content
        if len(audio) < 100:
            logger.warning("tts GSV2P 返回音频过小")
            return None
        return audio
    except Exception as e:
        logger.warning(f"tts GSV2P 合成失败: {type(e).__name__}: {e}")
        return None


def _detect_text_lang(text: str) -> str:
    """简单语言探测（zh/ja/en），对齐旧插件 TTSUtils.detect_language。"""
    zh = len(re.findall(r"[一-鿿]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    ja = len(re.findall(r"[぀-ゟ゠-ヿ]", text))
    total = zh + en + ja
    if total == 0 or zh / total > 0.3:
        return "zh"
    if ja / total > 0.3:
        return "ja"
    return "en" if en / total > 0.8 else "zh"


async def synthesize_sovits(text: str) -> bytes | None:
    """GPT-SoVITS 本机服务：POST {base}/tts，返回 wav 字节或 None。"""
    ref_audio = os.environ.get("TTS_SOVITS_REF_AUDIO", "").strip()
    prompt_text = os.environ.get("TTS_SOVITS_PROMPT_TEXT", "").strip()
    if not ref_audio or not prompt_text:
        return None  # 参考音频/提示文本未配置视为该后端不可用
    base = os.environ.get("TTS_SOVITS_URL", "").strip() or _SOVITS_DEFAULT_URL
    payload = {
        "text": text,
        "text_lang": _detect_text_lang(text),
        "ref_audio_path": ref_audio,
        "prompt_text": prompt_text,
        "prompt_lang": os.environ.get("TTS_SOVITS_PROMPT_LANG", "").strip() or "zh",
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(f"{base.rstrip('/')}/tts", json=payload)
        if resp.status_code != 200:
            logger.warning(f"tts GPT-SoVITS 失败: HTTP {resp.status_code} {resp.text[:200]}")
            return None
        return resp.content or None
    except Exception as e:
        logger.warning(f"tts GPT-SoVITS 合成失败: {type(e).__name__}: {e}")
        return None


# ========== 公共流程 ==========
def _backend_configured(backend: str) -> bool:
    """后端是否已配置（有凭据/参考音频），决定它是否参与自动降级。"""
    if backend == "doubao":
        return bool(os.environ.get("DOUBAO_TTS_API_KEY", "").strip())
    if backend == "siliconflow":
        return bool(os.environ.get("SILICONFLOW_API_KEY", "").strip())
    if backend == "gsv2p":
        return bool(os.environ.get("TTS_GSV2P_TOKEN", "").strip())
    if backend == "sovits":
        return bool(os.environ.get("TTS_SOVITS_REF_AUDIO", "").strip()
                    and os.environ.get("TTS_SOVITS_PROMPT_TEXT", "").strip())
    return False


def _default_backend() -> str:
    """默认后端：TTS_DEFAULT_BACKEND，非法值回退 doubao。"""
    b = os.environ.get("TTS_DEFAULT_BACKEND", "").strip().lower()
    return b if b in BACKENDS else "doubao"


def _check_rate_limit(chat_id: str) -> int:
    """返回剩余冷却秒数；0 表示可用并记录本次时间。"""
    now = time.time()
    remain = _MIN_INTERVAL - (now - _last_use.get(chat_id, 0.0))
    if remain > 0:
        return int(remain) + 1
    _last_use[chat_id] = now
    return 0


def _parse_args(args: str) -> tuple:
    """解析「文本 [后端]」：尾词命中后端名则拆出。返回 (text, backend)。"""
    args = (args or "").strip()
    parts = args.rsplit(None, 1)
    if len(parts) == 2 and parts[1].lower() in BACKENDS:
        return parts[0].strip(), parts[1].lower()
    return args, _default_backend()


async def _synthesize_with_fallback(text: str, backend: str) -> tuple:
    """先试指定后端，失败再依次试其他已配置后端。返回 (实际后端, 音频字节) 或 (None, None)。"""
    order = [backend] + [b for b in BACKENDS if b != backend]
    for b in order:
        if b != backend and not _backend_configured(b):
            continue  # 未配置的后端不参与降级
        fn = globals()[f"synthesize_{b}"]  # 运行时取值，便于测试 monkeypatch
        audio = await fn(text)
        if audio:
            if b != backend:
                logger.info(f"tts: {backend} 失败，已降级到 {b}")
            return b, audio
        logger.info(f"tts: 后端 {b} 合成失败，尝试下一个")
    return None, None


async def _synthesize_to_file(text: str, backend: str) -> Path | None:
    """截断 -> 合成（带降级）-> 落盘，返回文件路径；失败 None。"""
    text = text[:_MAX_TEXT_LEN]
    used, audio = await _synthesize_with_fallback(text, backend)
    if not audio:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = "wav" if used == "sovits" else "mp3"
    path = OUTPUT_DIR / f"tts_{used}_{int(time.time() * 1000)}.{ext}"
    path.write_bytes(audio)
    return path


async def _send_voice(chat_target: tuple, path: Path) -> None:
    """发送 voice 段。chat_target=(platform, target_id, kind)，kind=group|private。"""
    from junjun_core.contracts import ReplySet
    from junjun_core.gateway.router import get_gateway
    platform, target_id, kind = chat_target
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=[ReplySegment(type="voice", data=str(path))],
        should_reply=True,
    ))


def _no_backend_text() -> str:
    return ("语音功能还没配置哦（没有任何可用后端：缺 DOUBAO_TTS_API_KEY / "
            "SILICONFLOW_API_KEY / TTS_GSV2P_TOKEN / TTS_SOVITS_REF_AUDIO），先用文字吧。")


# ========== 命令：/tts /voice ==========
@register_command("tts", aliases=["voice"], plugin="tts",
                  description="文字转语音：/tts <文本> [后端 doubao/siliconflow/gsv2p/sovits]")
async def tts_cmd(ctx):
    args = (ctx.args or "").strip()
    if not args:
        return ("用法：/tts <文本> [后端]；可选后端："
                f"{'/'.join(BACKENDS)}（默认 {_default_backend()}）")

    if not any(_backend_configured(b) for b in BACKENDS):
        return _no_backend_text()

    remain = _check_rate_limit(ctx.session.chat_id)
    if remain > 0:
        return f"语音发得太频繁啦，{remain} 秒后再试。"

    text, backend = _parse_args(args)
    if not text:
        return "用法：/tts <文本> [后端]"

    path = await _synthesize_to_file(text, backend)
    if path is None:
        return "语音合成失败了（所有可用后端都没成功），稍后再试试吧。"
    await _send_voice((ctx.session.platform,
                       ctx.session.group_id if ctx.session.is_group else ctx.meta.user_id,
                       "group" if ctx.session.is_group else "private"), path)
    return None


# ========== 工具：LLM 触发 ==========
@tool("unified_tts")
async def unified_tts(text: str, backend: str = "") -> str:
    """把文本合成语音发到当前聊天。用户要求"发语音""用语音说""朗读""念一下""语音回复"时使用；
    未调用本工具前禁止只用文字假装已发语音。

    Args:
        text: 要读给用户听的内容（300 字内，越短效果越好）
        backend: 可选后端 doubao(豆包)/siliconflow(硅基流动)/gsv2p/sovits(GPT-SoVITS)，留空用默认
    """
    text = (text or "").strip()
    if not text:
        return "缺少要合成的文本。"

    if not any(_backend_configured(b) for b in BACKENDS):
        return "语音功能未配置（没有任何可用后端），用文字回复吧。"

    use_backend = (backend or "").strip().lower()
    if use_backend not in BACKENDS:
        use_backend = _default_backend()

    chat_id = current_chat_id.get()
    if chat_id:
        remain = _check_rate_limit(chat_id)
        if remain > 0:
            return f"语音发得太频繁了，{remain} 秒后再试，先用文字回复吧。"

    path = await _synthesize_to_file(text, use_backend)
    if path is None:
        return "语音合成失败了（所有可用后端都没成功），用文字回复吧。"

    if not chat_id:
        return f"语音已合成到 {path}，但拿不到当前会话，发不出去。"

    # chat_id 形如 "qq:ID:group|private"
    parts = chat_id.split(":")
    target = (parts[0], parts[1] if len(parts) > 1 else "",
              parts[2] if len(parts) > 2 else "private")
    await _send_voice(target, path)
    return f"语音已发送（{_BACKEND_NAMES.get(path.stem.split('_')[1], '')}后端）。不要再用文字重复语音内容。"


def probe_available() -> bool:
    """依赖探测：不阻断加载——所有后端都缺凭据时运行时再友好降级。"""
    if not any(_backend_configured(b) for b in BACKENDS):
        logger.info("tts: 所有后端均未配置，调用时将降级为文本提示")
    return True


TOOLS = [unified_tts]
