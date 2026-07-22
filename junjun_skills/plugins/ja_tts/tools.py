"""ja_tts 插件：豆包 Seed-TTS 2.0 日语语音合成（迁移自 mod/ja_tts_plugin，新架构重写）。

命令：/ja_tts <文本> [音色]、/ys <文本> [音色]
工具：ja_tts（LLM 可触发，语音发到当前会话）
协议：豆包双向 WS wss://openspeech.bytedance.com/api/v3/tts/bidirection，
     自定义二进制帧（4 字节头 + event/session/payload），握手流程：
     StartConnection -> ConnectionStarted -> StartSession -> SessionStarted
     -> TaskRequest(文本分块) -> FinishSession -> 收 TTSResponse 音频直到 SessionFinished
文本：日语直接送默认音色 ja_female_bv521_uranus_bigtts（原生日语，不转罗马音——用户指定）
降级：DOUBAO_TTS_API_KEY 未配置或合成失败时回友好文本，不抛异常
限制：文本上限 300 字（截断）；整体超时 30s；每会话 15s 限流
"""

import asyncio
import json
import os
import re
import struct
import time
import uuid
from enum import IntEnum
from pathlib import Path

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger
from junjun_skills.builtin.memory_skills import current_chat_id

logger = get_logger("plugin.ja_tts")

_WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
_RESOURCE_ID = "seed-tts-2.0"
_TIMEOUT = 30.0            # 整体合成超时（秒）
_MAX_TEXT_LEN = 300        # 文本长度上限（超出截断）
_MIN_INTERVAL = 15.0       # 每会话最小间隔（秒）

# 音频输出目录（NapCat record 段支持本地绝对路径）；测试可 monkeypatch
OUTPUT_DIR = Path(os.environ.get("JA_TTS_OUTPUT_DIR", r"E:\JunJun_Agent\data\tts"))

# 音色预设（对齐旧插件）
VOICE_PRESETS = {
    "ja": "ja_female_bv521_uranus_bigtts",          # 原生日语女声（默认）
    "jiaochuannv": "zh_female_jiaochuannv_uranus_bigtts",
    "shuangkuaisisi": "zh_female_shuangkuaisisi_uranus_bigtts",
    "wenwanshanshan": "saturn_zh_female_wenwanshanshan_cs_tob",
    "tiaopigongzhu": "saturn_zh_female_tiaopigongzhu_tob",
    "youyoujunzi": "zh_male_youyoujunzi_uranus_bigtts",
    "shaonianzixin": "zh_male_shaonianzixin_uranus_bigtts",
    "wennuanahu": "zh_male_wennuanahu_uranus_bigtts",
    "vv": "zh_female_vv_uranus_bigtts",
}
DEFAULT_VOICE = "ja"

# 每会话上次合成时间戳（chat_id -> ts）
_last_use: dict = {}


# ========== 豆包双向 WS 二进制帧协议（按官方协议重写，仅保留所需子集） ==========
class _MsgType(IntEnum):
    FULL_CLIENT_REQUEST = 0b1
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_SERVER = 0b1011


class _Event(IntEnum):
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    START_SESSION = 100
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    TASK_REQUEST = 200
    TTS_RESPONSE = 352


_FLAG_WITH_EVENT = 0b100
_FLAG_POSITIVE_SEQ = 0b1
_FLAG_NEGATIVE_SEQ = 0b11
# 写侧：连接级事件不带 session_id
_WRITE_NO_SESSION = {_Event.START_CONNECTION, _Event.FINISH_CONNECTION,
                     _Event.CONNECTION_STARTED, _Event.CONNECTION_FAILED}
# 读侧：连接级事件不读 session_id（比写侧多 ConnectionFinished）
_READ_NO_SESSION = _WRITE_NO_SESSION | {_Event.CONNECTION_FINISHED}
# 读侧：连接级下行事件额外带 connect_id
_READ_CONNECT_ID = {_Event.CONNECTION_STARTED, _Event.CONNECTION_FAILED,
                    _Event.CONNECTION_FINISHED}


def _marshal(msg_type: int, event: int, session_id: str = "", payload: bytes = b"{}") -> bytes:
    """序列化一帧：4 字节头（v1/头长4、类型|WithEvent、JSON|无压缩、保留字节）
    + event(int32) + [session_id] + payload 长度(uint32) + payload。"""
    buf = bytearray([0x11, (msg_type << 4) | _FLAG_WITH_EVENT, 0x10, 0x00])
    buf += struct.pack(">i", event)
    if event not in _WRITE_NO_SESSION:
        sid = session_id.encode("utf-8")
        buf += struct.pack(">I", len(sid)) + sid
    buf += struct.pack(">I", len(payload)) + payload
    return bytes(buf)


def _unmarshal(data: bytes) -> tuple:
    """反序列化一帧，返回 (msg_type, event, payload)。容错未知 event 值。"""
    header_size = (data[0] & 0x0F) * 4
    msg_type = data[1] >> 4
    flag = data[1] & 0x0F
    off = header_size
    if flag in (_FLAG_POSITIVE_SEQ, _FLAG_NEGATIVE_SEQ):
        off += 4  # sequence（不使用）
    event = 0
    if flag & _FLAG_WITH_EVENT:
        (event,) = struct.unpack_from(">i", data, off)
        off += 4
        if event not in _READ_NO_SESSION:
            (sid_len,) = struct.unpack_from(">I", data, off)
            off += 4 + sid_len
        if event in _READ_CONNECT_ID:
            (cid_len,) = struct.unpack_from(">I", data, off)
            off += 4 + cid_len
    payload = b""
    if off + 4 <= len(data):
        (size,) = struct.unpack_from(">I", data, off)
        off += 4
        payload = data[off:off + size]
    return msg_type, event, payload


async def _ws_recv(ws) -> tuple:
    """收一帧并解析；文本帧视为协议错误。"""
    data = await ws.recv()
    if not isinstance(data, bytes):
        raise RuntimeError(f"豆包 TTS 收到非二进制帧: {type(data).__name__}")
    return _unmarshal(data)


async def _wait_event(ws, event: int) -> None:
    """等待指定下行事件；遇到会话/连接失败直接抛错。"""
    while True:
        msg_type, ev, payload = await _ws_recv(ws)
        if msg_type == _MsgType.FULL_SERVER_RESPONSE:
            if ev == event:
                return
            if ev in (_Event.SESSION_FAILED, _Event.CONNECTION_FAILED):
                err = payload.decode("utf-8", "ignore") or f"event={ev}"
                raise RuntimeError(f"豆包 TTS 失败: {err}")


def _split_text(text: str, max_len: int = 60) -> list:
    """按句末标点分块并合并到 max_len 内（流式送文本，首包更快）。"""
    if not text:
        return []
    parts = re.split(r"([。！？!?；;\n]+)", text)
    chunks, buf = [], ""
    for p in parts:
        buf += p
        if p and re.search(r"[。！？!?；;\n]", p):
            if buf.strip():
                chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    merged = []
    for c in chunks:
        if merged and len(merged[-1]) + len(c) <= max_len:
            merged[-1] += c
        else:
            merged.append(c)
    return merged


async def _synthesize_ws(text: str, api_key: str, speaker: str) -> bytes:
    """走完整双向 WS 握手合成，返回 mp3 字节；失败抛异常（由 synthesize 兜底）。"""
    import websockets

    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": _RESOURCE_ID,
        "X-Api-Connect-Id": str(uuid.uuid4()),
        "X-Control-Require-Usage-Tokens-Return": "*",
    }
    req_params = {
        "speaker": speaker,
        "audio_params": {
            "format": "mp3",
            "sample_rate": 24000,
            "speech_rate": 0,
            "loudness_rate": 0,
        },
    }
    session_id = str(uuid.uuid4())

    async with websockets.connect(_WS_URL, additional_headers=headers,
                                  max_size=16 * 1024 * 1024) as ws:
        # 1) 建连 -> ConnectionStarted
        await ws.send(_marshal(_MsgType.FULL_CLIENT_REQUEST, _Event.START_CONNECTION))
        await _wait_event(ws, _Event.CONNECTION_STARTED)

        # 2) 建会话 -> SessionStarted
        start_payload = json.dumps({"req_params": req_params}, ensure_ascii=False).encode("utf-8")
        await ws.send(_marshal(_MsgType.FULL_CLIENT_REQUEST, _Event.START_SESSION,
                               session_id, start_payload))
        await _wait_event(ws, _Event.SESSION_STARTED)

        # 3) 后台分块送文本，结束后 FinishSession
        async def send_chunks():
            for chunk in _split_text(text) or [text]:
                task_payload = json.dumps({"req_params": {**req_params, "text": chunk}},
                                          ensure_ascii=False).encode("utf-8")
                await ws.send(_marshal(_MsgType.FULL_CLIENT_REQUEST, _Event.TASK_REQUEST,
                                       session_id, task_payload))
                await asyncio.sleep(0.01)
            await ws.send(_marshal(_MsgType.FULL_CLIENT_REQUEST, _Event.FINISH_SESSION,
                                   session_id))

        send_task = asyncio.create_task(send_chunks())
        try:
            # 4) 收音频直到 SessionFinished
            audio = bytearray()
            while True:
                msg_type, ev, payload = await _ws_recv(ws)
                if msg_type == _MsgType.AUDIO_ONLY_SERVER and ev == _Event.TTS_RESPONSE:
                    audio.extend(payload)
                elif msg_type == _MsgType.FULL_SERVER_RESPONSE:
                    if ev == _Event.SESSION_FINISHED:
                        break
                    if ev in (_Event.SESSION_FAILED, _Event.CONNECTION_FAILED):
                        err = payload.decode("utf-8", "ignore") or f"event={ev}"
                        raise RuntimeError(f"豆包 TTS 会话失败: {err}")
        finally:
            await send_task

        # 5) 收尾连接（失败不影响已拿到的音频）
        await ws.send(_marshal(_MsgType.FULL_CLIENT_REQUEST, _Event.FINISH_CONNECTION))
        try:
            await asyncio.wait_for(_wait_event(ws, _Event.CONNECTION_FINISHED), timeout=5)
        except Exception:
            pass

    if not audio:
        raise RuntimeError("豆包 TTS 未返回音频数据")
    return bytes(audio)


async def synthesize(text: str, speaker: str = "") -> bytes | None:
    """合成语音字节（独立 helper，测试 monkeypatch 它）；任何失败返回 None。"""
    api_key = os.environ.get("DOUBAO_TTS_API_KEY", "").strip()
    if not api_key:
        return None
    speaker = speaker or VOICE_PRESETS[DEFAULT_VOICE]
    try:
        return await asyncio.wait_for(_synthesize_ws(text, api_key, speaker), timeout=_TIMEOUT)
    except Exception as e:
        logger.warning(f"ja_tts 合成失败: {type(e).__name__}: {e}")
        return None


# ========== 公共流程 ==========
def _get_api_key() -> str:
    return os.environ.get("DOUBAO_TTS_API_KEY", "").strip()


def _check_rate_limit(chat_id: str) -> int:
    """返回剩余冷却秒数；0 表示可用并记录本次时间。"""
    now = time.time()
    remain = _MIN_INTERVAL - (now - _last_use.get(chat_id, 0.0))
    if remain > 0:
        return int(remain) + 1
    _last_use[chat_id] = now
    return 0


def _parse_args(args: str) -> tuple:
    """解析「文本 [音色]」：尾词命中音色预设则拆出。返回 (text, speaker)。"""
    args = (args or "").strip()
    if not args:
        return "", VOICE_PRESETS[DEFAULT_VOICE]
    parts = args.rsplit(None, 1)
    if len(parts) == 2 and parts[1].lower() in VOICE_PRESETS:
        return parts[0].strip(), VOICE_PRESETS[parts[1].lower()]
    return args, VOICE_PRESETS[DEFAULT_VOICE]


async def _synthesize_to_file(text: str, speaker: str) -> Path | None:
    """预处理（截断）-> 合成 -> 落盘 mp3，返回文件路径；失败 None。"""
    text = text[:_MAX_TEXT_LEN]
    audio = await synthesize(text, speaker)
    if not audio:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"ja_{int(time.time() * 1000)}.mp3"
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


# ========== 命令：/ja_tts /ys ==========
@register_command("ja_tts", aliases=["ys"], plugin="ja_tts",
                  description="日语语音合成：/ja_tts <文本> [音色]")
async def ja_tts_cmd(ctx):
    args = (ctx.args or "").strip()
    if not args:
        voices = "/".join(VOICE_PRESETS)
        return f"用法：/ja_tts <日语文本> [音色]；可选音色：{voices}（默认 ja）"

    if not _get_api_key():
        return "语音功能没配置哦（缺 DOUBAO_TTS_API_KEY），先用文字吧。"

    remain = _check_rate_limit(ctx.session.chat_id)
    if remain > 0:
        return f"语音发得太频繁啦，{remain} 秒后再试。"

    text, speaker = _parse_args(args)
    if not text:
        return "用法：/ja_tts <日语文本> [音色]"

    path = await _synthesize_to_file(text, speaker)
    if path is None:
        return "语音合成失败了，稍后再试试吧。"
    await _send_voice((ctx.session.platform,
                       ctx.session.group_id if ctx.session.is_group else ctx.meta.user_id,
                       "group" if ctx.session.is_group else "private"), path)
    return None


# ========== 工具：LLM 触发 ==========
@tool("ja_tts")
async def ja_tts_tool(text: str, speaker: str = "") -> str:
    """把日语（或中文）文本合成语音发到当前聊天。用户要求"用语音说""发日语语音""说句日语听听"时使用。

    Args:
        text: 要合成语音的文本（300 字内，越短效果越好）
        speaker: 可选音色名，如 ja（日语女声，默认）/vv/jiaochuannv/youyoujunzi，留空用默认
    """
    text = (text or "").strip()
    if not text:
        return "缺少要合成的文本。"

    if not _get_api_key():
        return "语音功能未配置（缺 DOUBAO_TTS_API_KEY），用文字回复吧。"

    chat_id = current_chat_id.get()
    if chat_id:
        remain = _check_rate_limit(chat_id)
        if remain > 0:
            return f"语音发得太频繁了，{remain} 秒后再试，先用文字回复吧。"

    speaker_id = VOICE_PRESETS.get((speaker or "").strip().lower(),
                                   VOICE_PRESETS[DEFAULT_VOICE])
    path = await _synthesize_to_file(text, speaker_id)
    if path is None:
        return "语音合成失败了，用文字回复吧。"

    if not chat_id:
        return f"语音已合成到 {path}，但拿不到当前会话，发不出去。"

    # chat_id 形如 "qq:ID:group|private"
    parts = chat_id.split(":")
    target = (parts[0], parts[1] if len(parts) > 1 else "",
              parts[2] if len(parts) > 2 else "private")
    await _send_voice(target, path)
    return "语音已发送。不要再用文字重复语音内容。"


def probe_available() -> bool:
    """依赖探测：不阻断加载——缺 key 在运行时友好降级。"""
    if not _get_api_key():
        logger.info("ja_tts: DOUBAO_TTS_API_KEY 未设置，调用时将降级为文本提示")
    return True


TOOLS = [ja_tts_tool]
