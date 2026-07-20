"""豆包 Seed-TTS 2.0 WebSocket 双向流式合成客户端。

协议: 使用官方 protocols.py 的二进制帧格式。
接口: doubao_tts_synthesize(text, api_key, speaker, ...) -> bytes
"""
import asyncio
import json
import uuid
from typing import Optional

import websockets

from junjun_core.observability import get_logger

logger = get_logger("doubao_tts")

URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

# 懒加载官方协议模块(避免启动时必导)
_protocols = None


def _get_protocols():
    global _protocols
    if _protocols is None:
        from . import protocols as _p
        _protocols = _p
    return _protocols


async def doubao_tts_synthesize(
    text: str,
    api_key: str,
    speaker: str = "zh_female_jiaochuannv_uranus_bigtts",
    resource_id: str = "seed-tts-2.0",
    audio_format: str = "mp3",
    sample_rate: int = 24000,
    speech_rate: int = 0,
    loudness_rate: int = 0,
    model: Optional[str] = None,
    context_texts: Optional[list] = None,
    explicit_language: Optional[str] = None,
    timeout: float = 30.0,
) -> bytes:
    """合成语音,返回音频字节。"""
    p = _get_protocols()

    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Connect-Id": str(uuid.uuid4()),
        "X-Control-Require-Usage-Tokens-Return": "*",
    }

    base_req_params: dict = {
        "speaker": speaker,
        "audio_params": {
            "format": audio_format,
            "sample_rate": sample_rate,
            "speech_rate": speech_rate,
            "loudness_rate": loudness_rate,
        },
    }
    if model:
        base_req_params["model"] = model
    if context_texts:
        base_req_params["context_texts"] = context_texts
    if explicit_language:
        base_req_params["explicit_language"] = explicit_language

    session_id = str(uuid.uuid4())

    async with websockets.connect(
        URL, additional_headers=headers, max_size=16 * 1024 * 1024
    ) as ws:
        # 1) 建连
        await p.start_connection(ws)
        await _wait_event(ws, p.MsgType.FullServerResponse, p.EventType.ConnectionStarted)

        # 2) 建会话
        start_payload = {"req_params": base_req_params}
        await p.start_session(ws, json.dumps(start_payload, ensure_ascii=False).encode("utf-8"), session_id)
        await _wait_event(ws, p.MsgType.FullServerResponse, p.EventType.SessionStarted)

        # 3) 发送文本 + 结束会话
        async def send_chunks():
            chunks = _split_text(text) or [text]
            for chunk in chunks:
                task_payload = {"req_params": {**base_req_params, "text": chunk}}
                await p.task_request(ws, json.dumps(task_payload, ensure_ascii=False).encode("utf-8"), session_id)
                await asyncio.sleep(0.01)
            await p.finish_session(ws, session_id)

        send_task = asyncio.create_task(send_chunks())

        # 4) 接收音频
        audio_data = bytearray()
        try:
            while True:
                msg = await asyncio.wait_for(p.receive_message(ws), timeout=timeout)
                if msg.type == p.MsgType.AudioOnlyServer and msg.event == p.EventType.TTSResponse:
                    audio_data.extend(msg.payload)
                elif msg.type == p.MsgType.FullServerResponse:
                    if msg.event == p.EventType.SessionFinished:
                        break
                    if msg.event in (p.EventType.SessionFailed, p.EventType.ConnectionFailed):
                        err_text = msg.payload.decode("utf-8", "ignore") if msg.payload else str(msg)
                        raise RuntimeError(f"豆包 TTS 会话失败: {err_text}")
        except asyncio.TimeoutError:
            raise RuntimeError("豆包 TTS 接收超时") from None

        await send_task

        # 5) 结束连接
        await p.finish_connection(ws)
        try:
            await asyncio.wait_for(
                _wait_event(ws, p.MsgType.FullServerResponse, p.EventType.ConnectionFinished),
                timeout=5,
            )
        except (asyncio.TimeoutError, Exception):
            pass

    if not audio_data:
        raise RuntimeError("豆包 TTS 未返回音频数据")
    return bytes(audio_data)


async def _wait_event(ws, msg_type, event_type):
    p = _get_protocols()
    while True:
        msg = await p.receive_message(ws)
        if msg.type == msg_type and msg.event == event_type:
            return msg
        if msg.type == p.MsgType.FullServerResponse and msg.event in (
            p.EventType.SessionFailed, p.EventType.ConnectionFailed
        ):
            err_text = msg.payload.decode("utf-8", "ignore") if msg.payload else str(msg)
            raise RuntimeError(f"豆包 TTS 失败: {err_text}")


def _split_text(text: str, max_len: int = 60) -> list:
    import re as _re
    if not text:
        return []
    parts = _re.split(r"([。！？!?；;\n]+)", text)
    chunks, buf = [], ""
    for p in parts:
        buf += p
        if p and _re.search(r"[。！？!?；;\n]", p):
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