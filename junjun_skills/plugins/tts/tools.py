"""TTS skill 插件：豆包 TTS 文本转语音（原 mod/ja_tts_plugin 迁移）。"""

import os
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from junjun_core.observability import get_logger
from junjun_skills.builtin.memory_skills import current_chat_id

logger = get_logger("skills.tts")


@tool
async def send_voice(text: str) -> str:
    """把一段话转成语音发出去。用户说"说句话听听""语音回我"或撒娇场景适合语音时使用。

    Args:
        text: 要说的话（口语化，30 字内效果最好）
    """
    api_key = os.environ.get("DOUBAO_TTS_API_KEY", "")
    if not api_key:
        return "语音功能未配置（缺 DOUBAO_TTS_API_KEY），用文字回复吧。"

    from junjun_skills.plugins.tts.doubao_tts import doubao_tts_synthesize
    try:
        audio = await doubao_tts_synthesize(text[:100], api_key=api_key)
    except Exception as e:
        logger.warning(f"TTS 合成失败: {e}")
        return "语音合成失败了，用文字回复吧。"
    if not audio:
        return "语音合成结果为空，用文字回复吧。"

    # 存临时文件走 file:// 发送（NapCat record 段支持本地路径）
    tmp = Path(tempfile.gettempdir()) / f"junjun_tts_{abs(hash(text)) % 99999}.mp3"
    tmp.write_bytes(audio)

    chat_id = current_chat_id.get()
    parts = chat_id.split(":")
    platform, target_id, kind = parts[0], parts[1], parts[2] if len(parts) > 2 else "private"
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=[ReplySegment(type="voiceurl", data=tmp.resolve().as_uri())],
        should_reply=True,
    ))
    return "语音已发送。不要再用文字重复语音内容。"


def probe_available() -> bool:
    if not os.environ.get("DOUBAO_TTS_API_KEY"):
        logger.info("TTS 插件: DOUBAO_TTS_API_KEY 未设置，禁用")
        return False
    return True


TTS_TOOLS = [send_voice]
