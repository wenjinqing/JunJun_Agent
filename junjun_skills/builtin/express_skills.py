"""表达系统 skill：send_emoji（发表情包）。"""

from langchain_core.tools import tool

from junjun_skills.builtin.memory_skills import current_chat_id


@tool
async def send_emoji(emotion_or_context: str) -> str:
    """发一张表情包辅助表达情绪。回复内容情绪浓（开心/嘲讽/无语等）时可偶尔使用，别刷屏。

    Args:
        emotion_or_context: 想表达的情感或语境，如"开心""无语""被夸了得意"
    """
    from junjun_express.emoji import emoji_manager
    chat_id = current_chat_id.get()
    picked = emoji_manager.pick(emotion_or_context, chat_id)
    if picked is None:
        return "表情包冷却中或库存为空，这次就不发了。"

    # 直发（不进回复后处理——表情包独立一条）
    parts = chat_id.split(":")
    platform, target_id, kind = parts[0], parts[1], parts[2] if len(parts) > 2 else "private"
    from pathlib import Path
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    file_uri = Path(picked["path"]).resolve().as_uri()  # file:///E:/... NapCat 支持
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=[ReplySegment(type="image", data=file_uri)],
        should_reply=True,
    ))
    return f"表情包已发（{picked['description'][:30]}）。不要在文字回复里重复描述它。"
