"""动作类内置 skill（阶段 5）：send_message / send_poke / get_weather / query_chat_history。

发送统一走 gateway.send_reply（与 send_emoji 同一出口），不直接碰接入层。

安全：跨会话操作在工具层硬校验管理员身份（junjun_core.security），
不依赖 prompt 自觉——即使注入骗过模型，越权调用照样被拒并上报管理员。
"""

import time

from langchain_core.tools import tool

from junjun_core.security import current_user_id, is_admin, report_violation
from junjun_skills.builtin.memory_skills import current_chat_id


def _split_chat_id(chat_id: str) -> tuple:
    parts = chat_id.split(":")
    return parts[0], parts[1], parts[2] if len(parts) > 2 else "private"


def _target_chat_id(target_id: str, is_group: bool) -> str:
    return f"qq:{target_id}:{'group' if is_group else 'private'}"


@tool
async def send_message(target_id: str, is_group: bool, text: str) -> str:
    """向指定群或私聊主动发一条消息。提醒/约定/跨会话传话时使用；闲聊回复别用它（正常回复即可）。

    Args:
        target_id: 目标群号或 QQ 号
        is_group: true=群聊 false=私聊
        text: 要发送的文字
    """
    cur_chat = current_chat_id.get()
    if _target_chat_id(target_id, is_group) != cur_chat and not is_admin(current_user_id.get()):
        report_violation(
            "跨会话发消息", current_user_id.get(), "", cur_chat,
            f"目标 {'群' if is_group else '私聊'} {target_id}，内容: {text[:60]}",
        )
        return "发送被拒绝：向其他群/私聊发消息只有管理员能指挥我做（已通知管理员）。"
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform="qq",
        target_group_id=target_id if is_group else None,
        target_user_id=None if is_group else target_id,
        segments=[ReplySegment(type="text", data=text)],
        should_reply=True,
    ))
    return f"消息已发送到{'群' if is_group else '私聊'} {target_id}。"


@tool
async def send_poke(user_id: str) -> str:
    """戳一戳某人。被要求戳人、或想俏皮地提醒对方时注意你时使用。

    Args:
        user_id: 要戳的 QQ 号
    """
    from junjun_core.config import get_global_config
    if not get_global_config().raw.get("chat", {}).get("enable_poke", True):
        return "戳一戳功能已被配置关闭（enable_poke=false）。"
    platform, target_id, kind = _split_chat_id(current_chat_id.get())
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=[ReplySegment(type="poke", data=user_id)],
        should_reply=True,
    ))
    return f"已戳了戳 {user_id}。"


@tool
async def get_weather(city: str) -> str:
    """查询天气。被问天气、温度、要不要带伞时使用。

    Args:
        city: 城市名，如"上海"
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://wttr.in/{city}",
                params={"format": "j1", "lang": "zh"},
                headers={"User-Agent": "curl/8"},
            )
            data = resp.json()
        cur = data["current_condition"][0]
        desc = cur["lang_zh"][0]["value"] if cur.get("lang_zh") else cur["weatherDesc"][0]["value"]
        return (
            f"{city}现在{desc}，气温{cur['temp_C']}°C（体感{cur['FeelsLikeC']}°C），"
            f"湿度{cur['humidity']}%，风速{cur['windspeedKmph']}km/h。"
        )
    except Exception as e:
        return f"天气查询失败了（{e}），稍后再试试吧。"


@tool
def query_chat_history(keyword: str, limit: int = 10) -> str:
    """搜索当前会话的聊天记录。被问"之前谁说过什么"、需要精确翻近期聊天原文时使用
    （模糊的久远记忆用 recall_memory）。

    Args:
        keyword: 要搜索的关键词
        limit: 最多返回条数，默认 10
    """
    from junjun_core.database.models import Messages
    chat_id = current_chat_id.get()
    rows = (
        Messages.select()
        .where(
            (Messages.chat_id == chat_id)
            & (Messages.processed_plain_text.contains(keyword))
        )
        .order_by(Messages.time.desc())
        .limit(max(1, min(50, limit)))
    )
    if not rows:
        return f"近期聊天记录里没有找到含「{keyword}」的消息。"
    lines = [f"含「{keyword}」的最近消息："]
    for r in rows:
        who = r.user_nickname or r.user_id or "我"
        when = time.strftime("%m-%d %H:%M", time.localtime(r.time))
        lines.append(f"- [{when}] {who}: {r.processed_plain_text[:80]}")
    return "\n".join(lines)
