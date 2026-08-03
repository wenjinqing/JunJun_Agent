"""动作类内置 skill（阶段 5）：send_message / send_poke / get_weather / query_chat_history。

发送统一走 gateway.send_reply（与 send_emoji 同一出口），不直接碰接入层。

安全：跨会话操作在工具层硬校验管理员身份（junjun_core.security），
不依赖 prompt 自觉——即使注入骗过模型，越权调用照样被拒并上报管理员。
"""

import time

from langchain_core.tools import tool

from junjun_core.security import current_user_id, is_admin_privileged, report_violation
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
    if _target_chat_id(target_id, is_group) != cur_chat and not is_admin_privileged():
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


_poke_last: dict = {}  # (chat_id, target) -> ts，5 分钟内重复戳同一人拒绝（对齐旧 acpoke）
_POKE_REPEAT_WINDOW = 300.0


async def _resolve_poke_target(name_or_id: str, chat_id: str) -> str:
    """QQ 号直接返回；昵称/群名片经 NapCat 群成员列表模糊解析。找不到返回空串。"""
    name_or_id = name_or_id.strip().lstrip("@")
    if name_or_id.isdigit():
        return name_or_id
    platform, target_id, kind = _split_chat_id(chat_id)
    if kind != "group":
        return ""
    from junjun_core import napcat_client
    members = await napcat_client.get_group_members(target_id) or []
    name_or_id = name_or_id.lower()
    for m in members:
        names = (str(m.get("card") or ""), str(m.get("nickname") or ""))
        if any(name_or_id == n.lower() for n in names if n):
            return str(m.get("user_id"))
    for m in members:  # 精确不中退到包含匹配
        names = (str(m.get("card") or ""), str(m.get("nickname") or ""))
        if any(name_or_id in n.lower() for n in names if n):
            return str(m.get("user_id"))
    return ""


@tool
async def send_poke(user_id: str) -> str:
    """戳一戳某人。被要求戳人、或想俏皮地提醒对方时注意你时使用。

    Args:
        user_id: 要戳的 QQ 号，也可以是群昵称/群名片（群聊里自动解析）
    """
    from junjun_core.config import get_global_config
    if not get_global_config().raw.get("chat", {}).get("enable_poke", True):
        return "戳一戳功能已被配置关闭（enable_poke=false）。"
    chat_id = current_chat_id.get()
    target = await _resolve_poke_target(user_id, chat_id)
    if not target:
        return f"群里没找到「{user_id}」这个人，换 QQ 号试试。"
    now = time.time()
    key = (chat_id, target)
    if now - _poke_last.get(key, 0) < _POKE_REPEAT_WINDOW:
        return f"刚戳过 {user_id} 了，歇会儿再戳。"
    platform, target_id, kind = _split_chat_id(chat_id)
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=[ReplySegment(type="poke", data=target)],
        should_reply=True,
    ))
    _poke_last[key] = now
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


# 聊天搜索限流（P6-5）：防 LLM 每轮都搜（延迟+成本）。命中才占额度之外的
# 粗暴防抖：不管有没有搜到都占——防「换个词再搜」连发。
_SEARCH_LOG: dict = {}  # chat_id -> deque(搜索时间戳，10 分钟滑窗)
_SEARCH_MAX = 5         # 每会话每 10 分钟最多 5 次
_SEARCH_WINDOW = 600.0


def _search_rate_limited(chat_id: str) -> bool:
    from collections import deque
    now = time.time()
    dq = _SEARCH_LOG.setdefault(chat_id, deque())
    while dq and now - dq[0] > _SEARCH_WINDOW:
        dq.popleft()
    if len(dq) >= _SEARCH_MAX:
        return True
    dq.append(now)
    return False


@tool
def query_chat_history(keyword: str, user: str = "", days: int = 30, limit: int = 8) -> str:
    """搜索**当前会话**的聊天记录原文（精确事实查找）。被问「上次谁说过什么/
    他发的那家店叫什么」这类需要翻原文的问题时使用；模糊的久远记忆用 recall_memory，
    不要每轮都搜。
    隐私边界写死：只能搜当前会话——群里搜不到任何私聊记录，私聊也搜不到群记录。

    Args:
        keyword: 要搜索的关键词
        user: 只看某个人发的（昵称片段或 QQ 号，留空=所有人）
        days: 搜最近几天的，默认 30；0 = 全部历史
        limit: 最多返回条数，默认 8（上限 8）
    """
    from junjun_core.database.models import Messages
    chat_id = current_chat_id.get()
    if _search_rate_limited(chat_id):
        return "刚搜过好几次了，先歇会儿再查（每 10 分钟最多搜 5 次）。"
    cond = ((Messages.chat_id == chat_id)
            & (Messages.processed_plain_text.contains(keyword)))
    if user:
        cond &= (Messages.user_nickname.contains(user) | (Messages.user_id == user))
    if days and days > 0:
        cond &= (Messages.time >= time.time() - days * 86400)
    rows = (
        Messages.select()
        .where(cond)
        .order_by(Messages.time.desc())
        .limit(max(1, min(8, limit)))
    )
    scope = f"最近 {days} 天" if days and days > 0 else "全部历史"
    who = f"{user} 发的" if user else ""
    if not rows:
        return f"{scope}的聊天记录里没有找到{who}含「{keyword}」的消息。"
    lines = [f"{scope}含「{keyword}」的{who}最近消息："]
    for r in rows:
        name = r.user_nickname or r.user_id or "我"
        when = time.strftime("%m-%d %H:%M", time.localtime(r.time))
        lines.append(f"- [{when}] {name}: {r.processed_plain_text[:80]}")
    return "\n".join(lines)


# 跨群围观念限流（独立于搜索额度）：每会话每 10 分钟最多 3 次
_PEEK_LOG: dict = {}
_PEEK_MAX = 3


def _peek_rate_limited(chat_id: str) -> bool:
    from collections import deque
    now = time.time()
    dq = _PEEK_LOG.setdefault(chat_id, deque())
    while dq and now - dq[0] > _SEARCH_WINDOW:
        dq.popleft()
    if len(dq) >= _PEEK_MAX:
        return True
    dq.append(now)
    return False


@tool
def peek_group_chat(group: str = "") -> str:
    """看看别的群最近在聊什么（仅私聊可用）。对方私聊里好奇「其他群/某个群
    最近在聊什么」时使用：留空给所有群的近况概览（每群最近几条 + 群号），
    指定群号则看那个群最近 20 条。
    隐私边界写死：只读群聊消息——任何私聊记录（你的/别人的）永远拿不到；
    群聊里本工具不可用（A 群的事不在 B 群说），只在私聊满足好奇心。

    Args:
        group: 群号（留空 = 所有群概览，概览里能看到各群群号）
    """
    from junjun_core.database.models import Messages
    chat_id = current_chat_id.get()
    if _peek_rate_limited(chat_id):
        return "刚看过好几眼了，歇会儿再看（每 10 分钟最多 3 次）。"
    rows = list(
        Messages.select()
        .where(Messages.group_id != "")  # 只读群聊：私聊消息 group_id 恒为空
        .order_by(Messages.time.desc())
        .limit(200)
    )
    if not rows:
        return "最近群里都没什么动静。"

    def _fmt(r):
        name = "我" if r.is_bot else (r.user_nickname or r.user_id or "?")
        when = time.strftime("%m-%d %H:%M", time.localtime(r.time))
        return f"- [{when}] {name}: {r.processed_plain_text[:60]}"

    group = (group or "").strip()
    if group:
        picked = [r for r in rows if group in (r.group_id or "") or group in r.chat_id][:20]
        if not picked:
            return f"没找到群号含「{group}」的群（先留空调一次看看有哪些群和群号）。"
        picked.reverse()  # 时间正序，读得顺
        gid = picked[0].group_id
        lines = [f"群 {gid} 最近的聊天："] + [_fmt(r) for r in picked]
        return "\n".join(lines)

    # 概览：最近活跃的 5 个群，每群最近 5 条
    by_group: dict = {}
    for r in rows:
        by_group.setdefault(r.chat_id, []).append(r)  # rows 已是时间倒序
    lines = ["各个群最近的动静（想看哪个群细节，再用群号调一次）："]
    for cid, rs in list(by_group.items())[:5]:
        lines.append(f"【群 {rs[0].group_id}】")
        lines.extend(_fmt(r) for r in reversed(rs[:5]))
    return "\n".join(lines)
