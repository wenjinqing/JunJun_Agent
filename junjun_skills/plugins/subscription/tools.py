"""subscription 插件：常驻订阅——「帮我盯着 P 站作者 / B 站 UP 主，更新了告诉我」。

- Agent 自然语言创建订阅（subscribe_updates 工具），落 Subscription 表常驻
- 调度任务每 10 分钟扫一轮（每条订阅自带 interval_minutes 节流），
  有更新走 gateway 主动推到订阅时所在会话
- 基线策略：订阅当下以最新内容为基线，不轰炸历史内容
- 删除权限：创建者本人或管理员（真实 QQ 号判定，非聊天内容）
"""

import re
import time
from typing import Optional

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_agent.loop.scheduler import ScheduledTask, scheduler
from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.subscription")

_CHECK_INTERVAL = 600        # 调度轮询间隔（秒）
_DEFAULT_INTERVAL_MIN = 30   # 单条订阅默认检查间隔（分钟）
_MAX_PER_CHAT = 10           # 单会话订阅上限
_NOTIFY_CAP = 3              # 单次最多推送几条新内容

_KIND_NAMES = {"pixiv_author": "P 站作者", "bili_up": "B 站 UP 主"}


def _cfg() -> dict:
    """读取 [subscription] 配置节（热改生效）。"""
    try:
        return get_global_config().raw.get("subscription", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------- 各源检查器
# 统一签名：(sub) -> (items, name)；items 为 [{seen, title, url}] 新内容（旧->新），
# name 为目标显示名（拿不到给 ""）。失败/无更新返回 ([], name)。

async def _check_pixiv_author(sub) -> tuple:
    from junjun_skills.plugins.pixiv_novel import tools as pixiv
    data = await pixiv._fetch_author_works(sub.target_id)
    if data.get("error"):
        logger.warning(f"订阅检查 pixiv 作者 {sub.target_id} 失败: {data['error']}")
        return [], ""
    name = data.get("author") or ""
    try:
        baseline = int(sub.last_seen or 0)
    except ValueError:
        baseline = 0
    fresh = []
    for item in reversed(data.get("novels") or []):  # novels 最新在前，反转为旧->新
        nid = str(item.get("id") or "")
        if nid.isdigit() and int(nid) > baseline:
            fresh.append({
                "seen": nid,
                "title": item.get("title") or "(无标题)",
                "url": f"https://www.pixiv.net/novel/show.php?id={nid}",
            })
    return fresh, name


async def _check_bili_up(sub) -> tuple:
    from junjun_skills.plugins.bilibili import tools as bili
    params = await bili._wbi_sign({"mid": sub.target_id, "ps": 5, "order": "pubdate"})
    data = await bili._fetch_json("https://api.bilibili.com/x/space/arc/search", params)
    vlist = ((((data or {}).get("data") or {}).get("list") or {}).get("vlist")) or []
    if not vlist:
        return [], ""
    name = str(vlist[0].get("author") or "")
    try:
        baseline = int(float(sub.last_seen or 0))
    except ValueError:
        baseline = 0
    fresh = []
    for v in sorted(vlist, key=lambda x: int(x.get("created") or 0)):  # 旧->新
        created = int(v.get("created") or 0)
        bvid = str(v.get("bvid") or "")
        if created > baseline and bvid:
            fresh.append({
                "seen": str(created),
                "title": v.get("title") or "(无标题)",
                "url": f"https://www.bilibili.com/video/{bvid}",
            })
    return fresh, name


_CHECKERS = {"pixiv_author": _check_pixiv_author, "bili_up": _check_bili_up}


# ---------------------------------------------------------------- 目标解析

async def _resolve_bili_mid(target: str) -> tuple:
    """B 站目标解析：数字当 mid；否则按昵称搜用户取第一个。返回 (mid, name)。"""
    target = target.strip()
    if target.isdigit():
        return target, ""
    from junjun_skills.plugins.bilibili import tools as bili
    params = await bili._wbi_sign({"search_type": "bili_user", "keyword": target})
    data = await bili._fetch_json("https://api.bilibili.com/x/web-interface/search/type", params)
    users = ((data or {}).get("data") or {}).get("result") or []
    if not users:
        return "", ""
    first = users[0]
    uname = re.sub(r"</?em[^>]*>", "", str(first.get("uname") or ""))
    return str(first.get("mid") or ""), uname


def _resolve_pixiv_uid(target: str) -> str:
    """P 站目标解析：纯数字 UID 或主页 URL 提取。"""
    target = target.strip()
    if target.isdigit():
        return target
    m = re.search(r"users/(\d+)", target)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- 订阅 CRUD

def _create(kind: str, target_id: str, chat_id: str, user_id: str,
            nickname: str, baseline: str, name: str = ""):
    from junjun_core.database.models import Subscription
    return Subscription.create(
        kind=kind, target_id=target_id, target_name=name,
        chat_id=chat_id, user_id=user_id, user_nickname=nickname,
        last_seen=baseline,
        interval_minutes=int(_cfg().get("default_interval_minutes", _DEFAULT_INTERVAL_MIN)),
        enabled=True, created_at=time.time())


def _chat_sub_count(chat_id: str) -> int:
    from junjun_core.database.models import Subscription
    return (Subscription.select()
            .where((Subscription.chat_id == chat_id) & (Subscription.enabled == True))  # noqa: E712
            .count())


async def _baseline_for(kind: str, target_id: str) -> tuple:
    """订阅当下的基线（最新内容标记 + 显示名），防止把历史内容当更新轰炸。"""
    fake = type("Sub", (), {"target_id": target_id, "last_seen": "0"})()
    items, name = await _CHECKERS[kind](fake)
    return (items[-1]["seen"] if items else "0"), name


# ---------------------------------------------------------------- 调度检查

def _fmt_notify(sub, items: list) -> str:
    lines = [f"「订阅更新」你盯的{_KIND_NAMES.get(sub.kind, sub.kind)}「{sub.target_name or sub.target_id}」"
             f"有 {len(items)} 条新内容："]
    for it in items[:_NOTIFY_CAP]:
        lines.append(f"《{it['title']}》\n{it['url']}")
    return "\n".join(lines)


async def _notify(sub, items: list) -> None:
    parts = sub.chat_id.split(":")
    platform, target_id, kind = parts[0], parts[1], parts[2] if len(parts) > 2 else "private"
    from junjun_core.contracts import ReplySet, ReplySegment
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=[ReplySegment(type="text", data=_fmt_notify(sub, items))],
    ))


async def check_subscriptions() -> None:
    """调度入口：扫全部启用订阅，到间隔的逐个检查，有更新主动推送。"""
    if not bool(_cfg().get("enable", True)):
        return
    from junjun_core.database.models import Subscription
    now = time.time()
    subs = list(Subscription.select().where(Subscription.enabled == True))  # noqa: E712
    for sub in subs:
        if now - (sub.last_checked or 0) < (sub.interval_minutes or _DEFAULT_INTERVAL_MIN) * 60:
            continue
        try:
            checker = _CHECKERS.get(sub.kind)
            if not checker:
                continue
            items, name = await checker(sub)
            if name and name != sub.target_name:
                sub.target_name = name
            if items:
                await _notify(sub, items)
                sub.last_seen = items[-1]["seen"]
                logger.info(f"订阅 #{sub.id}（{sub.kind}:{sub.target_id}）推送 {len(items)} 条更新")
        except Exception as e:
            logger.warning(f"订阅 #{sub.id} 检查失败: {type(e).__name__}: {e}")
        finally:
            sub.last_checked = now
            try:
                sub.save()
            except Exception:
                pass


# ---------------------------------------------------------------- LLM 工具

@tool
async def subscribe_updates(source: str, target: str) -> str:
    """订阅创作者更新，对方有新作品时你会主动发消息通知。
    用户说「帮我盯着/关注一下/订阅/更新了告诉我」某个 P 站作者或 B 站 UP 主时使用。

    Args:
        source: pixiv（P 站小说作者）或 bilibili（B 站 UP 主）
        target: P 站作者 UID（数字或主页 URL）；B 站 UP 主的 mid 或昵称
    """
    from junjun_skills.builtin.memory_skills import current_chat_id
    from junjun_core.security import current_user_id, current_nickname

    kind_map = {"pixiv": "pixiv_author", "p站": "pixiv_author",
                "bilibili": "bili_up", "b站": "bili_up"}
    kind = kind_map.get(source.strip().lower())
    if not kind:
        return f"不认识订阅源「{source}」，目前支持 pixiv（P 站作者）和 bilibili（B 站 UP 主）。"

    chat_id = current_chat_id.get()
    user_id = current_user_id.get()
    if not chat_id:
        return "拿不到当前会话，订阅创建失败。"
    if _chat_sub_count(chat_id) >= int(_cfg().get("max_per_chat", _MAX_PER_CHAT)):
        return f"这个会话的订阅已经到上限（{_MAX_PER_CHAT} 条）了，先删几条再订。"

    if kind == "pixiv_author":
        target_id = _resolve_pixiv_uid(target)
        found_name = ""
        if not target_id:
            return "没看懂这个 P 站作者，给我作者 UID（数字）或主页链接。"
    else:
        target_id, found_name = await _resolve_bili_mid(target)
        if not target_id:
            return f"没找到叫「{target}」的 UP 主，可以直接给我 ta 的 mid（空间 URL 里的数字）。"

    baseline, name = await _baseline_for(kind, target_id)
    sub = _create(kind, target_id, chat_id, user_id, current_nickname.get(),
                  baseline, name or found_name)
    display = sub.target_name or target_id
    return (f"订阅好了：{_KIND_NAMES[kind]}「{display}」（#{sub.id}）。"
            f"每 {sub.interval_minutes} 分钟查一次，有更新我会主动来说。"
            f"现有内容不会轰炸你，只盯新发的。想取消说「取消订阅 {sub.id}」。")


# ---------------------------------------------------------------- 列表/取消（工具与命令共用）

def _do_list(chat_id: str) -> str:
    from junjun_core.database.models import Subscription
    subs = list(Subscription.select().where(
        (Subscription.chat_id == chat_id) & (Subscription.enabled == True)))  # noqa: E712
    if not subs:
        return "这个会话还没有订阅。想盯谁就说「帮我盯着 xxx」。"
    lines = ["当前订阅："]
    for s in subs:
        lines.append(f"- #{s.id} {_KIND_NAMES.get(s.kind, s.kind)}「{s.target_name or s.target_id}」"
                     f"（{s.user_nickname or s.user_id} 订的，每 {s.interval_minutes} 分钟查）")
    return "\n".join(lines)


def _do_unsub(sub_id: str, caller: str) -> str:
    from junjun_core.database.models import Subscription
    from junjun_core.security import is_admin
    sid = (sub_id or "").strip().lstrip("#")
    if not sid.isdigit():
        return "给我订阅编号（数字），可以先 /subs 查。"
    sub = Subscription.get_or_none(Subscription.id == int(sid))
    if sub is None or not sub.enabled:
        return f"没找到订阅 #{sid}。"
    if caller != sub.user_id and not is_admin(caller):
        return f"#{sid} 是 {sub.user_nickname or sub.user_id} 订的，只有本人或管理员能取消。"
    sub.enabled = False
    sub.save()
    return f"已取消订阅 #{sid}（{_KIND_NAMES.get(sub.kind, sub.kind)}「{sub.target_name or sub.target_id}」）。"


@tool
def list_subscriptions() -> str:
    """查看当前会话的所有订阅（盯梢列表）。用户问「我订了什么/你在盯哪些」时使用。"""
    from junjun_skills.builtin.memory_skills import current_chat_id
    return _do_list(current_chat_id.get())


@tool
def unsubscribe(sub_id: str) -> str:
    """取消订阅。用户说「取消订阅/别盯了」时使用。

    Args:
        sub_id: 订阅编号（list_subscriptions 可查）
    """
    from junjun_core.security import current_user_id
    return _do_unsub(sub_id, current_user_id.get())


TOOLS = [subscribe_updates, list_subscriptions, unsubscribe]


# ---------------------------------------------------------------- 命令

@register_command("subs", aliases=["订阅列表"], plugin="subscription",
                  description="查看本会话订阅")
async def subs_cmd(ctx) -> str:
    return _do_list(ctx.session.chat_id)


@register_command("unsub", aliases=["取消订阅"], plugin="subscription",
                  description="取消订阅（/unsub <编号>）")
async def unsub_cmd(ctx) -> str:
    from junjun_core.security import current_user_id
    if not (ctx.args or "").strip():
        return "用法：/unsub <编号>（/subs 查编号）"
    return _do_unsub(ctx.args.strip(), current_user_id.get())


scheduler.add(ScheduledTask("subscription_check", check_subscriptions,
                            interval=_CHECK_INTERVAL, plugin="subscription"))
