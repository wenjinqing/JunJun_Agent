"""主动出站统一入口（严厉审查 P0-5/S2）。

系统曾有两条出站路径：主管线（processor：分条/清洗/记忆回填/落库）与
后台直发（tasks/async_jobs/订阅推送各写各的 gateway.send_reply，护栏全漏）。
每加一个出站护栏就得记得在直发路径重做一遍，漏一个就是一次线上事故
（「图呢」「一直说还在画」两次幻觉都源于这条裂缝）。

本模块是唯一指定直发口：路由解析 + 文本清洗 + 发送 + 记忆回填 + 出站落库。
所有「非回复当前消息」的主动发送（后台任务成品/任务汇报/订阅推送/定时推送）
都必须走这里。拦截器/命令路径由 CommandContext.send 自带回填（同语义）。
"""

from typing import List, Optional, Tuple

from junjun_core.contracts import ReplySegment, ReplySet
from junjun_core.observability import get_logger

logger = get_logger("agent.outbound")


def parse_route(chat_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    """chat_id（qq:ID:group|private）-> (platform, target_user_id, target_group_id)。"""
    parts = (chat_id or "").split(":")
    if len(parts) < 3:
        return parts[0] if parts and parts[0] else "qq", None, None
    platform, target, kind = parts[0], parts[1], parts[2]
    if kind == "group":
        return platform, None, target
    return platform, target, None


def _remember(chat_id: str, text: str) -> None:
    """出站回填：短期记忆（add_bot）+ 落库（_store_outbound）。

    直发消息不经过 inbound 管线，不手动记的话模型会忘了自己说过
    「画好啦/画砸了」，下轮被问「图呢」只能装傻（2026-08-04 实战）。
    """
    if not text:
        return
    try:
        from junjun_core.gateway.session_manager import get_session_manager
        s = get_session_manager().all_sessions().get(chat_id)
        if s is not None:
            if getattr(s, "memory", None) is not None:
                s.memory.add_bot(text)
            from junjun_agent.processor import _store_outbound
            _store_outbound(s, text)
    except Exception:
        pass


# ---------------------------------------------------------------- 全局日出站预算
# （2026-08-13 审查 P1）：主动搭话 loop 有自己的 per-chat 日限额
# （max_daily_proactive），但订阅推送/提醒/任务汇报/深研成品全都直走
# send_proactive，没有总闸——循环 bug 或订阅风暴就是无限刷屏。单点收口：
# 全局日预算 + 单会话日预算（[outbound] daily_global_budget/daily_chat_budget，
# 0=关闭对应维度），超限丢弃 + 当日首超私聊上报管理员（notify_admin 直走
# 网关不经本模块，无递归）。重启清零：可接受的粒度，持久化得不偿失。
_budget_day = ""
_budget_total = 0
_budget_per_chat: dict = {}
_budget_alerted = False


def _budget_cfg() -> Tuple[int, int]:
    try:
        from junjun_core.config import get_global_config
        raw = get_global_config().raw.get("outbound", {})
        return (int(raw.get("daily_global_budget", 300)),
                int(raw.get("daily_chat_budget", 60)))
    except Exception:
        return 300, 60


def _budget_reset_if_new_day() -> None:
    global _budget_day, _budget_total, _budget_per_chat, _budget_alerted
    import time as _time
    today = _time.strftime("%Y-%m-%d")
    if _budget_day != today:
        _budget_day, _budget_total = today, 0
        _budget_per_chat, _budget_alerted = {}, False


def _budget_check(chat_id: str) -> bool:
    """今日额度是否还够；够则记账。按尝试计（发送失败也计）——
    失败重试风暴同样烧预算，正是要拦的对象。"""
    global _budget_total
    _budget_reset_if_new_day()
    gb, cb = _budget_cfg()
    if gb > 0 and _budget_total >= gb:
        return False
    if cb > 0 and _budget_per_chat.get(chat_id, 0) >= cb:
        return False
    _budget_total += 1
    _budget_per_chat[chat_id] = _budget_per_chat.get(chat_id, 0) + 1
    return True


async def _budget_alert(chat_id: str, source: str) -> None:
    """当日首超私聊上报管理员——预算被打爆本身就是「哪里循环了」的信号。"""
    global _budget_alerted
    if _budget_alerted:
        return
    _budget_alerted = True
    try:
        from junjun_core.security import notify_admin
        await notify_admin(
            f"主动消息日预算已超限（{source} -> {chat_id}），"
            "今天的后续主动消息会被丢弃。如果不是在做批量推送，"
            "可能是哪里循环了，查一下日志。")
    except Exception:
        pass


def _reset_budget_for_test() -> None:
    """仅供测试。"""
    global _budget_day, _budget_total, _budget_per_chat, _budget_alerted
    _budget_day, _budget_total = "", 0
    _budget_per_chat, _budget_alerted = {}, False


async def send_proactive(chat_id: str, segments: List[ReplySegment], *,
                         source: str = "proactive", remember: bool = True) -> bool:
    """主动推送到会话。返回是否送达。

    - 文本段过主管线同款清洗（clean_markdown：去 markdown/星号表演）
    - 全局/单会话日预算闸（超限丢弃 + 当日首超上报管理员）
    - 发送成功且 remember=True 时回填记忆 + 落库（bot 记得自己说过什么）
    - 任何异常静默返回 False（直发绝不能炸主流程）
    """
    if not _budget_check(chat_id):
        logger.warning(f"[{chat_id}] 主动消息超出日预算被丢弃({source})")
        await _budget_alert(chat_id, source)
        return False
    cleaned: List[ReplySegment] = []
    for s in segments:
        if s.type == "text" and isinstance(s.data, str):
            try:
                from junjun_agent.postprocess.cleaner import clean_markdown
                s = ReplySegment(type="text", data=clean_markdown(s.data))
            except Exception:
                pass
        cleaned.append(s)
    try:
        from junjun_core.gateway.router import get_gateway
        platform, user_id, group_id = parse_route(chat_id)
        await get_gateway().send_reply(ReplySet(
            platform=platform,
            target_user_id=user_id,
            target_group_id=group_id,
            segments=cleaned,
            should_reply=True,
        ))
    except Exception as e:
        logger.warning(f"[{chat_id}] 主动推送失败({source}): {type(e).__name__}: {e}")
        return False
    if remember:
        text = "\n".join(s.data for s in cleaned
                         if s.type == "text" and isinstance(s.data, str)).strip()
        _remember(chat_id, text)
    return True
