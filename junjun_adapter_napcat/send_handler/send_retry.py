"""发送重试（2026-08-04 用户反馈：消息一次失败就丢）。

原链路：SendHandler 调一次 OneBot API，status != ok 就打条 warning 丢弃。
分层重试策略：

- 「no connection」/发送异常：请求大概率没离开本机 -> 直接重发
- 响应超时 / retcode 失败：NapCat EventChecker 有「实际送达却报失败」的
  误报前科（2026-08-03 实测），盲重发会刷屏 -> 先查消息历史确认没送达
  再补发一次
- 有文本指纹：历史近 _VERIFY_WINDOW 秒内出现同文本 -> 视为已送达，不重发
- 无文本（纯图/语音/视频）：等几秒补发一次（retcode 1200 下载超时 = 真没发，
  EventChecker 误报率可接受）
"""

import asyncio
import time
from typing import Optional

from ..logger import logger
from .nc_sending import nc_message_sender

_VERIFY_DELAY = 2.0     # 查历史前等几秒（等历史落库）
_MEDIA_RETRY_DELAY = 3.0
_VERIFY_WINDOW = 120    # 历史指纹比对窗口（秒）
_HISTORY_COUNT = 10


def _text_fingerprint(params: dict) -> str:
    """从 OneBot message 数组提取文本指纹（用于历史比对）；无文本返回 ""。"""
    parts = []
    for seg in params.get("message") or []:
        if isinstance(seg, dict) and seg.get("type") == "text":
            parts.append(str(seg.get("data", {}).get("text", "")))
    return "".join(parts).replace(" ", "").replace("　", "")[:60]


def _history_action(action: str, params: dict) -> Optional[tuple]:
    """主发送 action -> (历史查询 action, 历史查询参数)。"""
    if action == "send_group_msg":
        return "get_group_msg_history", {"group_id": params["group_id"],
                                         "count": _HISTORY_COUNT}
    if action == "send_private_msg":
        return "get_friend_msg_history", {"user_id": params["user_id"],
                                          "count": _HISTORY_COUNT}
    return None


async def _delivered_in_history(h_action: str, h_params: dict, fingerprint: str,
                                *, since: float) -> bool:
    """历史里近 _VERIFY_WINDOW 秒出现同文本 -> 视为已送达。查询失败按未送达处理。"""
    resp = await nc_message_sender.send_message_to_napcat(h_action, h_params)
    if resp.get("status") != "ok":
        logger.warning(f"发送确认：历史查询失败（按未送达补发）: {resp}")
        return False
    messages = (resp.get("data") or {}).get("messages") or []
    for m in messages:
        if since - float(m.get("time", 0)) > _VERIFY_WINDOW:
            continue
        raw = str(m.get("raw_message", "")).replace(" ", "").replace("　", "")
        if fingerprint and fingerprint in raw:
            return True
    return False


async def send_with_retry(action: str, params: dict) -> dict:
    """发送 + 分层重试。返回最终响应（status == ok 即送达）。"""
    resp = await nc_message_sender.send_message_to_napcat(action, params)
    if resp.get("status") == "ok":
        return resp
    err = str(resp.get("message", ""))
    sent_at = time.time()

    # 传输层失败：请求没执行，直接重发
    if err == "no connection":
        logger.warning(f"[{action}] 无连接，等重连后直接重发")
        return await nc_message_sender.send_message_to_napcat(action, params)

    history = _history_action(action, params)
    fingerprint = _text_fingerprint(params)

    if history and fingerprint:
        # EventChecker 误报前科：先查历史确认没送达再补发
        await asyncio.sleep(_VERIFY_DELAY)
        if await _delivered_in_history(history[0], history[1], fingerprint,
                                       since=sent_at):
            logger.info(f"[{action}] 报错但历史确认已送达（EventChecker 误报），不重发")
            return {"status": "ok", "_verified": True}
        logger.warning(f"[{action}] 历史确认未送达，补发一次")
        return await nc_message_sender.send_message_to_napcat(action, params)

    # 无文本指纹（纯媒体）或无法查历史：等几秒盲补一次
    await asyncio.sleep(_MEDIA_RETRY_DELAY)
    logger.warning(f"[{action}] 无文本指纹，直接补发一次")
    return await nc_message_sender.send_message_to_napcat(action, params)
