"""发送重试（2026-08-04 用户反馈：消息一次失败就丢）。

原链路：SendHandler 调一次 OneBot API，status != ok 就打条 warning 丢弃。
分层重试策略：

- 「no connection」/发送异常：请求大概率没离开本机 -> 直接重发
- 确定性失败签名（rich media transfer failed 等）：传输/解析层明确失败，
  直接补发——绝不走历史确认。2026-08-15 实锤：视频上传失败后 NTQQ 本地
  历史仍留「发送方可见的失败残影」（群友实际没收到，但 get_group_msg_history
  能查到该消息、连它的视频 URL 都取不出），历史确认被残影骗过吞掉补发
- 响应超时 / 其他 retcode 失败：NapCat EventChecker 有「实际送达却报失败」的
  误报前科（2026-08-03 实测），盲重发会刷屏 -> 先查消息历史确认没送达
  再补发一次
- 历史确认的比口径（2026-08-15 加严）：
  · 只认 bot 自己发的消息（self_id == sender）——群友同文本不算送达证据
  · 发的是媒体消息时，历史记录必须带同款 CQ 媒体段——同文本的纯文本
    消息（如上一条回执）不算媒体送达
- 无文本（纯图/语音/视频）：等几秒补发一次（retcode 1200 下载超时 = 真没发，
  EventChecker 误报率可接受）

首次失败的原始错误一律落日志（retcode/message/wording）——此前只进内存，
历史确认误杀时全链路只剩一句「误报不重发」，真实死因无痕。
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

# 确定性失败签名：命中即跳过历史确认直接补发（历史残影会骗过确认，见模块头）
_DEFINITIVE_FAIL_MARKERS = (
    "rich media transfer failed",  # 富媒体上传失败（2026-08-15 实锤）
    "识别url失败", "文件处理失败",    # NapCat 打不开文件/URL
    "packet cant get",             # 媒体包抓取失败
)


def _text_fingerprint(params: dict) -> str:
    """从 OneBot message 数组提取文本指纹（用于历史比对）；无文本返回 ""。"""
    parts = []
    for seg in params.get("message") or []:
        if isinstance(seg, dict) and seg.get("type") == "text":
            parts.append(str(seg.get("data", {}).get("text", "")))
    return "".join(parts).replace(" ", "").replace("　", "")[:60]


def _media_tag(params: dict) -> str:
    """消息里第一个媒体段的 CQ 前缀（历史比对用）；纯文本返回 ""。"""
    for seg in params.get("message") or []:
        if isinstance(seg, dict) and seg.get("type") in ("image", "video", "record"):
            return f"[CQ:{seg['type']}"
    return ""


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
                                *, since: float, media_tag: str = "") -> bool:
    """历史里近 _VERIFY_WINDOW 秒出现【bot 自己发的】同文本（媒体需带同款段）
    -> 视为已送达。查询失败按未送达处理。"""
    resp = await nc_message_sender.send_message_to_napcat(h_action, h_params)
    if resp.get("status") != "ok":
        logger.warning(f"发送确认：历史查询失败（按未送达补发）: {resp}")
        return False
    messages = (resp.get("data") or {}).get("messages") or []
    for m in messages:
        if since - float(m.get("time", 0)) > _VERIFY_WINDOW:
            continue
        # 只认自己发的：群友/别的号发的同文本不是送达证据。
        # self_id/user_id 缺失的老 NapCat 降级为只比对文本（旧行为）。
        self_id = str(m.get("self_id") or "")
        sender_id = str((m.get("sender") or {}).get("user_id")
                        or m.get("user_id") or "")
        if self_id and sender_id and sender_id != self_id:
            continue
        raw = str(m.get("raw_message", "")).replace(" ", "").replace("　", "")
        if fingerprint and fingerprint in raw:
            # 发的是媒体消息：历史记录必须带同款媒体段，防同文本纯文本假确认
            if media_tag and media_tag not in raw:
                continue
            return True
    return False


async def send_with_retry(action: str, params: dict) -> dict:
    """发送 + 分层重试。返回最终响应（status == ok 即送达）。"""
    resp = await nc_message_sender.send_message_to_napcat(action, params)
    if resp.get("status") == "ok":
        return resp
    err = str(resp.get("message", ""))
    # 原始错误必须落日志——历史确认误杀时这是唯一能还原死因的痕迹
    logger.warning(f"[{action}] 首次发送失败: retcode={resp.get('retcode')} "
                   f"message={err} wording={resp.get('wording', '')}")
    sent_at = time.time()

    # 传输层失败：请求没执行，直接重发
    if err == "no connection":
        logger.warning(f"[{action}] 无连接，等重连后直接重发")
        return await nc_message_sender.send_message_to_napcat(action, params)

    # 确定性失败：本地历史会留失败残影骗过确认，跳过历史直接补发一次
    low = f"{err} {resp.get('wording', '')}".lower()
    if any(mk in low for mk in _DEFINITIVE_FAIL_MARKERS):
        await asyncio.sleep(_MEDIA_RETRY_DELAY)
        logger.warning(f"[{action}] 确定性失败（{err[:60]}），跳过历史确认直接补发")
        return await nc_message_sender.send_message_to_napcat(action, params)

    history = _history_action(action, params)
    fingerprint = _text_fingerprint(params)

    if history and fingerprint:
        # EventChecker 误报前科：先查历史确认没送达再补发
        await asyncio.sleep(_VERIFY_DELAY)
        if await _delivered_in_history(history[0], history[1], fingerprint,
                                       since=sent_at, media_tag=_media_tag(params)):
            logger.info(f"[{action}] 报错但历史确认已送达（EventChecker 误报），不重发")
            return {"status": "ok", "_verified": True}
        logger.warning(f"[{action}] 历史确认未送达，补发一次")
        return await nc_message_sender.send_message_to_napcat(action, params)

    # 无文本指纹（纯媒体）或无法查历史：等几秒盲补一次
    await asyncio.sleep(_MEDIA_RETRY_DELAY)
    logger.warning(f"[{action}] 无文本指纹，直接补发一次")
    return await nc_message_sender.send_message_to_napcat(action, params)
