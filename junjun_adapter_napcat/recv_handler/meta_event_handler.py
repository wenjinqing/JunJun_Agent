"""元事件处理（心跳/生命周期）+ 心跳看门狗。

背景（2026-08-09「突然收不到消息、无断连日志」排查）：消息链路四段
QQ服务器 ↔ NapCat ↔ Adapter ↔ 网关，网关之前的三段此前没有任何
活性证据——出问题时只能靠猜。NapCat OneBot 侧配了 heartInterval=30s，
心跳本身就是最好的探针，且心跳体自带 status.online/good（NapCat 自认
与腾讯的连接状态）。读法：

- 心跳停 >90s      → NapCat↔Adapter WS 假死或 NapCat 进程卡死（重启 NapCat）
- 心跳在但 online/good=false → NapCat↔腾讯 协议层断了（等恢复或重启 NapCat）
- 心跳正常且在线    → NapCat 自认在线但没消息 → 腾讯侧吞消息（风控/群屏蔽）
"""

import asyncio
import time
from typing import Optional

from ..logger import logger

# NapCat onebot11 配置 heartInterval=30000；3 倍容忍抖动
HEARTBEAT_STALE_SECONDS = 90.0

_last_heartbeat_ts = 0.0   # 0 = 还没见过心跳（启动初期 NapCat 未连入，不告警）
_last_event_ts = 0.0       # 任何入站事件的活动时间（消息/通知/心跳）
_last_online = True        # NapCat 心跳自报的 online && good
_alarm_active = False      # 告警状态机：进入超时态报一次，恢复报一次


def note_activity() -> None:
    """任何入站事件（消息/通知/心跳）都更新活动时间。"""
    global _last_event_ts
    _last_event_ts = time.time()


def heartbeat_status() -> dict:
    """当前心跳状态快照（排查用）。"""
    now = time.time()
    return {
        "last_heartbeat_age": (now - _last_heartbeat_ts) if _last_heartbeat_ts else None,
        "last_event_age": (now - _last_event_ts) if _last_event_ts else None,
        "napcat_online": _last_online,
        "alarm_active": _alarm_active,
    }


def check_heartbeat(now: Optional[float] = None) -> Optional[str]:
    """检查心跳新鲜度，需要告警/恢复时返回日志文本，否则 None。

    状态机：未见心跳不告警（启动期）；超时进告警态只报一次；恢复报一次。
    """
    global _alarm_active
    if not _last_heartbeat_ts:
        return None
    age = (now if now is not None else time.time()) - _last_heartbeat_ts
    if age > HEARTBEAT_STALE_SECONDS and not _alarm_active:
        _alarm_active = True
        return (f"NapCat 心跳超时（{age:.0f}s 无心跳，间隔应 30s）——"
                "NapCat↔Adapter 连接假死或 NapCat 进程卡死，消息进不来；"
                "重启 NapCat 可恢复")
    if age <= HEARTBEAT_STALE_SECONDS and _alarm_active:
        _alarm_active = False
        return "NapCat 心跳恢复"
    return None


class MetaEventHandler:
    def __init__(self):
        pass

    async def handle_meta_event(self, raw_message: dict) -> None:
        global _last_heartbeat_ts, _last_online
        note_activity()
        # OneBot 11 字段为 meta_event_type（原 adapter 同名），不是 meta_type
        meta_type = raw_message.get("meta_event_type")
        if meta_type == "lifecycle":
            sub = raw_message.get("sub_type")
            if sub == "connect":
                logger.info("NapCat 已连接 (lifecycle connect)")
        elif meta_type == "heartbeat":
            _last_heartbeat_ts = time.time()
            status = raw_message.get("status") or {}
            online = bool(status.get("online", True)) and bool(status.get("good", True))
            if not online and _last_online:
                logger.warning(
                    f"NapCat 心跳自报离线/异常: {status} —— "
                    "NapCat↔腾讯 协议层断了（收不到消息但无断连日志的典型形态），"
                    "等它自己恢复或重启 NapCat")
            elif online and not _last_online:
                logger.info("NapCat 心跳自报恢复在线")
            _last_online = online
            logger.debug("NapCat heartbeat")


meta_event_handler = MetaEventHandler()


async def heartbeat_watchdog() -> None:
    """每 30s 检查一次心跳新鲜度（告警/恢复各报一次，不刷屏）。"""
    while True:
        await asyncio.sleep(30)
        try:
            msg = check_heartbeat()
            if msg:
                if "恢复" in msg:
                    logger.info(msg)
                else:
                    logger.warning(msg)
        except Exception as e:
            logger.warning(f"心跳看门狗自身异常（继续运行）: {type(e).__name__}: {e}")
