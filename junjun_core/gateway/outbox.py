"""WS outbox：adapter 断连期的出站消息暂存与重连回放（路线图候选 C）。

触发背景：用户报过「突然收不到消息/君君突然不理人」——gateway→adapter
断连时 send_reply 只记日志丢弃。设计（观察清单候选 C 的触发条件已满足）：
- 发送失败（无连接/异常）→ 落 OutboxMessage 表（走 db_writer 不挡事件循环）
- 回放触发：① 周期性 flush loop（默认 30s）② 该平台有入站消息（=连接活着）
- 防轰炸：TTL（默认 30 分钟，陈旧回复不像真人）+ 次数上限 + 单次批量上限
- 顺序：按 created_ts FIFO；某条失败就停（保住顺序，下轮继续）
"""

import json
import time
from typing import Optional

from junjun_core.observability import get_logger

logger = get_logger("gateway.outbox")

# 内存脏标记：平台 -> 有待回放。免每消息查库；flush 清空。
_dirty: dict = {}


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("gateway", {})
    except Exception:
        return {}


def enabled() -> bool:
    return bool(_cfg().get("outbox", True))


def mark_dirty(platform: str) -> None:
    _dirty[platform] = True


def enqueue(reply, payload: dict) -> None:
    """发送失败的回复暂存。

    reply: ReplySet（取 platform/目标）；payload: 调用方已构造好的
    MessageBase dict（send_reply 里 msg_base.to_dict()），避免重复构造。"""
    if not enabled():
        return
    from junjun_core.database import db_writer
    from junjun_core.database.models import OutboxMessage
    payload_json = json.dumps(payload, ensure_ascii=False)
    now = time.time()

    def _insert():
        OutboxMessage.create(
            platform=reply.platform,
            target_group_id=reply.target_group_id or "",
            target_user_id=reply.target_user_id or "",
            payload_json=payload_json, created_ts=now, attempts=0)

    try:
        db_writer.submit(_insert)
        mark_dirty(reply.platform)
        logger.warning(f"回复已暂存 outbox [{reply.platform}] "
                       f"目标={reply.target_group_id or reply.target_user_id}（重连后回放）")
    except Exception as e:
        logger.error(f"outbox 暂存失败，本条回复丢失: {type(e).__name__}: {e}")


async def flush(server, platform: str, *, batch: int = 50) -> int:
    """回放该平台暂存消息。返回成功条数。server: MessageServer 实例。

    次数语义：广播异常（timeout 等）才扣 attempts；「没有活连接」不扣——
    那不是消息的错，等连接回来再投。FIFO 保序：撞到第一个失败就停本轮。"""
    if not _dirty.get(platform):
        return 0
    from junjun_core.database import db_writer
    from junjun_core.database.models import OutboxMessage

    ttl = float(_cfg().get("outbox_ttl_seconds", 1800))
    max_attempts = int(_cfg().get("outbox_max_attempts", 10))
    now = time.time()
    try:
        rows = list(OutboxMessage.select()
                    .where(OutboxMessage.platform == platform)
                    .order_by(OutboxMessage.created_ts).limit(batch))
    except Exception as e:
        logger.warning(f"outbox 读取失败: {e}")
        return 0
    if not rows:
        _dirty.pop(platform, None)
        return 0

    sent = 0
    delete_ids: list = []
    bump_ids: list = []
    n_expired = n_doomed = 0
    for row in rows:
        if now - row.created_ts > ttl:
            delete_ids.append(row.id)
            n_expired += 1
            continue
        if row.attempts >= max_attempts:
            delete_ids.append(row.id)
            n_doomed += 1
            continue
        try:
            ok = await server.broadcast_to_platform(platform, json.loads(row.payload_json))
        except Exception as e:
            logger.warning(f"outbox 回放异常（扣一次机会，停本轮保序）: {type(e).__name__}: {e}")
            bump_ids.append(row.id)
            break
        if ok is False:
            break  # 没有活连接：不扣次数，下轮再试
        sent += 1
        delete_ids.append(row.id)

    for rid in delete_ids:
        db_writer.submit(lambda rid=rid:
                         OutboxMessage.delete().where(OutboxMessage.id == rid).execute())
    for rid in bump_ids:
        db_writer.submit(lambda rid=rid:
                         OutboxMessage.update(attempts=OutboxMessage.attempts + 1)
                         .where(OutboxMessage.id == rid).execute())
    if n_expired:
        logger.info(f"outbox 丢弃 {n_expired} 条过期回复（>{ttl:.0f}s，陈旧消息不回放）")
    if n_doomed:
        logger.warning(f"outbox 丢弃 {n_doomed} 条超限回复（attempts≥{max_attempts}）")
    if sent:
        logger.info(f"outbox 回放 [{platform}] {sent} 条")
    return sent


async def maybe_flush(server, platform: str) -> None:
    """入站消息证明连接活着：该平台有积压就立刻回放（不等周期 loop）。"""
    if _dirty.get(platform) and server is not None:
        await flush(server, platform)


async def flush_loop(get_server, *, interval: Optional[float] = None) -> None:
    """周期回放守护：对所有脏平台尝试 flush。get_server: () -> MessageServer|None。"""
    import asyncio
    if interval is None:
        interval = float(_cfg().get("outbox_flush_seconds", 30))
    while True:
        await asyncio.sleep(interval)
        if not enabled():
            continue
        server = get_server()
        if server is None:
            continue
        for platform in list(_dirty):
            try:
                # 平台没有活连接时跳过（省一次广播异常），让 TTL 自然清理
                conns = getattr(server, "platform_connections", {}).get(platform)
                if not conns:
                    continue
                await flush(server, platform)
            except Exception as e:
                logger.warning(f"outbox flush [{platform}] 异常: {type(e).__name__}: {e}")
