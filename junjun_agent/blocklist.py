"""屏蔽名单：管理员按会话拉黑某人（多为其他 bot）——被拉黑者的消息记完记忆
直接丢弃：不进决策队列、不触发命令/拦截器（0 token），@bot 也一样。

防 bot 互回循环的硬闸（2026-08-13 用户裁决：群里别的 bot 会跟君君互相回复，
客气话来回滚雪球烧 token）。注意「不搭理」不等于「看不见」——消息仍进短期
记忆与落库，人类接着聊时君君需要上下文。

命令（admin_only；群里用要 @君君 激活管理员权限，或私聊指定不了群——
名单按会话隔离，请在目标群里发）：
- /屏蔽 <QQ号>      本会话拉黑
- /取消屏蔽 <QQ号>  解除
- /屏蔽列表         看本会话拉黑了谁
"""

import os
import re
import time

from junjun_core.observability import get_logger
from junjun_agent.commands import register_command

logger = get_logger("blocklist")

_cache: "dict[str, set] | None" = None   # chat_id -> {user_id}；None=未从库里加载


def _load() -> None:
    global _cache
    if _cache is not None:
        return
    _cache = {}
    try:
        from junjun_core.database import BlockedUser
        for r in BlockedUser.select():
            _cache.setdefault(r.chat_id, set()).add(r.user_id)
    except Exception as e:
        logger.warning(f"屏蔽名单加载失败（按空名单降级）: {e}")


def is_blocked(chat_id: str, user_id: str) -> bool:
    """processor 每条消息的判定——纯内存读（名单极小，首次加载后零 IO）。"""
    if not user_id:
        return False
    _load()
    return user_id in _cache.get(chat_id, ())


def _extract_qq(args: str) -> str:
    """参数里抓 QQ 号（@段入站时已转成「@昵称 」文本，纯数字提取两种写法都兼容）。"""
    m = re.search(r"\d{5,12}", args or "")
    return m.group(0) if m else ""


def _reject_reason(user_id: str) -> str:
    """空串=允许操作。管理员（信任根）与 bot 自己不可被屏蔽。"""
    if not user_id:
        return "没说对象——格式：/屏蔽 QQ号"
    from junjun_core.security import get_admin_id
    if user_id == get_admin_id():
        return "不能屏蔽管理员本人。"
    if user_id == (os.environ.get("JUNJUN_QQ_ACCOUNT") or "\0"):
        return "不能屏蔽我自己呀。"
    return ""


def block(chat_id: str, user_id: str, by: str = "") -> tuple:
    """拉黑。返回 (成功与否, 失败原因)。"""
    reason = _reject_reason(user_id)
    if reason:
        return False, reason
    _load()
    if user_id in _cache.get(chat_id, ()):
        return False, f"{user_id} 本来就在屏蔽名单里。"
    try:
        from junjun_core.database import BlockedUser
        BlockedUser.create(chat_id=chat_id, user_id=user_id,
                           created_by=by, created_ts=time.time())
    except Exception as e:
        logger.warning(f"屏蔽名单写入失败: {type(e).__name__}: {e}")
        return False, f"写入失败（{type(e).__name__}），稍后再试。"
    _cache.setdefault(chat_id, set()).add(user_id)
    logger.info(f"[{chat_id}] 屏蔽 {user_id}（操作人 {by}）")
    return True, ""


def unblock(chat_id: str, user_id: str) -> tuple:
    _load()
    if user_id not in _cache.get(chat_id, ()):
        return False, f"{user_id} 不在本会话屏蔽名单里。"
    try:
        from junjun_core.database import BlockedUser
        BlockedUser.delete().where(
            BlockedUser.chat_id == chat_id,
            BlockedUser.user_id == user_id).execute()
    except Exception as e:
        logger.warning(f"屏蔽名单删除失败: {type(e).__name__}: {e}")
        return False, f"删除失败（{type(e).__name__}），稍后再试。"
    _cache[chat_id].discard(user_id)
    logger.info(f"[{chat_id}] 解除屏蔽 {user_id}")
    return True, ""


def list_blocked(chat_id: str) -> list:
    _load()
    return sorted(_cache.get(chat_id, ()))


def _reset_for_test() -> None:
    global _cache
    _cache = None


# ---------------------------------------------------------------- 管理员命令

@register_command("屏蔽", admin_only=True,
                  description="本会话不再回复某人（含@），防 bot 互回循环")
async def block_cmd(ctx):
    qq = _extract_qq(ctx.args)
    ok, msg = block(ctx.session.chat_id, qq, by=ctx.meta.user_id or "")
    if ok:
        return (f"好，本会话我不再理 {qq} 了（他说话我还是看得见，就是不回）。"
                f"想解除就 /取消屏蔽 {qq}")
    return msg


@register_command("取消屏蔽", admin_only=True,
                  description="解除本会话对某人的屏蔽")
async def unblock_cmd(ctx):
    qq = _extract_qq(ctx.args)
    if not qq:
        return "没说对象——格式：/取消屏蔽 QQ号"
    ok, msg = unblock(ctx.session.chat_id, qq)
    return f"好，恢复搭理 {qq} 了。" if ok else msg


@register_command("屏蔽列表", admin_only=True,
                  description="看本会话屏蔽了谁")
async def blocklist_cmd(ctx):
    blocked = list_blocked(ctx.session.chat_id)
    if not blocked:
        return "本会话没有屏蔽任何人。"
    return "本会话屏蔽名单：\n" + "\n".join(blocked)
