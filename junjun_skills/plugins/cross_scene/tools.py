"""cross_scene 插件：跨会话聊天记录查询（迁移自 cross_scene_chat_plugin，新架构重写）。

旧版是框架内部函数；新架构作为 tool 暴露，但属隐私敏感面——
只有管理员可用（工具层硬校验，越权自动上报），对齐安全体系。
"""

import time

from langchain_core.tools import tool

from junjun_core.observability import get_logger

logger = get_logger("plugin.cross_scene")

_MAX_LIMIT = 30


@tool
def query_cross_scene_chat(user_name: str = "", scene_type: str = "",
                           keyword: str = "", limit: int = 10) -> str:
    """查询某人在其他群/私聊里聊过的记录。被问"TA 之前在别的群说过什么"
    "我们私聊聊过这个吗"时使用。仅管理员可用（隐私保护）。

    Args:
        user_name: 昵称或 QQ 号（空=不限定人）
        scene_type: "group" 只看群 / "private" 只看私聊 / "" 全部
        keyword: 内容关键词（空=不限）
        limit: 最多返回条数，默认 10（上限 30）
    """
    from junjun_core.security import current_user_id, is_admin, report_violation
    from junjun_skills.builtin.memory_skills import current_chat_id

    cur_chat = current_chat_id.get()
    if not is_admin(current_user_id.get()):
        report_violation("跨会话查询聊天记录", current_user_id.get(), "", cur_chat,
                         f"user={user_name} scene={scene_type} kw={keyword}")
        return "查询被拒绝：跨会话聊天记录只有管理员能查（已通知管理员）。"

    from junjun_core.database.models import Messages
    q = Messages.select().where(Messages.chat_id != cur_chat)
    if user_name:
        q = q.where((Messages.user_nickname.contains(user_name))
                    | (Messages.user_id == user_name))
    if scene_type in ("group", "private"):
        q = q.where(Messages.chat_id.endswith(scene_type))
    if keyword:
        q = q.where(Messages.processed_plain_text.contains(keyword))
    rows = q.order_by(Messages.time.desc()).limit(max(1, min(_MAX_LIMIT, limit)))
    if not rows:
        return "其他会话里没有找到符合条件的记录。"
    lines = ["跨会话记录："]
    for r in rows:
        who = r.user_nickname or r.user_id or "我"
        when = time.strftime("%m-%d %H:%M", time.localtime(r.time))
        lines.append(f"- [{when}][{r.chat_id}] {who}: {r.processed_plain_text[:60]}")
    return "\n".join(lines)


TOOLS = [query_cross_scene_chat]
