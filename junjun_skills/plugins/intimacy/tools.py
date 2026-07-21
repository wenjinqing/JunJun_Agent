"""intimacy 插件：好感度查询（迁移自 intimacy_query_plugin，新架构重写）。

累计层在 junjun_express/intimacy.py（core，processor 每条消息自动累计）。
本插件只提供查询面：
- raw 关键词命令：查看好感度 / 我的好感度 / 好感度查询 / 查好感度
- tool query_intimacy：LLM 被问关系/好感时拿数据自己组织语言
"""

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_core.observability import get_logger

logger = get_logger("plugin.intimacy")


def _describe(user_id: str, nickname: str) -> str:
    """好感度描述文本（分数 + 等级 + 画像摘要）。"""
    from junjun_express.intimacy import get_intimacy
    score, count, level = get_intimacy(user_id)
    text = (f"{nickname or '你'} 和我的好感度是 {score:.1f}/100（{level}），"
            f"一共互动过 {count} 次。")
    try:
        from junjun_memory.user_profile import get_profile_store
        block = get_profile_store().build_relation_block("qq", user_id, nickname)
        if block:
            text += f"\n{block}"
    except Exception:
        pass
    return text


@register_command("查看好感度", aliases=["我的好感度", "好感度查询", "查好感度"],
                  raw=True, plugin="intimacy", description="查看你和我之间的好感度")
async def intimacy_cmd(ctx):
    return _describe(ctx.meta.user_id or "", ctx.meta.nickname)


@tool
def query_intimacy() -> str:
    """查询当前对话用户和你的好感度/关系。被问"我们关系怎么样""你对我印象如何"
    "好感度多少"时使用，拿数据后用自己的话自然回答。"""
    from junjun_core.security import current_user_id
    from junjun_skills.builtin.memory_skills import current_chat_id
    uid = current_user_id.get()
    if not uid:
        # tool 执行在独立 context，ContextVar 可能丢失——从 chat_id 私聊兜底
        chat_id = current_chat_id.get()
        parts = chat_id.split(":")
        if len(parts) == 3 and parts[2] == "private":
            uid = parts[1]
    if not uid:
        return "暂时确定不了对方是谁，查不了好感度。"
    return _describe(uid, "")


TOOLS = [query_intimacy]
