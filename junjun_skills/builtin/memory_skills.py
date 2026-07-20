"""记忆类内置 skill（阶段 4）：recall_memory / save_memory / manage_user_profile / query_jargon。

会话上下文通过 contextvar 注入（skill 执行时读当前 chat_id/platform）。
"""

from contextvars import ContextVar

from langchain_core.tools import tool

# processor 在每轮决策前设置；skill 执行时读取
current_chat_id: ContextVar[str] = ContextVar("junjun_chat_id", default="")
current_platform: ContextVar[str] = ContextVar("junjun_platform", default="qq")


@tool
async def recall_memory(query: str) -> str:
    """回忆过去聊过的内容。当被问"你还记得吗"、需要历史信息、或话题涉及以前的约定/事件时使用。

    Args:
        query: 要回忆的内容关键词，如"上次说的火锅店"
    """
    from junjun_memory.long_term import get_long_term_memory
    items = await get_long_term_memory().search(query, top_k=5, chat_id=current_chat_id.get() or None)
    if not items:
        # 放宽到全库再试一次
        items = await get_long_term_memory().search(query, top_k=3)
    if not items:
        return "没有找到相关记忆。"
    lines = ["相关记忆："]
    for it in items:
        lines.append(f"- {it.text}")
    return "\n".join(lines)


@tool
async def save_memory(content: str, importance: float = 0.8) -> str:
    """主动记住重要信息。当聊天中出现值得长期记住的事（约定、喜好、重要事件）时使用。

    Args:
        content: 要记住的内容，一句话概括，如"甲下周三过生日"
        importance: 重要程度 0-1，默认 0.8
    """
    from junjun_memory.long_term import get_long_term_memory
    await get_long_term_memory().add(
        content, current_chat_id.get(), weight=max(0.1, min(1.5, importance)), kind="fact",
    )
    return "已记住。"


@tool
def manage_user_profile(user_id: str, category: str, content: str) -> str:
    """更新对某个用户的了解。当用户透露自己的信息（名字、喜好、职业、关系）时使用。

    Args:
        user_id: 用户的 QQ 号（从消息上下文获取）
        category: 分类，如 喜好/身份/关系/称呼
        content: 具体内容，如"爱吃火锅"
    """
    from junjun_memory.user_profile import get_profile_store
    get_profile_store().add_point(current_platform.get(), user_id, category, content)
    return f"已更新对 {user_id} 的了解：{category} - {content}"


@tool
def query_jargon(term: str) -> str:
    """查询群黑话/梗的含义。遇到不懂的缩写、梗、圈内用语时使用。

    Args:
        term: 要查的词，如"awsl"
    """
    from junjun_express.jargon import lookup_jargon
    explanation = lookup_jargon(term, current_chat_id.get())
    return f"「{term}」的意思是：{explanation}" if explanation else f"黑话库里没有「{term}」的记录。"


@tool
def learn_jargon(term: str, explanation: str) -> str:
    """学习新黑话。当群友解释了某个梗/缩写的含义，或你从上下文推断出含义时使用。

    Args:
        term: 黑话词条
        explanation: 含义解释
    """
    from junjun_express.jargon import record_jargon
    record_jargon(term, explanation, current_chat_id.get())
    return f"学到了：「{term}」= {explanation}"
