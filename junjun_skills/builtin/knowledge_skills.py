"""知识库 skill：search_knowledge / import_knowledge。"""

from langchain_core.tools import tool

from junjun_skills.builtin.memory_skills import current_chat_id


@tool
async def search_knowledge(question: str) -> str:
    """查询知识库回答事实性问题。被问到设定/资料/文档类问题时使用（区别于 recall_memory 查聊天记忆）。

    Args:
        question: 要查的问题
    """
    from junjun_memory.knowledge import get_knowledge_base
    paras = await get_knowledge_base().search(question, top_k=3)
    if not paras:
        return "知识库里没有相关内容。"
    return "知识库相关内容：\n" + "\n---\n".join(p[:400] for p in paras)


@tool
async def import_knowledge(text: str) -> str:
    """把一段资料导入知识库长期保存。群友分享了值得保存的设定/教程/资料时使用。

    Args:
        text: 要导入的资料原文
    """
    from junjun_memory.knowledge import get_knowledge_base
    n = await get_knowledge_base().import_text(text)
    return "已导入知识库。" if n else "内容太短或已存在，未导入。"
