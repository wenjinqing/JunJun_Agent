"""内置 skill：use_skill——取回 Agent 技能手册全文（md skills 的按需加载入口）。"""

from langchain_core.tools import tool


@tool
def use_skill(name: str) -> str:
    """查看你的技能手册全文。system prompt 里【技能包】列出了你有哪些手册及适用场景；
    当当前任务命中某个手册的场景时，先调这个工具拿到详细做法再动手。
    Args:
        name: 手册名（技能包列表里横线前的名字，如 video-watching）
    """
    from junjun_skills.skills_md import get_skill, skill_names
    body = get_skill(name)
    if body is not None:
        return body
    names = skill_names()
    if not names:
        return "（当前没有任何技能手册。）"
    return f"没有叫「{name}」的手册。现有手册：{', '.join(names)}"
