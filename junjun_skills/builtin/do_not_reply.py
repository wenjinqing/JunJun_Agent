"""内置 skill: 显式沉默。

Agent 判断不该回复时调用。agent 层通过扫描消息历史中的 do_not_reply
tool_call 判定沉默（LangChain 工具在独立 context 执行，contextvar 写不回；
哨兵文本方案又有泄漏风险，故用 tool_call 记录本身作为信号）。
"""

from langchain_core.tools import tool

SILENCE_TOOL_NAME = "do_not_reply"


@tool
def do_not_reply(reason: str) -> str:
    """本条消息不需要回复时调用（闲聊与你无关、刚回复过不宜刷屏、话题不适合插话）。

    Args:
        reason: 不回复的简短原因
    """
    return f"好的，本次保持沉默（{reason}）。不要再输出任何内容。"
