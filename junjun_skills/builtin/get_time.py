"""内置 skill: 获取当前时间。"""

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_time() -> str:
    """获取现在的日期、时间和星期几。当用户问时间、日期、今天星期几时使用。"""
    now = datetime.now()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return f"{now.strftime('%Y年%m月%d日 %H:%M:%S')} 星期{weekdays[now.weekday()]}"
