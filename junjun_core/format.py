"""通用格式化小函数。

2026-08-13 审查 P2：_fmt_size 在 adapter message_handler 与 workspace 插件
各抄了一份（后者还没 GB 档和零值兜底）——收敛到此处单一实现。
"""


def fmt_size(n: int) -> str:
    """文件大小人性化：1234567 -> "1.4MB"；n<=0（0 字节/字段缺失）-> "大小未知"。"""
    if n <= 0:
        return "大小未知"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n}B"
