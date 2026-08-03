"""pixiv 插件：P 站内容统一获取（官方 AJAX API）。

子模块：
- client.py  共享 HTTP/cookie/反爬层
- setu.py    /setu 随机图（官方搜索/排行，修复裸词静默丢弃 bug）
- illust.py  /pixiv 命令族（search/illust/rank/author/new/dl）
- novel.py   /novel 小说下载（从 pixiv_novel 迁入，逻辑不变）

全部走官方接口（2026-08-03 起），不再依赖 Lolicon 第三方聚合。
命令经各子模块 import 时装饰器注册；本模块仅做汇总。
"""

from . import illust as _illust  # noqa: F401 注册 /pixiv
from . import novel as _novel    # noqa: F401 注册 /novel
from . import setu as _setu      # noqa: F401 注册 /setu

# 仅命令执行，不注册任何 LLM 工具
TOOLS = []
