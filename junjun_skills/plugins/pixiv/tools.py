"""pixiv 插件：P 站内容统一获取（官方 AJAX API）。

子模块：
- client.py      共享 HTTP/cookie/反爬层
- setu.py        /setu 随机图（官方搜索/排行，修复裸词静默丢弃 bug）
- illust.py      /pixiv 命令族（search/illust/rank/author/new/dl）
- novel.py       /novel 小说下载（从 pixiv_novel 迁入，逻辑不变）
- agent_tools.py LLM 工具面（Agent 主动搜索/推荐/下载）

全部走官方接口（2026-08-03 起），不再依赖 Lolicon 第三方聚合。
命令经各子模块 import 时装饰器注册；本模块仅做汇总。

双通道策略（2026-08-03 用户定）：
- 命令通道（0 token）：/setu /pixiv /novel，语义不变
- 工具通道（Agent 自主调用）：搜索推荐群私通用；下载发送仅私聊
  （群聊调下载类工具会收到解释性拒绝，模型据此改给链接）
"""

from . import agent_tools as _agent_tools  # noqa: F401
from . import illust as _illust  # noqa: F401 注册 /pixiv
from . import novel as _novel    # noqa: F401 注册 /novel
from . import setu as _setu      # noqa: F401 注册 /setu

# LLM 工具：Agent 可主动调用（搜索/推荐/私聊下载），与命令并存
TOOLS = list(_agent_tools.TOOLS)
