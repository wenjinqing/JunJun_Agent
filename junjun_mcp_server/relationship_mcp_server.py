"""relationship MCP server：对齐原 mcp_server/relationship_mcp_server.py 工具面。

存储直接复用阶段 4 的 PersonInfo/memory_points（MCP 只是访问通道，不另建存储）。
stdio 传输，由君君自身作为 MCP 客户端调用（验证 MCP 闭环）。

独立进程运行：python -m junjun_mcp_server.relationship_mcp_server
"""

import sys
from pathlib import Path

# 独立进程入口：确保仓库根在 path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# stdio 传输下 stdout 必须是纯 JSON-RPC。junjun_core 的 import 链
# （gateway -> maim_message）会向 stdout 打日志污染协议流（实测客户端
# BrokenResourceError）。对策：启动时在 stdout 重定向到 stderr 的上下文里
# 预 import 全部业务依赖，之后才把 stdout 还给 MCP 传输层。
import contextlib
import logging
import structlog

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))

with contextlib.redirect_stdout(sys.stderr):
    import junjun_core  # noqa: F401
    from junjun_core.database import init_database
    from junjun_memory.user_profile import get_profile_store
    init_database()
    # 劫持所有已注册 stdout logging handler 到 stderr（maim_message 等）
    for h in logging.root.handlers:
        if getattr(h, "stream", None) is sys.stdout:
            h.stream = sys.stderr

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("junjun-relationship")

_PENALTIES = {
    "insult": -10.0, "harassment": -8.0, "spam": -5.0, "unfriendly": -3.0,
    "inappropriate": -5.0, "aggressive": -4.0, "disrespect": -3.0,
    "negative_attitude": -2.0, "ignore_response": -1.0, "cold_response": -0.5,
}
_SEVERITY = {"minor": 0.5, "moderate": 1.0, "severe": 1.5, "extreme": 2.0}


def _store():
    return get_profile_store()


@mcp.tool()
def apply_relationship_penalty(user_id: str, platform: str, penalty_type: str,
                               severity: str = "moderate", reason: str = "") -> str:
    """应用亲密度惩罚。用户辱骂/骚扰/刷屏/不友善时使用。

    penalty_type: insult/harassment/spam/unfriendly/inappropriate/aggressive/
    disrespect/negative_attitude/ignore_response/cold_response
    severity: minor/moderate/severe/extreme
    """
    base = _PENALTIES.get(penalty_type)
    if base is None:
        return f"未知惩罚类型: {penalty_type}"
    value = base * _SEVERITY.get(severity, 1.0)
    _store().add_point(platform, user_id, "关系事件",
                       f"惩罚[{penalty_type}/{severity}] {reason}"[:80],
                       weight=min(1.0, abs(value) / 10))
    return f"已记录惩罚: {penalty_type} x{_SEVERITY.get(severity, 1.0)} = {value}"


@mcp.tool()
def update_user_impression(user_id: str, platform: str, impression: str,
                           confidence: float = 0.8) -> str:
    """更新对用户的总体印象。观察到用户新的性格特点/行为模式时使用。"""
    _store().add_point(platform, user_id, "印象", impression[:100],
                       weight=max(0.1, min(1.0, confidence)))
    return f"印象已更新: {impression[:50]}"


@mcp.tool()
def add_user_tag(user_id: str, platform: str, tag: str, category: str = "特征") -> str:
    """给用户打标签（兴趣/身份/性格特征等）。"""
    _store().add_point(platform, user_id, category[:20], tag[:50], weight=0.7)
    return f"标签已加: [{category}] {tag}"


@mcp.tool()
def get_user_profile(user_id: str, platform: str) -> str:
    """查询用户画像（印象/标签/关系事件）。"""
    points = _store().get_points(platform, user_id, top_k=15)
    if not points:
        return "该用户暂无画像记录。"
    return "\n".join(f"[{p['category']}] {p['content']} (w={p['weight']:.2f})" for p in points)


@mcp.tool()
def set_user_name(user_id: str, platform: str, name: str) -> str:
    """记住用户的称呼/名字。"""
    _store().set_name(platform, user_id, name[:30])
    return f"已记住称呼: {name}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
