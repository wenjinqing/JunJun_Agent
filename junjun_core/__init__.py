"""君君核心包：网关、配置、可观测、数据库。"""

from .config import get_global_config
from .observability import get_logger, initialize_logging, lf
from .gateway import get_gateway, get_router
from .contracts import ReplySet, ReplySegment

__all__ = ["get_global_config", "get_logger", "initialize_logging", "lf", "get_gateway", "get_router", "ReplySet", "ReplySegment"]
