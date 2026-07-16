"""结构化日志（structlog + rich），对齐原 common/logger.py 接口。"""

import logging
import sys
from typing import Optional

import structlog

_initialized = False


def initialize_logging(level: str = "INFO") -> None:
    global _initialized
    if _initialized:
        return
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore
        except Exception:
            pass

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _initialized = True


def get_logger(name: str = "junjun") -> "structlog.stdlib.BoundLogger":
    if not _initialized:
        initialize_logging()
    return structlog.get_logger(name).bind()  # type: ignore
