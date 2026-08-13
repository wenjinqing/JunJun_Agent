"""结构化日志（structlog + rich），对齐原 common/logger.py 接口。

2026-08-13 审查 P1：此前只打 stdout——bot/adapter 崩溃即丢事故现场
（WebUI 日志页是内存环形缓冲，进程死即清空）。现在 initialize_logging
默认把渲染好的日志行双写进 logs/<log_name>.log 轮转文件（10MB×5）：
- 双写用自定义流而不是改走 stdlib 管线——PrintLoggerFactory 现状下大量
  测试靠 capsys 接输出，tee 内动态取 sys.stdout 保持兼容，零行为变化
- 文件侧剥 ANSI 颜色码（ConsoleRenderer 的颜色码落盘是乱码）
- 写盘任何异常静默——日志出问题绝不拖死业务
- 多进程各写各的文件（bot/adapter 不同名），RotatingFileHandler 跨进程
  在 Windows 上 rename 必炸，所以没用 stdlib handler 方案
"""

import logging
import re
import sys
from pathlib import Path

import structlog

_initialized = False

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 5
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _TeeStream:
    """structlog PrintLoggerFactory 的目标流：动态写当前 sys.stdout
    （pytest capsys 兼容）+ 同步写轮转日志文件（剥 ANSI）。"""

    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8", errors="replace")

    def write(self, s: str) -> int:
        try:
            sys.stdout.write(s)
        except Exception:
            pass
        try:
            self._f.write(_ANSI.sub("", s))
            self._f.flush()
            if self._f.tell() > _MAX_BYTES:
                self._rotate()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        for st in (sys.stdout, self._f):
            try:
                st.flush()
            except Exception:
                pass

    def _rotate(self) -> None:
        self._f.close()
        for i in range(_BACKUPS - 1, 0, -1):
            src = self._path.with_name(f"{self._path.name}.{i}")
            dst = self._path.with_name(f"{self._path.name}.{i + 1}")
            if src.exists():
                dst.unlink(missing_ok=True)
                src.rename(dst)
        self._path.rename(self._path.with_name(f"{self._path.name}.1"))
        self._f = self._path.open("a", encoding="utf-8", errors="replace")


def initialize_logging(level: str = "INFO", log_name: str | None = "bot") -> None:
    """初始化日志。log_name：双写目标文件名（logs/<log_name>.log）；
    None = 只打控制台（测试/临时脚本）。"""
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

    factory = None
    if log_name:
        try:
            factory = structlog.PrintLoggerFactory(_TeeStream(_LOG_DIR / f"{log_name}.log"))
        except Exception:
            factory = None   # 落盘失败不挡启动

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
        logger_factory=factory or structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _initialized = True


def get_logger(name: str = "junjun") -> "structlog.stdlib.BoundLogger":
    if not _initialized:
        initialize_logging()
    return structlog.get_logger(name).bind()  # type: ignore
