"""Adapter 专用日志（轻量包装）。"""

import logging
import sys
from pathlib import Path

_logger = logging.getLogger("junjun_adapter")
if not _logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(h)
    # 落盘轮转（2026-08-13 审查 P1：adapter 崩溃即丢事故现场）：与 core 的
    # structlog tee 不同文件（adapter.log），单进程单文件无跨进程 rename 冲突
    try:
        from logging.handlers import RotatingFileHandler
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "adapter.log",
            maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        _logger.addHandler(fh)
    except Exception:
        pass  # 落盘失败不挡启动
    _logger.setLevel(logging.INFO)
    _logger.propagate = False  # 防止根 logger 重复输出一遍

class _Logger:
    def info(self, m): _logger.info(m)
    def warning(self, m): _logger.warning(m)
    def error(self, m, **k): _logger.error(m)
    def debug(self, m): _logger.debug(m)

logger = _Logger()
