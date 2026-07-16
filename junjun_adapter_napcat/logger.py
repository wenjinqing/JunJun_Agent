"""Adapter 专用日志（轻量包装）。"""

import logging
import sys

_logger = logging.getLogger("junjun_adapter")
if not _logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(h)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False  # 防止根 logger 重复输出一遍

class _Logger:
    def info(self, m): _logger.info(m)
    def warning(self, m): _logger.warning(m)
    def error(self, m, **k): _logger.error(m)
    def debug(self, m): _logger.debug(m)

logger = _Logger()
