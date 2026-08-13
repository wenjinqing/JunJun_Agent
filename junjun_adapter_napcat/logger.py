"""Adapter 日志：统一走 core 的 structlog 管线（2026-08-13 审查 P2）。

此前本模块自带 stdlib logger + RotatingFileHandler 写 logs/adapter.log，
而 run_adapter.py 的 initialize_logging(log_name="adapter") 又让 core tee
写同一个文件——同进程双写者、两套轮转互踩（tee rename 时 stdlib handler
还握着句柄，Windows 上 rename 炸但被 except 吞掉，表现为轮转静默失效）。
旧 _Logger 门面的 error(m, **k) 还静默吞 exc_info。

现在只留接口皮：initialize_logging 由进程入口负责（run_adapter.py 已做），
模块侧只取 logger。测试进程里没人初始化时 get_logger 自举默认 bot.log，无害。
"""

from junjun_core.observability import get_logger

logger = get_logger("junjun_adapter")
