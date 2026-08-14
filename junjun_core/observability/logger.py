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

2026-08-14 修两处（生产事故实锤）：
- get_logger 懒初始化曾以默认 log_name="bot" 直接完成全量初始化并置
  _initialized——import 链上任何一个模块级 get_logger 都会抢在入口
  （run_adapter 的 initialize_logging("adapter")）之前占坑，入口调用静默
  no-op，adapter 全天日志全写进 bot.log、adapter.log 停更。现在懒兜底只配
  控制台（_bootstrap_console_only），落盘只能由进程入口显式初始化决定；
  附带收益：测试/脚本子进程 import 了 core 也不再追加污染生产 bot.log。
- _rotate 旧实现先 close 再 rename，rename 失败（别的进程/阅读器持有句柄）
  文件句柄永久关闭、write 的 except 静默吞掉——进程日志无声停更。现在
  失败保底重开原文件继续追加，本轮轮转放弃、下次写再试。
"""

import logging
import re
import sys
from pathlib import Path

import structlog

_initialized = False
_bootstrapped = False   # 仅控制台兜底已配（不碰日志文件，落盘留给入口显式初始化）

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
        try:
            self._f.close()
            for i in range(_BACKUPS - 1, 0, -1):
                src = self._path.with_name(f"{self._path.name}.{i}")
                dst = self._path.with_name(f"{self._path.name}.{i + 1}")
                if src.exists():
                    dst.unlink(missing_ok=True)
                    src.rename(dst)
            self._path.rename(self._path.with_name(f"{self._path.name}.1"))
        except Exception:
            pass   # rename 被占用/盘故障：放弃本轮轮转，下次写再试
        finally:
            # 保底重开——旧实现 rename 一炸句柄永久关闭，进程日志静默停更
            if self._f.closed:
                try:
                    self._f = self._path.open("a", encoding="utf-8", errors="replace")
                except Exception:
                    pass


def _configure(level: str, factory) -> None:
    """structlog/stdlib 一次性装配（控制台编码 + basicConfig + processors）。"""
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
        logger_factory=factory,
        cache_logger_on_first_use=True,
    )


def initialize_logging(level: str = "INFO", log_name: str | None = "bot") -> None:
    """初始化日志。log_name：双写目标文件名（logs/<log_name>.log）；
    None = 只打控制台（测试/临时脚本）。"""
    global _initialized
    if _initialized:
        return
    factory = None
    if log_name:
        try:
            factory = structlog.PrintLoggerFactory(_TeeStream(_LOG_DIR / f"{log_name}.log"))
        except Exception:
            factory = None   # 落盘失败不挡启动

    _configure(level, factory or structlog.PrintLoggerFactory())
    _initialized = True


def _bootstrap_console_only() -> None:
    """import 期 get_logger 的兜底：只配控制台，绝不占 logs/*.log、不置
    _initialized——入口（run_junjun/run_adapter）随后的显式 initialize_logging
    才能抢到落盘配置。注意 cache_logger_on_first_use：兜底窗口内就打过日志的
    代理会定格在控制台工厂，import 期噪音容忍这一点。"""
    global _bootstrapped
    if _initialized or _bootstrapped:
        return
    _configure("INFO", structlog.PrintLoggerFactory())
    _bootstrapped = True


def get_logger(name: str = "junjun") -> "structlog.stdlib.BoundLogger":
    if not _initialized and not _bootstrapped:
        _bootstrap_console_only()
    return structlog.get_logger(name).bind()  # type: ignore
