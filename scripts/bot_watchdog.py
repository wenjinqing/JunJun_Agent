# -*- coding: utf-8 -*-
"""bot/adapter 看门狗（2026-08-13 审查 P1，抄 napcat_watchdog 骨架）。

此前三件套里 NapCat 有看门狗、bot/adapter 裸奔——进程崩了没人知道，
用户侧表现「君君突然不理人」，要等人工发现手动重启。本看门狗：

- 拉起 adapter + bot 两个子进程并常驻看护（顺序：adapter 先、bot 后）
- 进程退出 → 带退避自动重启（5/15/30/60/120s 阶梯）
- bot 启动宽限期后探测网关端口，连续探测失败按死亡处理（事件循环卡死
  但进程活着的半死态）
- 崩溃循环熔断：一小时内重启超 12 次仍不稳定 -> 放弃并大声记日志
  （无限重启会把 QQ 重连频率刷到风控线上，宁停勿刷）
- PID 文件单实例：重复启动自动接管旧实例（同 napcat_watchdog）

用法：uv run python scripts/bot_watchdog.py
（NapCat 仍由 napcat_watchdog.py 看护；本脚本只管 adapter + bot。）
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "bot_watchdog.log", encoding="utf-8")],
)
log = logging.getLogger("bot_watchdog")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
_PID_FILE = DATA_DIR / "bot_watchdog.pid"

PY = sys.executable  # uv run 进来就是 venv 解释器
CHECK_INTERVAL = 5.0       # 巡检间隔（秒）
BOOT_GRACE = 90.0          # 启动宽限：网关端口监听需要时间
PROBE_FAIL_LIMIT = 3       # 端口连续探测失败几次按死亡处理
BACKOFF = [5, 15, 30, 60, 120]     # 重启退避阶梯（秒）
MAX_RESTARTS_PER_HOUR = 12         # 崩溃循环熔断线


# ---------------------------------------------------------------- 单实例接管

def _pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/fi", f"PID eq {pid}", "/nh"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def _replace_old_instance() -> None:
    """PID 文件接管：已有看门狗在跑则杀掉旧实例，由本实例接管。"""
    if _PID_FILE.exists():
        try:
            old = int(_PID_FILE.read_text().strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            subprocess.run(["taskkill", "/f", "/pid", str(old)],
                           capture_output=True)
            time.sleep(1)
            log.info(f"旧看门狗 (PID {old}) 已结束，本实例接管")
    _PID_FILE.write_text(str(os.getpid()))


# ---------------------------------------------------------------- 看护逻辑

def _gateway_port() -> int:
    """bot 网关端口（config/bot_config.toml [gateway] port，缺省 8192）。"""
    try:
        import tomllib
        cfg = tomllib.loads((ROOT / "config" / "bot_config.toml")
                            .read_text(encoding="utf-8"))
        return int(cfg.get("gateway", {}).get("port", 8192))
    except Exception:
        return 8192


def _probe_port(port: int) -> bool:
    """网关端口是否接受连接（bot 半死态探针）。"""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


@dataclass
class ProcState:
    name: str
    cmd: list
    port: int = 0                       # 0 = 不做端口探测（adapter 无监听口）
    proc: object = None
    started_at: float = 0.0
    probe_fails: int = 0
    restarts: list = field(default_factory=list)   # 近一小时重启时间戳
    next_retry: float = 0.0
    gave_up: bool = False


def supervise_once(st: ProcState, now: float, *, spawn, probe) -> str:
    """巡检一轮：返回 spawn / kill+spawn / wait / give_up。

    spawn/probe 依赖注入：测试喂假进程假探针，不拉真进程。
    """
    if st.gave_up:
        return "give_up"
    alive = st.proc is not None and st.proc.poll() is None
    if alive:
        healthy = True
        if st.port and now - st.started_at > BOOT_GRACE:
            healthy = probe(st.port)
        if healthy:
            st.probe_fails = 0
            return "wait"
        st.probe_fails += 1
        if st.probe_fails < PROBE_FAIL_LIMIT:
            log.warning(f"{st.name} 端口 {st.port} 探测失败 "
                        f"（{st.probe_fails}/{PROBE_FAIL_LIMIT}）")
            return "wait"
        log.warning(f"{st.name} 端口连续探测失败，按死亡处理（半死态）")
        try:
            st.proc.kill()
        except Exception:
            pass
        alive = False
    if alive or now < st.next_retry:
        return "wait"
    st.restarts = [t for t in st.restarts if now - t < 3600]
    if len(st.restarts) >= MAX_RESTARTS_PER_HOUR:
        st.gave_up = True
        log.error(f"{st.name} 一小时重启 {MAX_RESTARTS_PER_HOUR} 次仍不稳定，"
                  "看门狗放弃（防崩溃循环刷爆 QQ 风控），请人工排查后重启本脚本")
        return "give_up"
    st.proc = spawn(st.cmd)
    st.started_at = now
    st.restarts.append(now)
    idx = min(len(st.restarts) - 1, len(BACKOFF) - 1)
    st.next_retry = now + BACKOFF[idx]
    st.probe_fails = 0
    log.info(f"{st.name} 已拉起 (PID {getattr(st.proc, 'pid', '?')})，"
             f"近一小时第 {len(st.restarts)} 次")
    return "spawn"


def _spawn(cmd: list):
    return subprocess.Popen(cmd, cwd=str(ROOT))


def main() -> None:
    _replace_old_instance()
    port = _gateway_port()
    states = [
        ProcState("adapter", [PY, str(ROOT / "scripts" / "run_adapter.py")]),
        ProcState("bot", [PY, str(ROOT / "scripts" / "run_junjun.py")], port=port),
    ]
    log.info(f"看门狗启动：adapter + bot（网关端口 {port}）")
    try:
        while True:
            now = time.time()
            for st in states:
                try:
                    supervise_once(st, now, spawn=_spawn, probe=_probe_port)
                except Exception as e:
                    log.warning(f"巡检 {st.name} 异常（下轮继续）: {type(e).__name__}: {e}")
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        log.info("收到退出信号，关停子进程...")
    finally:
        for st in states:
            if st.proc is not None and st.proc.poll() is None:
                try:
                    st.proc.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
