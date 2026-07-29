"""NapCat 看门狗（君君专用）：启动 NapCat + 掉线/崩溃自动重启，一体化。

功能：
- 启动时 QQ.exe 不在 -> 直接拉起 NapCat（快速登录，免扫码）
- 运行中每 30s 检测：OneBot get_status online=false（被踢下线）
  或 QQ.exe 进程消失（崩溃）-> 自动重启
- 免扫码原理：launcher-user.bat <qq> 走 NapCat 快速登录（本地缓存
  凭证），只要不是扫码首次登录，重启都自动上线；只有腾讯风控
  强制失效时才需要人工扫码（1 小时限 5 次重启防疯狂循环）

检测原理：
- 读 NapCat config/onebot11_10000001.json 里已启用的 HTTP server，
  调 OneBot get_status：data.online == false 即被踢下线

用法：
    双击 scripts/napcat_watchdog.py     # 前台窗口（输出全写日志文件）
    pythonw scripts/napcat_watchdog.py  # 无窗口后台跑

所有输出写入 data/napcat_watchdog.log（控制台不再打印）。
单实例：已有一个看门狗在跑时，新实例直接退出（PID 文件锁）。

注意：taskkill 会杀掉本机所有 QQ.exe（本机 QQ 专供 bot 使用）。
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- 配置
NAPCAT_DIR = Path(r"E:\MaiM\NapCat.Shell")
QQ = "10000001"               # 君君
LAUNCH_BAT = "启动君君.bat"      # 内部调 launcher-user.bat <qq>（快速登录）
CHECK_INTERVAL = 30        # 检测间隔（秒）
FAIL_THRESHOLD = 2         # 连续失败几次才重启
RESTART_COOLDOWN = 120     # 重启后冷却（秒），等登录完成
MAX_RESTARTS_PER_HOUR = 5  # 超限休眠，防风控下疯狂重启
OVERLOAD_SLEEP = 1800      # 超限后休眠（秒）

LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "napcat_watchdog.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],  # 输出全进日志
)
log = logging.getLogger("napcat_watchdog")

_PID_FILE = LOG_FILE.parent / "napcat_watchdog.pid"


def _pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/fi", f"PID eq {pid}", "/nh"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def _ensure_single_instance() -> None:
    """PID 文件锁：已有看门狗在跑则退出，防多实例重复 taskkill/重启。"""
    if _PID_FILE.exists():
        try:
            old = int(_PID_FILE.read_text().strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            log.warning(f"已有看门狗在运行 (PID {old})，本实例退出")
            sys.exit(0)
    _PID_FILE.write_text(str(os.getpid()))


# ---------------------------------------------------------------- 健康检查

def _http_endpoint():
    """从 onebot11_<qq>.json 找第一个启用的 HTTP server，返回 (url, token)。"""
    cfg_path = NAPCAT_DIR / "config" / f"onebot11_{QQ}.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"配置解析失败: {e}")
        return None
    for srv in (cfg.get("network", {}).get("httpServers") or []):
        if srv.get("enable"):
            host = srv.get("host", "127.0.0.1")
            return f"http://{host}:{srv['port']}/get_status", srv.get("token", "")
    return None


def check_online() -> bool:
    """True=在线 / False=掉线或 HTTP 不可达。"""
    ep = _http_endpoint()
    if ep is None:
        log.warning("onebot11 配置里没有启用的 HTTP server，无法检测在线状态")
        return False
    url, token = ep
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or {}
        return bool(data.get("online")) and bool(data.get("good", True))
    except Exception as e:
        log.warning(f"get_status 失败: {type(e).__name__}: {e}")
        return False


def qq_process_alive() -> bool:
    """QQ.exe 进程是否存在。"""
    out = subprocess.run(
        ["tasklist", "/fi", "imagename eq QQ.exe", "/nh"],
        capture_output=True, text=True).stdout
    return "QQ.exe" in out


# ---------------------------------------------------------------- 启动 / 重启

def launch() -> None:
    """拉起 NapCat（快速登录，免扫码）。"""
    bat = NAPCAT_DIR / LAUNCH_BAT
    if not bat.exists():
        log.error(f"启动脚本不存在: {bat}")
        return
    subprocess.Popen(["cmd", "/c", "start", "/min", "", str(bat)],
                     cwd=str(NAPCAT_DIR))
    log.info(f"已拉起 {LAUNCH_BAT}（快速登录 QQ {QQ}）")


def restart() -> None:
    log.warning("开始重启 NapCat：taskkill QQ.exe ...")
    subprocess.run(["taskkill", "/f", "/im", "QQ.exe"],
                   capture_output=True)  # 进程不存在也报错，忽略
    time.sleep(5)
    launch()


# ---------------------------------------------------------------- 主循环

def main() -> None:
    _ensure_single_instance()
    log.info(f"看门狗启动：监控君君（QQ {QQ}），每 {CHECK_INTERVAL}s 检测一次")
    # 一体化：启动时 NapCat 没跑就先拉起
    if not qq_process_alive():
        log.info("QQ.exe 未运行，启动 NapCat...")
        launch()
        time.sleep(RESTART_COOLDOWN)

    fail_count = 0
    restart_times: list = []
    cooldown_until = time.time()  # 首次立即检测

    while True:
        now = time.time()
        if now < cooldown_until:
            time.sleep(CHECK_INTERVAL)
            continue

        if not qq_process_alive():
            reason = "QQ.exe 进程不存在（崩溃）"
            failed = True
        elif not check_online():
            reason = "君君掉线（被踢/登录态失效）"
            failed = True
        else:
            reason = ""
            failed = False

        if failed:
            fail_count += 1
            log.warning(f"检测失败 ({fail_count}/{FAIL_THRESHOLD}): {reason}")
        else:
            if fail_count:
                log.info("恢复正常")
            fail_count = 0

        if fail_count >= FAIL_THRESHOLD:
            restart_times = [t for t in restart_times if now - t < 3600]
            if len(restart_times) >= MAX_RESTARTS_PER_HOUR:
                log.error(f"1 小时内已重启 {MAX_RESTARTS_PER_HOUR} 次，"
                          f"休眠 {OVERLOAD_SLEEP // 60} 分钟（可能要扫码/风控）")
                time.sleep(OVERLOAD_SLEEP)
                restart_times = []
                fail_count = 0
                continue
            restart()
            restart_times.append(time.time())
            fail_count = 0
            cooldown_until = time.time() + RESTART_COOLDOWN

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("看门狗已停止")
