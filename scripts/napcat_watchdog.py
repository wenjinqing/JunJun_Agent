"""NapCat 看门狗：检测 QQ 被强制下线 / NapCat 崩溃，自动重启重连。

检测原理：
- 读 NapCat config/onebot11_<qq>.json 里已启用的 HTTP server，
  调 OneBot get_status：data.online == false 即被踢下线
- HTTP 无响应 + QQ.exe 进程不在 = NapCat 崩溃
- 无 HTTP server 的账号只参与「陪跑重启」（QQ.exe 是多账号共享进程，
  踢下线重启必须 taskkill 整个 QQ.exe，会把所有账号一起带起来）

重启策略：
- 连续 2 次检测失败才重启（防抖动）
- taskkill /f /im QQ.exe -> 等 5s -> 逐个拉起账号启动 bat
- 重启后冷却 120s（登录需要时间），1 小时内最多重启 5 次，超限休眠 30 分钟

用法：
    python scripts/napcat_watchdog.py            # 前台跑
    start /min pythonw scripts/napcat_watchdog.py  # 后台静默跑

注意：taskkill 会杀掉本机所有 QQ.exe（本机 QQ 专供 bot 使用，无个人 QQ 冲突）。
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
ACCOUNTS = [
    {"qq": "2477702109", "bat": "启动君君.bat"},
    {"qq": "1033245881", "bat": "启动伊伊.bat"},
]
CHECK_INTERVAL = 30        # 检测间隔（秒）
FAIL_THRESHOLD = 2         # 连续失败几次才重启
RESTART_COOLDOWN = 120     # 重启后冷却（秒），等登录完成
MAX_RESTARTS_PER_HOUR = 5  # 超限休眠，防疯狂重启循环
OVERLOAD_SLEEP = 1800      # 超限后休眠（秒）

LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "napcat_watchdog.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("napcat_watchdog")


# ---------------------------------------------------------------- 健康检查

def _http_endpoint(qq: str):
    """从 onebot11_<qq>.json 找第一个启用的 HTTP server，返回 (url, token)。"""
    cfg_path = NAPCAT_DIR / "config" / f"onebot11_{qq}.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"{qq} 配置解析失败: {e}")
        return None
    for srv in (cfg.get("network", {}).get("httpServers") or []):
        if srv.get("enable"):
            host = srv.get("host", "127.0.0.1")
            return f"http://{host}:{srv['port']}/get_status", srv.get("token", "")
    return None


def check_online(qq: str):
    """True=在线 / False=掉线或异常 / None=无法检测（无 HTTP server）。"""
    ep = _http_endpoint(qq)
    if ep is None:
        return None
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
        log.warning(f"{qq} get_status 失败: {type(e).__name__}: {e}")
        return False


def qq_process_alive() -> bool:
    """QQ.exe 进程是否存在。"""
    out = subprocess.run(
        ["tasklist", "/fi", "imagename eq QQ.exe", "/nh"],
        capture_output=True, text=True).stdout
    return "QQ.exe" in out


# ---------------------------------------------------------------- 重启

def restart_all() -> None:
    log.warning("开始重启 NapCat：taskkill QQ.exe ...")
    subprocess.run(["taskkill", "/f", "/im", "QQ.exe"],
                   capture_output=True)  # 进程不存在也返回错误，忽略
    time.sleep(5)
    for acc in ACCOUNTS:
        bat = NAPCAT_DIR / acc["bat"]
        if not bat.exists():
            log.error(f"启动脚本不存在: {bat}")
            continue
        # start /min 独立窗口拉起，bat 内 pause 不影响
        subprocess.Popen(
            ["cmd", "/c", "start", "/min", "", str(bat)],
            cwd=str(NAPCAT_DIR))
        log.info(f"已拉起 {acc['bat']} (QQ {acc['qq']})")
        time.sleep(3)  # 错开启动


# ---------------------------------------------------------------- 主循环

def main() -> None:
    log.info(f"看门狗启动：监控 {[a['qq'] for a in ACCOUNTS]}，"
             f"每 {CHECK_INTERVAL}s 检测一次")
    fail_count = 0
    restart_times: list = []
    cooldown_until = 0.0

    while True:
        now = time.time()
        if now < cooldown_until:
            time.sleep(CHECK_INTERVAL)
            continue

        # 进程不在 = 崩了；进程在 + get_status 掉线 = 被踢
        alive = qq_process_alive()
        statuses = {a["qq"]: check_online(a["qq"]) for a in ACCOUNTS}
        offline = [qq for qq, ok in statuses.items() if ok is False]
        undetectable = [qq for qq, ok in statuses.items() if ok is None]

        if not alive:
            reason = "QQ.exe 进程不存在（崩溃）"
            failed = True
        elif offline:
            reason = f"账号掉线: {offline}"
            failed = True
        else:
            failed = False

        if failed:
            fail_count += 1
            log.warning(f"检测失败 ({fail_count}/{FAIL_THRESHOLD}): {reason}")
        else:
            if fail_count:
                log.info("恢复正常")
            fail_count = 0

        if fail_count >= FAIL_THRESHOLD:
            # 1 小时内重启次数限制
            restart_times = [t for t in restart_times if now - t < 3600]
            if len(restart_times) >= MAX_RESTARTS_PER_HOUR:
                log.error(f"1 小时内已重启 {MAX_RESTARTS_PER_HOUR} 次，"
                          f"休眠 {OVERLOAD_SLEEP // 60} 分钟（可能要扫码/风控）")
                time.sleep(OVERLOAD_SLEEP)
                restart_times = []
                fail_count = 0
                continue
            restart_all()
            restart_times.append(time.time())
            fail_count = 0
            cooldown_until = time.time() + RESTART_COOLDOWN

        if undetectable:
            log.debug(f"无 HTTP server 无法主动检测（跟随重启）: {undetectable}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("看门狗已停止")
