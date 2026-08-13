"""httpx 客户端统一工厂（2026-08-13 审查 P2）。

默认 trust_env=False：trust_env=True 会读环境变量代理乃至 Windows 注册表
系统代理——localhost/内网请求也被送进代理，本机代理不回流 localhost 就是
502/ECONNRESET（2026-08-13 Langfuse 脚本实锤，napcat_client 早年同款注释）。
明确要穿代理的调用方（GFW 封锁站点的抓取）显式传 trust_env=True。

命名纪律：本模块曾名 http.py——junjun_core 目录进 PYTHONPATH 时（PyCharm
source root/显式设定）会遮蔽标准库 http 包，urllib.request 内部 import
http.client 直接循环导入炸死进程（napcat_watchdog 起不来，当日实锤）。
junjun_core 下模块名永远避开标准库同名。
"""

import httpx


def make_client(*, trust_env: bool = False, **kw) -> httpx.Client:
    return httpx.Client(trust_env=trust_env, **kw)


def make_async_client(*, trust_env: bool = False, **kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(trust_env=trust_env, **kw)
