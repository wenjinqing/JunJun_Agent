"""NapCat OneBot HTTP API 客户端（插件迁移基础设施）。

用途：插件需要「主动调用」OneBot 能力（查群成员列表、上传文件等），
这类调用不是消息回复，走 NapCat 的 HTTP API 直连。
地址/token 从 .env 读（NAPCAT_HTTP_BASE / NAPCAT_HTTP_TOKEN），未配置时所有调用
返回 None 降级——插件应给出「功能不可用」的友好回复而不是炸掉。
（注意与 NAPCAT_TOKEN 区分：后者是 NapCat→Adapter WS 连入鉴权，别混用。）

NapCat HTTP 服务端默认端口 3000（在 NapCat 配置里开启 httpServers）。
"""

import os
from typing import Optional

from junjun_core.observability import get_logger

logger = get_logger("napcat.client")

_TIMEOUT = 15.0


def _base() -> str:
    return os.environ.get("NAPCAT_HTTP_BASE", "").strip().rstrip("/")


def available() -> bool:
    return bool(_base())


async def call(action: str, params: Optional[dict] = None,
               timeout: float = _TIMEOUT) -> Optional[dict]:
    """调用 OneBot HTTP API。成功返回 data 字段；不可用/失败返回 None。"""
    base = _base()
    if not base:
        logger.debug(f"NAPCAT_HTTP_BASE 未配置，跳过 {action}")
        return None
    try:
        import httpx
        headers = {}
        token = os.environ.get("NAPCAT_HTTP_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # trust_env=False：本地 127.0.0.1 的 HTTP 请求不该走系统代理
        # （实测代理拦截返回 502 -> resp.json() 空 JSONDecodeError）
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(f"{base}/{action}", json=params or {}, headers=headers)
            data = resp.json()
        if data.get("status") == "ok" or data.get("retcode") == 0:
            return data.get("data")
        logger.warning(f"OneBot {action} 返回异常: retcode={data.get('retcode')} msg={data.get('msg')}")
        return None
    except Exception as e:
        logger.warning(f"OneBot {action} 调用失败: {type(e).__name__}: {e}")
        return None


async def get_group_members(group_id: str) -> Optional[list]:
    """群成员列表 [{user_id, nickname, card, ...}]；失败 None。"""
    return await call("get_group_member_list", {"group_id": int(group_id)})


async def get_group_member_info(group_id: str, user_id: str) -> Optional[dict]:
    return await call("get_group_member_info", {
        "group_id": int(group_id), "user_id": int(user_id), "no_cache": True})


async def upload_group_file(group_id: str, file_path: str, name: str = "") -> bool:
    """上传群文件（本地路径或 http URL）。成功 True。"""
    params = {"group_id": int(group_id), "file": file_path, "name": name or file_path.rsplit("/", 1)[-1]}
    data = await call("upload_group_file", params, timeout=120.0)
    return data is not None


async def upload_private_file(user_id: str, file_path: str, name: str = "") -> bool:
    params = {"user_id": int(user_id), "file": file_path, "name": name or file_path.rsplit("/", 1)[-1]}
    data = await call("upload_private_file", params, timeout=120.0)
    return data is not None


def qq_avatar_url(user_id: str, size: int = 640) -> str:
    """QQ 头像直链（无需 API）。"""
    return f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s={size}"
