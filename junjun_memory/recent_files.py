"""每会话最近收到的文件引用（chat_id -> [{name,size,url,ts}]）。

「文件入口」闭环的会话归属层：对方发文件 -> 适配器解出 file_ref -> 网关
InboundMeta.file_refs -> processor 登记到这里 -> workspace_save_file 工具凭
「最近收到的文件」下载进工作区。对齐 vision.recent_image_urls 的语义
（TTL 10 分钟、内存态、重启即丢——QQ 链接本身有时效，不需要持久化）。
"""

import time
from collections import deque
from typing import Dict, List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.recent_files")

_RECENT: Dict[str, deque] = {}
_RECENT_TTL = 600.0   # QQ 文件链接有时效，10 分钟外的「最近的文件」多半已过期
_RECENT_MAX = 5       # 每会话只记最近几个，防刷屏灌爆


def note_recent_file(chat_id: str, ref: dict) -> None:
    """登记一条入站文件引用（{name,size,url}）。坏数据静默跳过。"""
    if not chat_id or not isinstance(ref, dict) or not ref.get("url"):
        return
    try:
        dq = _RECENT.setdefault(chat_id, deque(maxlen=_RECENT_MAX * 2))
        dq.append((time.time(), {"name": str(ref.get("name") or "未命名文件"),
                                 "size": int(ref.get("size") or 0),
                                 "url": str(ref["url"])}))
    except Exception as e:
        logger.debug(f"文件登记失败（忽略）: {e}")


def recent_files(chat_id: str, ttl: float = _RECENT_TTL) -> List[dict]:
    """该会话最近 ttl 秒内收到过的文件 [{name,size,url}]，新在前，上限 _RECENT_MAX。"""
    now = time.time()
    out = [ref for ts, ref in _RECENT.get(chat_id, ()) if now - ts <= ttl]
    out.reverse()
    return out[:_RECENT_MAX]


def recent_file(chat_id: str, ttl: float = _RECENT_TTL) -> Optional[dict]:
    """最近一个文件（没有返回 None）——workspace_save_file 的默认目标。"""
    files = recent_files(chat_id, ttl)
    return files[0] if files else None


def _reset_for_test() -> None:
    _RECENT.clear()
