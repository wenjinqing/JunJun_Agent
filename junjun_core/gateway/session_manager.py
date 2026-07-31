"""会话管理：按 platform:chat_id 维护会话状态。

分层约束：junjun_core 不 import 上层包。memory/agent 为通用槽位，
由 junjun_agent 层在首次处理时注入（processor 模式，见 gateway/router.py）。

会话淘汰（P1-8）：会话/Agent（各自持有 httpx 连接池）原只增不减，
长跑内存必涨。空闲超 TTL（默认 3 天）或总数超上限的会话被淘汰，
淘汰时经 on_evict 回调释放上层资源（agent 连接池、会话队列条目）。
"""

import asyncio
import time
from typing import Callable, Dict, Optional

_SESSION_TTL = 3 * 86400.0   # 空闲会话淘汰阈值（秒）：3 天无消息
_SESSION_MAX = 500           # 会话数硬上限（黑名单模式下防无限增长）
_SWEEP_INTERVAL = 600.0      # 扫描节流：最多每 10 分钟扫一次


class ChatSession:
    """单个聊天会话（群或私聊）。

    chat_id 格式：群聊 "{platform}:{group_id}:group"，私聊 "{platform}:{user_id}:private"
    """
    def __init__(self, chat_id: str, platform: str = "qq", group_id: Optional[str] = None, user_id: Optional[str] = None):
        self.chat_id = chat_id
        self.platform = platform
        self.group_id = group_id
        self.user_id = user_id
        # 上层注入槽位（junjun_agent 填充，core 不感知具体类型）
        self.memory = None            # ShortTermMemory
        self.agent = None             # JunJunAgent
        self.silenced_until_call = False  # no_reply_until_call 沉默模式
        self.last_active_ts = 0.0     # 最后收到消息时间（主动系统空闲判定）

    @property
    def is_group(self) -> bool:
        return self.group_id is not None


class ChatSessionManager:
    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._last_sweep = 0.0
        # 淘汰回调（junjun_agent 层注册：关 agent 连接池 + 清会话队列条目）
        self.on_evict: Optional[Callable[[ChatSession], None]] = None

    def get_or_create(self, message_base) -> ChatSession:
        """从 maim_message MessageBase 推导 chat_id 并返回会话。"""
        info = message_base.message_info
        platform = info.platform
        group_info = info.group_info
        if group_info:
            chat_id = f"{platform}:{group_info.group_id}:group"
            if chat_id not in self._sessions:
                self._sessions[chat_id] = ChatSession(chat_id, platform, group_id=str(group_info.group_id))
        else:
            uid = info.user_info.user_id
            chat_id = f"{platform}:{uid}:private"
            if chat_id not in self._sessions:
                self._sessions[chat_id] = ChatSession(chat_id, platform, user_id=str(uid))
        self._maybe_sweep()
        return self._sessions[chat_id]

    def _maybe_sweep(self) -> None:
        now = time.time()
        if now - self._last_sweep >= _SWEEP_INTERVAL:
            self._last_sweep = now
            self.evict_idle()

    def evict_idle(self, ttl: float = _SESSION_TTL,
                   max_sessions: int = _SESSION_MAX) -> int:
        """淘汰空闲超 TTL 的会话；仍超上限则按最久未活跃继续淘汰。返回淘汰数。"""
        now = time.time()
        victims = [cid for cid, s in self._sessions.items()
                   if s.last_active_ts and now - s.last_active_ts > ttl]
        over = len(self._sessions) - len(victims) - max_sessions
        if over > 0:
            rest = sorted(
                (s for cid, s in self._sessions.items() if cid not in victims),
                key=lambda s: s.last_active_ts)
            victims += [s.chat_id for s in rest[:over]]
        for cid in victims:
            session = self._sessions.pop(cid, None)
            if session is None:
                continue
            if self.on_evict is not None:
                try:
                    self.on_evict(session)
                except Exception:
                    pass
            _close_agent_pool(session)
        return len(victims)

    def all_sessions(self) -> Dict[str, ChatSession]:
        return self._sessions


def _close_agent_pool(session: ChatSession) -> None:
    """关闭会话 Agent 的模型连接池（有事件循环则异步调，没有则跳过）。"""
    agent = getattr(session, "agent", None)
    aclose = getattr(agent, "aclose", None) if agent is not None else None
    if aclose is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    async def _safe():
        try:
            await aclose()
        except Exception:
            pass
    loop.create_task(_safe())


session_manager = ChatSessionManager()

def get_session_manager() -> ChatSessionManager:
    return session_manager
