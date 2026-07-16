"""会话管理：按 platform:chat_id 维护会话状态。

分层约束：junjun_core 不 import 上层包。memory/agent 为通用槽位，
由 junjun_agent 层在首次处理时注入（processor 模式，见 gateway/router.py）。
"""

from typing import Dict, Optional


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
        return self._sessions[chat_id]

    def all_sessions(self) -> Dict[str, ChatSession]:
        return self._sessions


session_manager = ChatSessionManager()

def get_session_manager() -> ChatSessionManager:
    return session_manager
