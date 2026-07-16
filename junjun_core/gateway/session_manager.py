"""会话管理：按 platform:chat_id 维护会话状态。阶段 1 仅记录，Agent 占位 echo。"""

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
        self.history: list = []  # 阶段 1 占位，阶段 2 接 Agent 上下文

    def add_message(self, text: str) -> None:
        self.history.append(text)
        if len(self.history) > 20:
            self.history = self.history[-20:]


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
            session = ChatSession(chat_id, platform, group_id=str(group_info.group_id))
        else:
            uid = info.user_info.user_id
            chat_id = f"{platform}:{uid}:private"
            session = ChatSession(chat_id, platform, user_id=str(uid))
        if chat_id not in self._sessions:
            self._sessions[chat_id] = session
        return self._sessions[chat_id]

    def all_sessions(self) -> Dict[str, ChatSession]:
        return self._sessions


session_manager = ChatSessionManager()

def get_session_manager() -> ChatSessionManager:
    return session_manager
