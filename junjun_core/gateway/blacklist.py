"""黑白名单：群/私聊/ban_user/ban_bot。"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChatListConfig:
    group_list_type: str = "whitelist"  # whitelist / blacklist
    group_list: List[int] = field(default_factory=list)
    private_list_type: str = "whitelist"
    private_list: List[int] = field(default_factory=list)
    ban_user_id: List[int] = field(default_factory=list)
    ban_qq_bot: bool = False

    @classmethod
    def from_raw(cls, raw_chat: dict) -> "ChatListConfig":
        return cls(
            group_list_type=raw_chat.get("group_list_type", "whitelist"),
            group_list=[int(x) for x in raw_chat.get("group_list", [])],
            private_list_type=raw_chat.get("private_list_type", "whitelist"),
            private_list=[int(x) for x in raw_chat.get("private_list", [])],
            ban_user_id=[int(x) for x in raw_chat.get("ban_user_id", [])],
            ban_qq_bot=raw_chat.get("ban_qq_bot", False),
        )

    def allow(self, user_id, group_id: Optional[int] = None, ignore_ban: bool = False) -> bool:
        """判断是否允许聊天。返回 False 则消息被丢弃。"""
        uid = int(user_id) if user_id is not None else None
        gid = int(group_id) if group_id is not None else None
        if gid is not None:
            if self.group_list_type == "whitelist" and gid not in self.group_list:
                return False
            if self.group_list_type == "blacklist" and gid in self.group_list:
                return False
        else:
            if uid is not None:
                if self.private_list_type == "whitelist" and uid not in self.private_list:
                    return False
                if self.private_list_type == "blacklist" and uid in self.private_list:
                    return False
        if not ignore_ban and uid is not None and uid in self.ban_user_id:
            return False
        return True
