"""短期记忆：按会话滑动窗口。

群聊消息渲染带昵称前缀，Agent 能分清谁在说话。
阶段 4 升级为 LangGraph checkpointer；本阶段内存窗口。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryEntry:
    role: str  # "user" / "bot"
    text: str
    nickname: str = ""
    user_id: str = ""
    message_id: str = ""
    at_bot: bool = False


@dataclass
class ShortTermMemory:
    max_size: int = 80
    entries: List[MemoryEntry] = field(default_factory=list)

    def add_user(self, text: str, nickname: str, user_id: str = "",
                 message_id: str = "", at_bot: bool = False) -> None:
        self.entries.append(MemoryEntry(
            role="user", text=text, nickname=nickname,
            user_id=user_id, message_id=message_id, at_bot=at_bot,
        ))
        self._trim()

    def add_bot(self, text: str) -> None:
        self.entries.append(MemoryEntry(role="bot", text=text))
        self._trim()

    def _trim(self) -> None:
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]

    def render(self, limit: Optional[int] = None) -> str:
        """渲染为对话文本（供 prompt）。群聊格式 `昵称: 内容`。"""
        entries = self.entries[-limit:] if limit else self.entries
        lines = []
        for e in entries:
            if e.role == "bot":
                lines.append(f"你: {e.text}")
            else:
                prefix = f"{e.nickname or e.user_id}"
                mark = " [@你]" if e.at_bot else ""
                lines.append(f"{prefix}{mark}: {e.text}")
        return "\n".join(lines)

    def last_user_entry(self) -> Optional[MemoryEntry]:
        for e in reversed(self.entries):
            if e.role == "user":
                return e
        return None
