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

    def render(self, limit: Optional[int] = None, *, mark_latest: bool = False,
               include_bot: bool = True) -> str:
        """渲染为对话文本（供 prompt）。群聊格式 `昵称: 内容`。

        管理员消息带「(管理员)」系统标记——按真实 user_id 判定，聊天内容无法伪造，
        是 LLM 识别管理员指令的锚点（配合 persona 安全段）。

        mark_latest: True 时最后一条 user 消息前缀「【最新】」——帮模型聚焦。
        include_bot: True 时 bot 回复进 context（记忆效果），但前缀「你(历史):」
        明确标记为「已发生的历史输出」而非「待接续的话」（防复读关键）。

        边界感知（LangChain trim_messages 语义）：永远以 user 消息开头，
        不从 bot 回复中间截断——模型不会把被截断的历史当成待续写文本。
        """
        from junjun_core.security import is_admin
        entries = self.entries[-limit:] if limit else self.entries
        lines = []
        # 找最后一条 user 消息的下标（mark_latest 用）
        last_user_idx = -1
        if mark_latest:
            for i in range(len(entries) - 1, -1, -1):
                if entries[i].role == "user":
                    last_user_idx = i
                    break
        # 边界感知：跳过开头的 bot 消息（不从 bot 回复中间截断）
        start = 0
        while start < len(entries) and entries[start].role == "bot":
            start += 1
        for i, e in enumerate(entries[start:], start=start):
            if e.role == "bot":
                if include_bot:
                    # 标记为「历史输出」而非「待接续的话」（防复读关键）
                    lines.append(f"你(历史): {e.text}")
                # 默认不进 context（include_bot=False 时）
            else:
                prefix = f"{e.nickname or e.user_id}"
                if is_admin(e.user_id):
                    prefix += "(管理员)"
                mark = " [@你]" if e.at_bot else ""
                if mark_latest and i == last_user_idx:
                    prefix = f"【最新】{prefix}"
                lines.append(f"{prefix}{mark}: {e.text}")
        return "\n".join(lines)

    def last_user_entry(self) -> Optional[MemoryEntry]:
        for e in reversed(self.entries):
            if e.role == "user":
                return e
        return None
