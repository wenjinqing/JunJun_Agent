"""短期记忆：按会话滑动窗口。

群聊消息渲染带昵称前缀，Agent 能分清谁在说话。
阶段 4 升级为 LangGraph checkpointer；本阶段内存窗口。
"""

from dataclasses import dataclass, field
from typing import List, Optional

# 管理员锚点防伪：昵称/消息内容里不得出现系统标记样式或换行，
# 否则群名片「xx(管理员)」或消息内伪造行能冒充系统打的标记（代码层
# is_admin_privileged 闸门不受影响，但 LLM 认知锚点会被污染）
_ADMIN_MARKER_TOKENS = ("(管理员)", "（管理员）")


def _sanitize_nickname(nickname: str) -> str:
    """昵称里的系统标记样式一律剥掉（用户可任意改群名片，不可信）。"""
    name = nickname or ""
    for token in _ADMIN_MARKER_TOKENS:
        name = name.replace(token, "")
    return name.replace("\n", " ").strip()


def _sanitize_text(text: str) -> str:
    """消息内容换行压成可见符号——防止消息体内伪造一行「昵称(管理员): ...」。"""
    return (text or "").replace("\r", " ").replace("\n", " ⏎ ")


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
               include_bot: bool = True, for_security: bool = False) -> str:
        """渲染为对话文本（供 prompt）。群聊格式 `昵称: 内容`。

        管理员消息带「(管理员)」系统标记——按真实 user_id 判定，聊天内容无法伪造，
        是 LLM 识别管理员指令的锚点（配合 persona 安全段）。

        mark_latest: True 时最后一条 user 消息前缀「【最新】」——帮模型聚焦。
        include_bot: True 时 bot 回复进 context（记忆效果），但前缀「你(历史):」
        明确标记为「已发生的历史输出」而非「待接续的话」（防复读关键）。
        for_security: True 时保留（管理员）标记（安全验证用）；
        False（默认）时管理员显示为普通群友（不影响回复意愿）。

        边界感知（LangChain trim_messages 语义）：永远以 user 消息开头，
        不从 bot 回复中间截断——模型不会把被截断的历史当成待续写文本。
        """
        from junjun_core.security import is_admin
        from junjun_memory.echo import normalize_echo
        entries = self.entries[-limit:] if limit else self.entries
        # bot 历史去重（防上下文自污染，2026-08-04）：同一句话术只保留最近一次
        # 出现——否则 context 里「你(历史): xxx」堆 N 次，模型把它当成自己的
        # 说话习惯继续复读，形成正反馈污染循环
        bot_last = {}
        for i, e in enumerate(entries):
            if e.role == "bot":
                n = normalize_echo(e.text)
                if n:
                    bot_last[n] = i
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
                    n = normalize_echo(e.text)
                    if n and bot_last.get(n) != i:
                        continue  # 更早的重复出现：跳过，只留最近一次
                    # 标记为「历史输出」而非「待接续的话」（防复读关键）
                    lines.append(f"你(历史): {e.text}")
                # 默认不进 context（include_bot=False 时）
            else:
                prefix = _sanitize_nickname(e.nickname) or e.user_id
                # 管理员标记：for_security=True 才保留（安全验证用），
                # 默认不显示——L2/L3 看到的都是普通群友，不影响回复意愿
                if for_security and is_admin(e.user_id):
                    prefix += "(管理员)"
                mark = " [@你]" if e.at_bot else ""
                if mark_latest and i == last_user_idx:
                    prefix = f"【最新】{prefix}"
                lines.append(f"{prefix}{mark}: {_sanitize_text(e.text)}")
        return "\n".join(lines)

    def last_user_entry(self) -> Optional[MemoryEntry]:
        for e in reversed(self.entries):
            if e.role == "user":
                return e
        return None
