"""命令总线：0-token 命令层（插件迁移基础设施）。

旧插件的 /cmd 命令在新架构不该走 LLM（浪费 token 且不可靠）——
在决策漏斗之前直接拦截处理。

注册：
    from junjun_agent.commands import register_command

    @register_command("draw", aliases=["绘图", "画图"], plugin="ai_draw",
                      description="AI 画图")
    async def draw_cmd(ctx) -> str:
        return f"画好了: ..."      # 返回 str = 文本回复

    # 关键词命令（无 "/" 前缀，全句匹配或句首匹配）：
    @register_command("抽老婆", raw=True, plugin="wife")
    async def wife_cmd(ctx): ...

CommandContext 提供：session/meta/args/reply()/send()。
admin_only=True 时非管理员触发会被拒并上报（走 security 同一套）。

processor 在表达反思拦截之后、复读/漏斗之前调用 dispatch()；
返回 True 表示消息已被命令消费，不再进漏斗。
"""

from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from junjun_core.contracts import ReplySegment, ReplySet
from junjun_core.observability import get_logger

logger = get_logger("commands")


@dataclass
class CommandContext:
    """命令处理上下文。"""
    session: object          # ChatSession
    meta: object             # InboundMeta
    args: str = ""           # 命令名之后的参数文本（已 strip）

    async def reply(self, text: str) -> None:
        """文本回复到当前会话。"""
        await self.send([ReplySegment(type="text", data=text)])

    async def send(self, segments: List[ReplySegment]) -> None:
        """任意段回复到当前会话（image/video/at/poke 等）。"""
        from junjun_core.gateway.router import get_gateway
        await get_gateway().send_reply(ReplySet(
            platform=self.session.platform,
            target_group_id=self.session.group_id,
            target_user_id=None if self.session.is_group else self.meta.user_id,
            segments=segments,
            should_reply=True,
        ))

    async def send_forward(self, title: str, content: str, *, nickname: str = "君君") -> None:
        """长内容合并转发（防刷屏）。单条 >200 字或含多行列表时优先用此。

        content 为完整文本（可多行），adapter 打包为 OneBot 合并转发发出。
        """
        import json
        nodes = [{
            "type": "node",
            "data": {
                "name": nickname,
                "uin": str(getattr(self.session, "bot_user_id", "") or "10000001"),
                "content": [{"type": "text", "data": {"text": content}}],
            },
        }]
        await self.send([ReplySegment(type="text", data=f"📋 {title}"),
                        ReplySegment(type="forward", data=json.dumps(nodes, ensure_ascii=False))])


# handler: async (CommandContext) -> Optional[str]；返回 str 自动作为文本回复
Handler = Callable[[CommandContext], Awaitable[Optional[str]]]


@dataclass
class Command:
    name: str
    handler: Handler
    plugin: str = "builtin"
    aliases: tuple = ()
    raw: bool = False                # True=关键词命令（整句/句首匹配，不要求 "/" 前缀）
    admin_only: bool = False
    description: str = ""


_commands: List[Command] = []


def register_command(name: str, *, aliases=(), plugin: str = "builtin",
                     raw: bool = False, admin_only: bool = False,
                     description: str = ""):
    """装饰器：注册命令。"""
    def deco(fn: Handler) -> Handler:
        _commands.append(Command(
            name=name, handler=fn, plugin=plugin, aliases=tuple(aliases),
            raw=raw, admin_only=admin_only, description=description,
        ))
        logger.debug(f"注册命令: {'(raw) ' if raw else ''}/{name} [{plugin}]")
        return fn
    return deco


def list_commands() -> List[dict]:
    return [{"name": c.name, "plugin": c.plugin, "raw": c.raw,
             "admin_only": c.admin_only, "description": c.description}
            for c in _commands]


def _match(text: str) -> Optional[tuple]:
    """匹配命令。返回 (Command, args) 或 None。

    "/" 开头的文本不会误中 raw 命令：raw 匹配要求整句相等或「关键词+空格」
    开头，"/关键词" 两种都不满足，天然安全，无需特判。
    """
    if not text:
        return None
    for c in _commands:
        names = (c.name, *c.aliases)
        if c.raw:
            # 关键词命令：整句相等，或「关键词+空格/参数」开头
            for n in names:
                if text == n:
                    return c, ""
                if text.startswith(n) and len(text) > len(n) and text[len(n)] in " \t":
                    return c, text[len(n):].strip()
        elif text.startswith("/"):
            body = text[1:]
            for n in names:
                if body == n:
                    return c, ""
                if body.startswith(n) and len(body) > len(n) and body[len(n)] in " \t":
                    return c, body[len(n):].strip()
    return None


async def dispatch(session, meta) -> bool:
    """尝试按命令消费消息。True=已处理（不进漏斗）。"""
    text = (meta.text or "").strip()
    hit = _match(text)
    if not hit:
        return False
    cmd, args = hit

    from junjun_skills import registry
    if not registry.is_plugin_enabled(cmd.plugin):
        logger.debug(f"命令 /{cmd.name} 所属插件 [{cmd.plugin}] 已禁用，忽略")
        return False

    if cmd.admin_only:
        from junjun_core.security import is_admin_privileged, report_violation
        if not is_admin_privileged():
            report_violation(f"管理员命令 /{cmd.name}", meta.user_id or "",
                             meta.nickname, session.chat_id, text[:60])
            ctx = CommandContext(session=session, meta=meta, args=args)
            await ctx.reply("这个命令只有管理员能用哦（已通知管理员）。")
            return True

    ctx = CommandContext(session=session, meta=meta, args=args)
    try:
        result = await cmd.handler(ctx)
        if isinstance(result, str) and result:
            await ctx.reply(result)
        logger.info(f"命令已处理 [/{cmd.name}] [{session.chat_id}] {meta.nickname}")
    except Exception as e:
        logger.error(f"命令 /{cmd.name} 执行异常: {type(e).__name__}: {e}")
        try:
            await ctx.reply(f"命令执行出错了（{type(e).__name__}），稍后再试试吧。")
        except Exception:
            pass
    return True


def clear_commands() -> None:
    """仅供测试。"""
    _commands.clear()
