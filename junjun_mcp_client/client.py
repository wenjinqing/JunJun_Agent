"""MCP 客户端：连接多个 server，工具转 LangChain BaseTool 注入 registry。

- config/mcp_servers.toml 声明 server（command/args/cwd/env，stdio 传输）
- 启动逐个连接（60s 超时），失败降级跳过不阻塞
- 工具命名空间 mcp_<server>_<tool>，与内置 skill 冲突由 registry 重名报错承担

2026-07-31 P1-9b 持久 session（原：每次工具调用冷启动子进程——Windows 下
npx 冷启动数秒且计入 30s 工具超时，慢且超时常发）：
- 启动时为每个 server 建立持久 session（进程常驻，调用只发 JSON-RPC）
- 看门狗每 60s 健康检查（list_tools 5s 超时），失败自动重连并重绑工具
- 工具调用经 holder 间接寻址，重连后无需重注册
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, List, Optional

import tomlkit

from junjun_core.observability import get_logger

logger = get_logger("mcp.client")


class _NonJsonNoiseFilter(logging.Filter):
    """只压 mcp stdio 的「非 JSON 行」噪音（某些 server 把数据 print 到 stdout
    污染协议流，解析失败不影响数据），其余协议错误/断连照常 WARN——
    原 CRITICAL 一刀切把真错误也吞了，线上排障零线索。"""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if record.levelno <= logging.ERROR and "JSON" in msg:
            return False
        return True


logging.getLogger("mcp.client.stdio").addFilter(_NonJsonNoiseFilter())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_CONFIG = PROJECT_ROOT / "config" / "mcp_servers.toml"

_CONNECT_TIMEOUT = 60.0   # 冷启动 npx/uvx 首次解析+下载较慢，60s 实测不够
_TOOL_TIMEOUT = 30.0
_RESULT_MAX_CHARS = 2000
_WATCHDOG_INTERVAL = 60.0  # 健康检查间隔（秒）
_HEALTH_TIMEOUT = 5.0      # 健康检查超时（秒）

import re as _re

# JSON-RPC 确定性错误（请求无效/方法不存在/参数错）：重试无意义，直接降级
_DETERMINISTIC_MCP_RE = _re.compile(r"MCP error -3260[012]")


class _DeterministicMcpError(ValueError):
    """确定性 MCP 失败的包装（ValueError 子类 → retry_async 的 _NO_RETRY 直接放行）。"""

# 仅管理员可用的 MCP 工具（按工具原名匹配，注册时包权限门）
# apply_relationship_penalty：惩罚处置行为，不能交给群友触发
_ADMIN_TOOLS = {"apply_relationship_penalty"}


def load_server_configs() -> Dict[str, dict]:
    """读 mcp_servers.toml。文件缺失返回空。

    env 值支持 "" 占位符——从进程环境变量（.env）替换，
    秘钥不落 toml（该文件入库）。
    """
    if not MCP_CONFIG.exists():
        return {}
    import os
    with open(MCP_CONFIG, "r", encoding="utf-8") as f:
        data = tomlkit.parse(f.read()).unwrap()

    def _sub(value: str) -> str:
        if value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    servers = {}
    for name, cfg in data.get("servers", {}).items():
        if not cfg.get("enable", True):
            continue
        raw_env = dict(cfg.get("env", {}))
        servers[name] = {
            "transport": "stdio",
            "command": str(cfg["command"]).replace("${REPO_ROOT}", str(PROJECT_ROOT)),
            "args": [str(a).replace("${REPO_ROOT}", str(PROJECT_ROOT)) for a in cfg.get("args", [])],
            "cwd": str(cfg.get("cwd", "")).replace("${REPO_ROOT}", str(PROJECT_ROOT)) or None,
            "env": {k: _sub(str(v)) for k, v in raw_env.items()} or None,
        }
    return servers


class MCPManager:
    def __init__(self):
        self._clients: Dict[str, object] = {}     # server 名 -> MultiServerMCPClient
        self._sessions: Dict[str, object] = {}    # server 名 -> 持久 ClientSession
        self._stacks: Dict[str, AsyncExitStack] = {}  # server 名 -> session 生命周期栈
        self._holders: Dict[str, List[dict]] = {}  # server 名 -> 工具 coro 间接寻址 holder
        self._configs: Dict[str, dict] = {}
        self._tools: List = []
        self._watchdog: Optional[asyncio.Task] = None

    @property
    def tools(self) -> List:
        return self._tools

    async def start(self) -> int:
        """并发连接全部 server（持久 session）并拉工具。返回可用工具数；全失败返回 0 不抛。"""
        configs = load_server_configs()
        if not configs:
            logger.info("无 MCP server 配置，跳过")
            return 0
        self._configs = configs
        results = await asyncio.gather(
            *[self._connect_one(name, cfg) for name, cfg in configs.items()])
        # 命名空间前缀 + 结果截断包装（经 holder 间接寻址，重连后自动指向新 session）
        self._tools = []
        for name, tools in results:
            for t in tools:
                self._tools.append(self._wrap(t, name))
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.create_task(self._watchdog_loop(), name="mcp-watchdog")
        return len(self._tools)

    async def stop(self) -> None:
        """关闭全部持久 session 与子进程（进程退出前调用）。"""
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        for name, stack in list(self._stacks.items()):
            try:
                await stack.aclose()
            except Exception as e:
                logger.debug(f"MCP server [{name}] session 关闭异常（忽略）: {e}")
        self._stacks.clear()
        self._sessions.clear()

    async def _connect_one(self, name: str, cfg: dict) -> tuple:
        """单 server：持久 session（重试 3 次）+ 拉工具。失败返回 (name, []) 不影响其他。"""
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
        for attempt in (1, 2, 3):
            stack = AsyncExitStack()
            try:
                client = MultiServerMCPClient({name: cfg})
                session = await asyncio.wait_for(
                    stack.enter_async_context(client.session(name)),
                    timeout=_CONNECT_TIMEOUT)
                tools = await asyncio.wait_for(load_mcp_tools(session),
                                               timeout=_CONNECT_TIMEOUT)
                self._clients[name] = client
                self._sessions[name] = session
                self._stacks[name] = stack
                self._holders.setdefault(name, [])
                logger.info(f"MCP server [{name}] 持久 session 已建立: {len(tools)} 个工具")
                return name, tools
            except Exception as e:
                try:
                    await stack.aclose()
                except Exception:
                    pass
                if attempt == 3:
                    logger.warning(f"MCP server [{name}] 重试 3 次均失败（降级跳过）: {type(e).__name__}: {e}")
                else:
                    logger.info(f"MCP server [{name}] 第 {attempt} 次连接失败，重试: {type(e).__name__}")
        return name, []

    # ---------- 看门狗：健康检查 + 自动重连 ----------

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(_WATCHDOG_INTERVAL)
            for name in list(self._sessions.keys()):
                try:
                    await asyncio.wait_for(self._sessions[name].list_tools(),
                                           timeout=_HEALTH_TIMEOUT)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"MCP server [{name}] 健康检查失败，重连: "
                                   f"{type(e).__name__}: {e}")
                    await self._reconnect(name)

    async def _reconnect(self, name: str) -> None:
        """重建持久 session 并按工具名重绑 holder（已注册的工具无需重注册）。"""
        cfg = self._configs.get(name)
        if cfg is None:
            return
        old = self._stacks.pop(name, None)
        if old is not None:
            try:
                await old.aclose()
            except Exception:
                pass
        self._sessions.pop(name, None)
        _name, tools = await self._connect_one(name, cfg)
        new_by_name = {t.name: t for t in tools}
        rebound = 0
        for holder in self._holders.get(name, []):
            nt = new_by_name.get(holder["tool_name"])
            if nt is not None and nt.coroutine is not None:
                holder["coro"] = nt.coroutine
                rebound += 1
        logger.info(f"MCP server [{name}] 重连完成，重绑 {rebound} 个工具")

    def register_all(self) -> None:
        """注入 skill registry（重名由 registry 报错）。_ADMIN_TOOLS 包权限门。"""
        from junjun_skills.registry import register
        for t in self._tools:
            try:
                # t.name 已在 _wrap 加 mcp_ 前缀；匹配原始名
                raw_name = t.name[len("mcp_"):] if t.name.startswith("mcp_") else t.name
                register(t, plugin="mcp", admin_only=raw_name in _ADMIN_TOOLS)
            except ValueError as e:
                logger.warning(f"MCP 工具注册冲突（跳过）: {e}")

    def _wrap(self, tool, server_name: str = ""):
        """加 mcp_ 前缀 + 超时 + 结果截断 + 内容提取。

        langchain-mcp-adapters 工具是 content_and_artifact 格式：
        coroutine 返回 (content, artifact) 二元组——包装必须保持该结构。

        2026-07-24 调整（用户反馈格式难看）：
        - content 是 [{'type': 'text', 'text': '...'}] 列表时，提取纯文本
        - 不用 markdown 格式，纯文本输出

        2026-07-31：coroutine 经 holder 间接寻址——看门狗重连后 holder 指向
        新 session 的工具，已注册到 registry 的包装对象无需变更。
        """
        original_coro = tool.coroutine
        if original_coro is not None:
            holder = {"tool_name": tool.name, "coro": original_coro}
            self._holders.setdefault(server_name, []).append(holder)

            async def guarded(*args, _holder=holder, **kwargs):
                from junjun_core.retry import retry_async

                async def _call():
                    try:
                        return await asyncio.wait_for(_holder["coro"](*args, **kwargs),
                                                      timeout=_TOOL_TIMEOUT)
                    except Exception as e:
                        # 确定性失败（-32600/-32601/-32602 参数/方法错）重试无意义，
                        # 包装成 ValueError 让 retry_async 直接放行（2026-08-03 实战：
                        # BV 号传错格式被重试 3 次，白等 3s+ 还刷 3 条服务端报错）
                        if _DETERMINISTIC_MCP_RE.search(str(e)):
                            raise _DeterministicMcpError(str(e)) from e
                        raise

                try:
                    # 瞬态失败（网络抖动/限流/ECONNRESET）重试 3 次再降级
                    result = await retry_async(_call, attempts=3, base_delay=1.0,
                                               label=tool.name)
                except asyncio.TimeoutError:
                    return "工具调用超时（30s），请换个方式或稍后再试。", None
                except _DeterministicMcpError as e:
                    # 参数类失败：服务端报错本身含正确用法，原样喂回让模型自我纠正
                    logger.info(f"MCP 工具 {tool.name} 参数类失败（不重试）: {e}")
                    return (f"工具拒绝了这次调用：{str(e)[:200]}。"
                            f"按报错里的提示修正参数再试，或换其他方式回答。"), None
                except Exception as e:
                    # 重试 3 次仍失败：降级为工具结果文本，
                    # 绝不外抛——ToolException 会炸掉整个 agent 轮次导致沉默
                    logger.warning(f"MCP 工具 {tool.name} 重试 3 次均失败（降级为错误文本）: "
                                   f"{type(e).__name__}: {e}")
                    return (f"这个工具调用失败了（{type(e).__name__}: {str(e)[:150]}），"
                            f"换其他方式回答，或直接告诉用户暂时查不了。"), None

                content, artifact = result if isinstance(result, tuple) else (result, None)
                text = _extract_text(content) if content is not None else ""
                if len(text) > _RESULT_MAX_CHARS:
                    # 超长按合并转发打包（防刷屏 + 不丢内容）
                    import json
                    nickname = "君君"
                    nodes = [{
                        "type": "node",
                        "data": {
                            "name": nickname,
                            "uin": "",
                            "content": [{"type": "text", "data": {"text": text}}],
                        },
                    }]
                    return json.dumps({
                        "type": "forward",
                        "text": f"📋 {tool.name} 结果（共 {len(text)} 字）",
                        "nodes": nodes,
                    }, ensure_ascii=False), artifact
                return text, artifact
            tool.coroutine = guarded
        if not tool.name.startswith("mcp_"):
            tool.name = f"mcp_{tool.name}"
        return tool


def _extract_text(content) -> str:
    """从 MCP content 提取纯文本。

    content 可能是：
    - 字符串：直接返回
    - [{'type': 'text', 'text': '...'}]：提取 text 字段拼接
    - 其他对象：str() 兜底
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else str(content)
    return str(content)


mcp_manager = MCPManager()


