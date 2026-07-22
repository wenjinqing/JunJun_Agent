"""MCP 客户端：连接多 server，工具转 LangChain BaseTool 注入 registry。

- config/mcp_servers.toml 声明 server（command/args/cwd/env，stdio 传输）
- 启动逐个连接（10s 超时），失败降级跳过不阻塞
- 工具命名空间 mcp_<server>_<tool>，与内置 skill 冲突检测由 registry 重名报错承担
"""

import asyncio
from pathlib import Path
from typing import Dict, List

import tomlkit

from junjun_core.observability import get_logger

logger = get_logger("mcp.client")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_CONFIG = PROJECT_ROOT / "config" / "mcp_servers.toml"

_CONNECT_TIMEOUT = 10.0
_TOOL_TIMEOUT = 30.0
_RESULT_MAX_CHARS = 2000


def load_server_configs() -> Dict[str, dict]:
    """读 mcp_servers.toml。文件缺失返回空。

    env 值支持 "${VAR}" 占位符——从进程环境变量（.env）替换，
    密钥不落 toml（该文件入库）。
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
        self._client = None
        self._tools: List = []

    @property
    def tools(self) -> List:
        return self._tools

    async def start(self) -> int:
        """连接全部 server 并拉工具。返回可用工具数；全失败返回 0 不抛。"""
        configs = load_server_configs()
        if not configs:
            logger.info("无 MCP server 配置，跳过")
            return 0
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # 逐 server 隔离连接：一个坏不拖全部
        ok_configs = {}
        for name, cfg in configs.items():
            try:
                probe = MultiServerMCPClient({name: cfg})
                tools = await asyncio.wait_for(probe.get_tools(), timeout=_CONNECT_TIMEOUT)
                ok_configs[name] = cfg
                logger.info(f"MCP server [{name}] 连接成功: {len(tools)} 个工具")
            except Exception as e:
                logger.warning(f"MCP server [{name}] 连接失败（降级跳过）: {type(e).__name__}: {e}")

        if not ok_configs:
            return 0
        self._client = MultiServerMCPClient(ok_configs)
        raw_tools = await self._client.get_tools()

        # 命名空间前缀 + 结果截断包装
        self._tools = [self._wrap(t) for t in raw_tools]
        return len(self._tools)

    def _wrap(self, tool):
        """加 mcp_ 前缀 + 超时 + 结果截断。

        langchain-mcp-adapters 工具是 content_and_artifact 格式，
        coroutine 返回 (content, artifact) 二元组——包装必须保持该结构。
        """
        original_coro = tool.coroutine
        if original_coro is not None:
            async def guarded(*args, _orig=original_coro, **kwargs):
                try:
                    result = await asyncio.wait_for(_orig(*args, **kwargs), timeout=_TOOL_TIMEOUT)
                except asyncio.TimeoutError:
                    return "工具调用超时（30s），请换个方式或稍后再试。", None

                def _truncate(text):
                    if isinstance(text, str) and len(text) > _RESULT_MAX_CHARS:
                        return text[:_RESULT_MAX_CHARS] + "…（结果过长已截断）"
                    return text

                if isinstance(result, tuple) and len(result) == 2:
                    return _truncate(result[0]), result[1]
                return _truncate(result)
            tool.coroutine = guarded
        if not tool.name.startswith("mcp_"):
            tool.name = f"mcp_{tool.name}"
        return tool

    def register_all(self) -> None:
        """注入 skill registry（重名由 registry 报错）。"""
        from junjun_skills.registry import register
        for t in self._tools:
            try:
                register(t)
            except ValueError as e:
                logger.warning(f"MCP 工具注册冲突（跳过）: {e}")


mcp_manager = MCPManager()
