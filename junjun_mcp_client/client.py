"""MCP 客户端：连接多 server，工具转 LangChain BaseTool 注入 registry。

- config/mcp_servers.toml 声明 server（command/args/cwd/env，stdio 传输）
- 启动逐个连接（60s 超时），失败降级跳过不阻塞
- 工具命名空间 mcp_<server>_<tool>，与内置 skill 冲突检测由 registry 重名报错承担
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List

import tomlkit

from junjun_core.observability import get_logger

logger = get_logger("mcp.client")

# mcp SDK 的 stdout_reader 对非 JSON 行打 logger.exception——
# 某些第三方 server（bilibili-mcp-js 等）会把数据 print 到 stdout 污染协议流。
# 解析失败静默（数据不会丢，只是 server 自己的日志噪音），其他错误仍 WARN。
logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_CONFIG = PROJECT_ROOT / "config" / "mcp_servers.toml"

_CONNECT_TIMEOUT = 60.0   # 冷启动 npx/uvx 首次解析+下载较慢（10s 实测不够）
_TOOL_TIMEOUT = 30.0
_RESULT_MAX_CHARS = 2000

# 仅管理员可用的 MCP 工具（按工具原名匹配，注册时包权限门）
# apply_relationship_penalty：惩罚是处置行为，不能交给群友触发
_ADMIN_TOOLS = {"apply_relationship_penalty"}


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

        # 逐 server 隔离连接：一个坏不拖全部；冷启动慢，失败重试一次
        ok_configs = {}
        for name, cfg in configs.items():
            for attempt in (1, 2):
                try:
                    probe = MultiServerMCPClient({name: cfg})
                    tools = await asyncio.wait_for(probe.get_tools(), timeout=_CONNECT_TIMEOUT)
                    ok_configs[name] = cfg
                    logger.info(f"MCP server [{name}] 连接成功: {len(tools)} 个工具")
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"MCP server [{name}] 连接失败（降级跳过）: {type(e).__name__}: {e}")
                    else:
                        logger.info(f"MCP server [{name}] 首次连接失败，重试: {type(e).__name__}")

        if not ok_configs:
            return 0
        self._client = MultiServerMCPClient(ok_configs)
        # get_tools 可能因某个 server 的 stdio 污染/bug 抛异常——
        # 逐 server 拉取，一个失败不影响其他（对齐 start 的降级语义）
        raw_tools = []
        for name in ok_configs:
            try:
                single = MultiServerMCPClient({name: ok_configs[name]})
                tools = await asyncio.wait_for(single.get_tools(), timeout=_CONNECT_TIMEOUT)
                raw_tools.extend(tools)
            except Exception as e:
                logger.warning(f"MCP server [{name}] 拉取工具失败（跳过）: {type(e).__name__}: {e}")

        # 命名空间前缀 + 结果截断包装
        self._tools = [self._wrap(t) for t in raw_tools]
        return len(self._tools)

    def _wrap(self, tool):
        """加 mcp_ 前缀 + 超时 + 结果截断。

        langchain-mcp-adapters 工具是 content_and_artifact 格式，
        coroutine 返回 (content, artifact) 二元组——包装必须保持该结构。

        2026-07-22 调整（用户反馈 MCP 长结果被拆太碎）：
        - 结果超长时不再截断，而是**拼接为合并转发格式**（forward 段）
        - 单条超过 _FORWARD_THRESHOLD 才走合并转发，短结果仍直发
        """
        original_coro = tool.coroutine
        if original_coro is not None:
            async def guarded(*args, _orig=original_coro, **kwargs):
                try:
                    result = await asyncio.wait_for(_orig(*args, **kwargs), timeout=_TOOL_TIMEOUT)
                except asyncio.TimeoutError:
                    return "工具调用超时（30s），请换个方式或稍后再试。", None

                content, artifact = result if isinstance(result, tuple) else (result, None)
                text = str(content) if content is not None else ""
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
                return content, artifact
            tool.coroutine = guarded
        if not tool.name.startswith("mcp_"):
            tool.name = f"mcp_{tool.name}"
        return tool

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


mcp_manager = MCPManager()
