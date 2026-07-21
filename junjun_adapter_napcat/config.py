"""君君 NapCat Adapter 配置。"""

import os
from dataclasses import dataclass
from pathlib import Path

import tomlkit


@dataclass
class NapcatServerConfig:
    host: str = "127.0.0.1"
    port: int = 8095
    token: str = ""
    heartbeat_interval: int = 30


@dataclass
class MaibotServerConfig:
    host: str = "127.0.0.1"
    port: int = 8092
    platform_name: str = "qq"
    token: str = ""


@dataclass
class ChatConfig:
    group_list_type: str = "blacklist"
    group_list: list = None
    private_list_type: str = "blacklist"
    private_list: list = None
    ban_user_id: list = None
    ban_qq_bot: bool = False


@dataclass
class AdapterConfig:
    napcat_server: NapcatServerConfig = None
    maibot_server: MaibotServerConfig = None
    chat: ChatConfig = None

    @classmethod
    def load(cls, path=None):
        pkg_dir = Path(__file__).resolve().parent
        path = path or (pkg_dir / "config.toml")
        with open(path, "r", encoding="utf-8") as f:
            data = tomlkit.parse(f.read()).unwrap()
        ns = data.get("napcat_server", {})
        ms = data.get("maibot_server", {})
        ch = data.get("chat", {})
        return cls(
            napcat_server=NapcatServerConfig(
                host=ns.get("host", "127.0.0.1"),
                port=int(ns.get("port", 8095)),
                # 优先读 .env 的 NAPCAT_TOKEN（对应 NapCat 的 accessToken，令牌不入库）
                token=os.environ.get("NAPCAT_TOKEN", "") or ns.get("token", ""),
                heartbeat_interval=int(ns.get("heartbeat_interval", 30)),
            ),
            maibot_server=MaibotServerConfig(
                host=ms.get("host", "127.0.0.1"),
                port=int(ms.get("port", 8092)),
                platform_name=ms.get("platform_name", "qq"),
                # 优先读 .env 的 GATEWAY_TOKEN（与核心共用一处，令牌不入库）；
                # toml 的 token 字段仅作无 .env 环境的回退
                token=os.environ.get("GATEWAY_TOKEN", "") or ms.get("token", ""),
            ),
            chat=ChatConfig(
                group_list_type=ch.get("group_list_type", "blacklist"),
                group_list=list(ch.get("group_list", [])),
                private_list_type=ch.get("private_list_type", "blacklist"),
                private_list=list(ch.get("private_list", [])),
                ban_user_id=list(ch.get("ban_user_id", [])),
                ban_qq_bot=ch.get("ban_qq_bot", False),
            ),
        )


global_config = None


def get_config() -> AdapterConfig:
    global global_config
    if global_config is None:
        global_config = AdapterConfig.load()
    return global_config
