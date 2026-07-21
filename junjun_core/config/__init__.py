"""配置加载入口。"""
from .config import (
    GlobalConfig,
    get_global_config,
    load_toml,
    register_config_listener,
    notify_config_changed,
    persist_bot_config,
)

__all__ = [
    "GlobalConfig", "get_global_config", "load_toml",
    "register_config_listener", "notify_config_changed", "persist_bot_config",
]
