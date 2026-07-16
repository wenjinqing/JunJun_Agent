"""配置加载入口。"""
from .config import GlobalConfig, get_global_config, load_toml

__all__ = ["GlobalConfig", "get_global_config", "load_toml"]
