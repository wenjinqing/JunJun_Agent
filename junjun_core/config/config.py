"""君君配置加载：支持 toml + ${VAR} 环境变量插值。"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import tomlkit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _interpolate(value: Any, *, strict: bool = False) -> Any:
    """递归替换字符串中的 ${VAR} 占位符。

    strict=True: 未设置或空值时报错。
    strict=False: 未设置/空值时替换为空字符串（骨架阶段友好）。
    """
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            val = os.environ.get(var)
            if val is None or val == "":
                if strict:
                    raise ValueError(f"配置引用的环境变量未设置: {var}")
                return ""
            return val
        return _VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, strict=strict) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, strict=strict) for v in value]
    return value


def load_toml(path: Path, *, strict_env: bool = False) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc = tomlkit.parse(f.read())
    return _interpolate(doc.unwrap(), strict=strict_env)


@dataclass
class BotConfig:
    platform: str = "qq"
    qq_account: str = ""
    nickname: str = "君君"
    alias_names: list = field(default_factory=list)


@dataclass
class GlobalConfig:
    bot: BotConfig = field(default_factory=BotConfig)
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[Path] = None, *, strict_env: bool = False) -> "GlobalConfig":
        path = path or (CONFIG_DIR / "bot_config.toml")
        data = load_toml(path, strict_env=strict_env)
        bot_data = data.get("bot", {})
        bot = BotConfig(
            platform=bot_data.get("platform", "qq"),
            qq_account=bot_data.get("qq_account", ""),
            nickname=bot_data.get("nickname", "君君"),
            alias_names=bot_data.get("alias_names", []),
        )
        return cls(bot=bot, raw=data)


global_config: Optional[GlobalConfig] = None


def get_global_config() -> GlobalConfig:
    global global_config
    if global_config is None:
        global_config = GlobalConfig.load()
    return global_config
