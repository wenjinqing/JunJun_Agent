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


# ---------- 配置热改（WebUI）----------

_listeners: list = []


def register_config_listener(fn) -> None:
    """注册热改监听：fn(changed: list[str])，changed 为 "section.key=value" 列表。

    大多数组件每次使用都实时读 raw（天然热生效）；缓存型消费者（如模型槽）
    应注册监听以便失效重建。
    """
    _listeners.append(fn)


def notify_config_changed(changed: list) -> None:
    for fn in _listeners:
        try:
            fn(changed)
        except Exception:
            pass  # 监听者异常不影响主链路


def persist_bot_config(changed: list, path: Optional[Path] = None) -> None:
    """把热改键写回 bot_config.toml（tomlkit 往返保留注释；tmp+replace 原子写）。

    changed: [(section, key), ...]——只写回这些键，避免把 env 插值后的值固化进文件。
    原文件中值为 ${VAR} 占位符的键跳过不写。
    """
    path = path or (CONFIG_DIR / "bot_config.toml")
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc = tomlkit.parse(f.read())
    cfg = get_global_config()
    for section, key in changed:
        original = doc.get(section, {}).get(key) if section in doc else None
        if isinstance(original, str) and _VAR_RE.search(original):
            continue  # ${VAR} 占位符键不写回
        if section not in doc:
            doc[section] = tomlkit.table()
        doc[section][key] = cfg.raw[section][key]
    tmp = path.with_suffix(".toml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))
    tmp.replace(path)
