"""pytest 共享 fixture。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _fake_bot_config(monkeypatch):
    """所有测试用固定配置，不读磁盘 toml / 不依赖 .env。"""
    import junjun_core.config.config as cfg_mod

    fake = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(
            platform="qq", qq_account="2477702109",
            nickname="君君", alias_names=["猫娘"],
        ),
        raw={
            "bot": {"qq_account": "2477702109", "nickname": "君君"},
            "chat": {
                "talk_value": 0.9,
                "mentioned_bot_reply": True,
                "max_context_size": 80,
                "group_list_type": "blacklist", "group_list": [],
                "private_list_type": "blacklist", "private_list": [],
                "ban_user_id": [], "ban_qq_bot": False,
            },
            "personality": {
                "personality": "你是君君，测试人设。",
                "reply_style": "简短",
                "interest": "测试",
            },
            "memory": {"max_agent_iterations": 5},
            "gateway": {"host": "127.0.0.1", "port": 8092},
        },
    )
    monkeypatch.setattr(cfg_mod, "global_config", fake)
    yield fake


@pytest.fixture(autouse=True)
def _clean_skill_registry():
    """每个测试后清空 skill 注册表，避免跨测试污染。"""
    yield
    from junjun_skills import registry
    registry.clear()
