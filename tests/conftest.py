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
            platform="qq", qq_account="10000001",
            nickname="君君", alias_names=["猫娘"],
        ),
        raw={
            "bot": {"qq_account": "10000001", "nickname": "君君"},
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
            "gateway": {"host": "127.0.0.1", "port": 8192},
            "keyword_reaction": {
                "keyword_rules": [
                    {"keywords": ["人机", "机器人", "ai", "AI"], "reaction": "俏皮承认自己是AI"},
                ],
            },
            "response_post_process": {"enable_response_post_process": True},
            "response_splitter": {
                "enable": True, "max_sentence_num": 5, "max_chars_per_message": 120,
                "enable_kaomoji_protection": False, "enable_overflow_return_all": True,
            },
            "chinese_typo": {
                "enable": True, "error_rate": 0.01, "min_freq": 9,
                "tone_error_rate": 0.1, "word_replace_rate": 0.006,
            },
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


@pytest.fixture(autouse=True)
def _isolate_tool_health_state(tmp_path, monkeypatch):
    """工具健康度/熔断/失败日志是模块级全局态，不隔离就是跨测试顺序依赖
    （2026-08-13 实锤：registry 包装幂等修正后真实记账生效，前文文件的失败
    把熔断攒开，后文文件同 chat 同工具的调用收到「熔断拦截」文本噎死断言）。
    顺手兜底硬约束：不打补丁的测试此前会真写生产 data/tool_health.json /
    tool_failures.jsonl（2026-08-06 同类事故）。"""
    from junjun_skills import breaker, health, patches
    monkeypatch.setattr(health, "_STATE_PATH", tmp_path / "tool_health.json")
    monkeypatch.setattr(patches, "_LOG_PATH", tmp_path / "tool_failures.jsonl")
    monkeypatch.setattr(patches, "_STATE_PATH", tmp_path / "patches_state.json")
    health._reset_for_test()
    breaker._failures.clear()
    yield
    health._reset_for_test()
    breaker._failures.clear()
