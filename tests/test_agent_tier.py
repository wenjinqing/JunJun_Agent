"""双腿路由测试（2026-08-11 token 优化 P2）：纯闲聊轮走 agent_light 轻腿。

判 light 是「加宽命中面」——按仓库纪律，误判回归断言是主体：
任何带工具意图/事实需求/媒体/长文/管理员身份的输入都必须判 full（强链），
只有纯闲聊短句才允许判 light。
"""

import pytest

from junjun_agent.router import agent_tier


@pytest.fixture
def routing_on(_fake_bot_config):
    _fake_bot_config.raw["agent"] = {"complexity_routing": True}
    return _fake_bot_config


class TestConfigGate:
    def test_default_off_always_full(self):
        """开关默认关：再像闲聊也判 full（回到现状腿）。"""
        assert agent_tier("在吗") == "full"
        assert agent_tier("哈哈哈哈笑死") == "full"

    def test_on_simple_chat_light(self, routing_on):
        for t in ("在吗", "哈哈哈哈笑死", "早啊", "你在干嘛", "晚安啦",
                  "君君贴贴", "我也觉得", "确实", "好无聊啊", "你吃了吗"):
            assert agent_tier(t) == "light", t


class TestMisjudgmentRegression:
    """误判回归：以下输入一律 full——错判 light 会亏工具可靠性/人格。"""

    def test_tool_intents_full(self, routing_on):
        for t in ("帮我定个明早8点的闹钟", "记得提醒我开会", "订阅这个UP主",
                  "盯着这个作者更新了告诉我", "取消订阅", "帮我画一只猫",
                  "画张涩图", "发个语音给我", "唱首歌", "帮我调研一下这个话题",
                  "搜一下绝区零配队", "帮我查一下快递", "查一下明天天气怎么样"):
            assert agent_tier(t) == "full", t

    def test_fact_words_full(self, routing_on):
        for t in ("发布会什么时候开", "有什么最新消息", "今天天气如何",
                  "现在几点了", "这个多少钱"):
            assert agent_tier(t) == "full", t

    def test_media_and_links_full(self, routing_on):
        assert agent_tier("看看这个", has_media=True) == "full"
        assert agent_tier("这个视频讲了啥 https://b23.tv/abc") == "full"
        assert agent_tier("帮我看看 www.example.com") == "full"

    def test_long_or_complex_full(self, routing_on):
        assert agent_tier("君" * 51) == "full"  # 超轻腿长上限
        assert agent_tier("") == "full"
        assert agent_tier("帮我查一下资料然后写成报告发给我") == "full"  # 多步
        assert agent_tier("每天早上给我发天气预报") == "full"  # 时程承诺

    def test_admin_privileged_full(self, routing_on, monkeypatch):
        """管理员特权态（管理员+@bot）的拜托一律强链。"""
        from junjun_core.security import set_caller
        monkeypatch.setenv("ADMIN_QQ", "99999")
        set_caller("99999", at_bot=True, is_group=True, nickname="管理员")
        try:
            assert agent_tier("在吗") == "full"
        finally:
            set_caller("", at_bot=False, is_group=True, nickname="")


class TestPrivateDefaultsFull:
    """私聊默认不走轻腿（2026-08-16 生产实锤：私聊轻腿连续三轮承诺画图
    零调用，「我不相信你了」——工具明明钉在工具集里，弱模型选择先哄
    不做事）。私聊低频高价值，省不下几个 token，质量优先。"""

    def test_private_chat_always_full_by_default(self, routing_on):
        for t in ("在吗", "摩西摩西，亲爱的", "我不相信你了", "晚安啦"):
            assert agent_tier(t, is_group=False) == "full", t

    def test_private_light_requires_explicit_opt_in(self, _fake_bot_config):
        """显式打开 complexity_routing_private 才放行私聊轻腿。"""
        _fake_bot_config.raw["agent"] = {"complexity_routing": True,
                                         "complexity_routing_private": True}
        assert agent_tier("在吗", is_group=False) == "light"
        # 工具意图私聊照样强链（新开关只放行纯闲聊）
        assert agent_tier("帮我画一只猫", is_group=False) == "full"

    def test_group_light_unchanged(self, routing_on):
        """群聊轻腿行为不变（新开关只管私聊；缺省 is_group=True 保持旧语义）。"""
        assert agent_tier("在吗", is_group=True) == "light"
        assert agent_tier("在吗") == "light"


class TestLightModelFallback:
    def test_slot_unconfigured_falls_back(self, _fake_bot_config, monkeypatch):
        """agent_light 槽未配置/加载失败 -> None（调用方回落强链）。"""
        import junjun_agent.agent as agent_mod

        class _Session:
            chat_id = "qq:1:group"
            is_group = True

        def _boom(task):
            raise ValueError("任务槽未配置")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        a = agent_mod.JunJunAgent.__new__(agent_mod.JunJunAgent)
        a.session = _Session()
        a._light_model = None
        assert a._get_light_model() is None

    def test_slot_configured_caches(self, monkeypatch):
        import junjun_agent.agent as agent_mod

        class _Session:
            chat_id = "qq:1:group"
            is_group = True

        sentinel = object()
        calls = []

        def _fake_get(task):
            calls.append(task)
            return sentinel

        monkeypatch.setattr("junjun_llm.get_chat_model", _fake_get)
        a = agent_mod.JunJunAgent.__new__(agent_mod.JunJunAgent)
        a.session = _Session()
        a._light_model = None
        assert a._get_light_model() is sentinel
        assert a._get_light_model() is sentinel  # 第二次走缓存
        assert calls == ["agent_light"]
