"""L1 规则门单测。"""

import random

from junjun_agent.funnel import L1Config, L1Result, rule_gate, is_addressed


CFG = L1Config(talk_value=0.5, mentioned_bot_reply=True, nickname="君君", alias_names=("猫娘",))


def _gate(**kw):
    defaults = dict(
        text="随便聊聊", is_group=True, at_bot=False, is_self=False,
        silenced_until_call=False, cfg=CFG,
    )
    defaults.update(kw)
    return rule_gate(**defaults)


class TestSelfMessage:
    def test_self_message_dropped(self):
        assert _gate(is_self=True, at_bot=True) is L1Result.DROP


class TestAddressedBypass:
    def test_at_bot_goes_straight_to_agent(self):
        assert _gate(at_bot=True) is L1Result.TO_AGENT

    def test_nickname_call_goes_to_agent(self):
        assert _gate(text="君君几点了") is L1Result.TO_AGENT

    def test_alias_call_goes_to_agent(self):
        assert _gate(text="猫娘在吗") is L1Result.TO_AGENT

    def test_mentioned_bot_reply_false_no_bypass(self):
        cfg = L1Config(talk_value=1.0, mentioned_bot_reply=False, nickname="君君")
        r = rule_gate(text="x", is_group=True, at_bot=True, is_self=False,
                      silenced_until_call=False, cfg=cfg, rand=random.Random(1))
        assert r is L1Result.TO_GATE  # 不旁路，但 talk_value=1.0 进 gate


class TestTalkValue:
    def test_group_hit_probability(self):
        # Random(42).random() ≈ 0.639 > 0.5 -> DROP
        assert _gate(rand=random.Random(42)) is L1Result.DROP

    def test_group_pass_probability(self):
        cfg = L1Config(talk_value=0.99, nickname="君君")
        r = rule_gate(text="x", is_group=True, at_bot=False, is_self=False,
                      silenced_until_call=False, cfg=cfg, rand=random.Random(42))
        assert r is L1Result.TO_GATE

    def test_talk_value_zero_always_drop(self):
        cfg = L1Config(talk_value=0.0, nickname="君君")
        for seed in range(10):
            r = rule_gate(text="x", is_group=True, at_bot=False, is_self=False,
                          silenced_until_call=False, cfg=cfg, rand=random.Random(seed))
            assert r is L1Result.DROP


class TestPrivate:
    def test_private_goes_to_gate(self):
        assert _gate(is_group=False) is L1Result.TO_GATE


class TestSilencedMode:
    def test_silenced_drops_normal_message(self):
        assert _gate(silenced_until_call=True) is L1Result.DROP

    def test_silenced_released_by_at(self):
        assert _gate(silenced_until_call=True, at_bot=True) is L1Result.TO_AGENT

    def test_silenced_released_by_nickname(self):
        assert _gate(silenced_until_call=True, text="君君出来") is L1Result.TO_AGENT


class TestIsAddressed:
    def test_at_flag(self):
        assert is_addressed("无关文本", CFG, at_bot=True)

    def test_plain_text_not_addressed(self):
        assert not is_addressed("今天天气不错", CFG, at_bot=False)
