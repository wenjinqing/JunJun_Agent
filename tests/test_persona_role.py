"""persona 角色组装测试：设定卡 + 示例集拼接（自设形态）。

2026-08-04 起 behavior_examples 从「整体替换 personality」改为「拼接」：
设定卡给「你是谁」，示例集给「说话长什么样」。
"""

import junjun_agent.persona as persona


class TestRolePersona:
    def test_examples_concatenated_after_personality(self):
        p = {"personality": "你是君君。", "behavior_examples": "被夸→「才没有」"}
        role = persona._role_persona(p, "君君")
        assert "你是君君。" in role          # 设定卡保留
        assert "被夸→「才没有」" in role      # 示例集拼在后面
        assert role.index("你是君君。") < role.index("被夸→「才没有」")
        assert "不要照抄原句" in role         # 防复读标注

    def test_no_examples_falls_back_to_personality(self):
        p = {"personality": "你是君君。"}
        assert persona._role_persona(p, "君君") == "你是君君。"

    def test_empty_personality_defaults(self):
        assert persona._role_persona({}, "君君") == "你是君君。"

    def test_blank_examples_treated_as_absent(self):
        p = {"personality": "你是君君。", "behavior_examples": "   "}
        assert persona._role_persona(p, "君君") == "你是君君。"


class TestInterruptPhrasesClean:
    """打断复读固定文案不得含已知污染源口头禅（2026-08-04 实锤：
    「略略略~」「杂鱼们就会这一句？」由 repeat.py 固定文案学进人设）。"""

    def test_no_known_toxic_catchphrases(self):
        from junjun_agent.loop.repeat import _INTERRUPT_PHRASES
        joined = "".join(_INTERRUPT_PHRASES)
        assert "略略略" not in joined
        assert "杂鱼" not in joined
