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


class TestPersonaBrief:
    """persona_brief：utils 单发调用的统一声口来源（2026-08-04 全面排查：
    提醒/观后感/主动消息/汇报等单发调用看不到主 prompt，没有速写就是通用 AI 腔）。"""

    def test_configured_brief_wins(self, monkeypatch):
        from junjun_core.config import get_global_config
        p = get_global_config().raw.setdefault("personality", {})
        monkeypatch.setitem(p, "persona_brief", "群里的猫娘学姐：从容温柔")
        assert persona.persona_brief() == "群里的猫娘学姐：从容温柔"

    def test_fallback_to_personality_first_line(self):
        # conftest 假配置 personality 首行是「你是君君，测试人设。」
        assert persona.persona_brief() == "你是君君，测试人设。"

    def test_empty_personality_neutral_default(self, monkeypatch):
        from junjun_core.config import get_global_config
        p = get_global_config().raw.setdefault("personality", {})
        monkeypatch.setitem(p, "personality", "")
        monkeypatch.setitem(p, "persona_brief", "")
        assert persona.persona_brief() == "中文口语短句，像跟熟人发微信"


class TestVoicePromptsCarryBrief:
    """所有 utils 单发 prompt 都必须带 persona_brief 注入点——
    防再出现「写死人设」或「无人设裸奔」（intention 曾硬编「毒舌猫娘老婆」）。"""

    def test_intention_gen_prompt(self):
        from junjun_agent.loop.intention import _GEN_PROMPT
        assert "{persona_brief}" in _GEN_PROMPT
        assert "毒舌" not in _GEN_PROMPT      # 写死的人设片段已清除
        assert "猫娘老婆" not in _GEN_PROMPT

    def test_all_voice_prompts(self):
        from junjun_agent.loop.reminder import _REMIND_PROMPT
        from junjun_agent.loop.perception_followup import _COMPOSE_PROMPT
        from junjun_agent.loop.proactive import _TOPIC_PROMPT
        from junjun_agent.loop.async_jobs import _LEAD_PROMPT
        for prompt in (_REMIND_PROMPT, _COMPOSE_PROMPT, _TOPIC_PROMPT, _LEAD_PROMPT):
            assert "{persona_brief}" in prompt


class TestTaskTemplatePools:
    """任务完成/占线模板直发绕过 echo guard——池子必须够大（防口头禅）。"""

    def test_pools_big_enough(self):
        from junjun_agent.tasks import _BUSY_TEMPLATES, _DONE_TEMPLATES
        for kind, pool in _DONE_TEMPLATES.items():
            assert pool == [] or len(pool) >= 6, f"{kind} 池子太小"
        assert len(_BUSY_TEMPLATES) >= 6
