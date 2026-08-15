"""persona 角色组装测试。

2026-08-15 通用 agent 转向瘦身（用户拍板「保留基本角色特征，不全删」）：
- _role_persona 只剩一句话速写（persona_brief 同源），整卡/示例集不再每轮注入
- 前缀缓存纪律：当前时间/keyword 命中/必回提示全部后置出核心段，
  核心段跨轮字节级稳定
"""

import junjun_agent.persona as persona


class TestRolePersona:
    def test_role_is_brief_with_nickname(self):
        # conftest 假配置 personality 首行「你是君君，测试人设。」
        assert persona._role_persona("君君") == "你是君君。你是君君，测试人设。"

    def test_full_card_not_injected(self, monkeypatch):
        """设定卡第二行起不进 role——瘦身的核心证据。"""
        from junjun_core.config import get_global_config
        p = get_global_config().raw.setdefault("personality", {})
        monkeypatch.setitem(p, "personality", "你是君君。\n傲娇，爱说杂鱼。")
        monkeypatch.setitem(p, "persona_brief", "")
        role = persona._role_persona("君君")
        assert "你是君君。" in role          # 速写（首行回退）在
        assert "杂鱼" not in role            # 整卡其余部分不注入

    def test_examples_never_injected(self, monkeypatch):
        """behavior_examples 不再进任何 prompt 路径（2026-08-15 瘦身）。"""
        from junjun_core.config import get_global_config
        p = get_global_config().raw.setdefault("personality", {})
        monkeypatch.setitem(p, "behavior_examples", "被夸→「才没有」")
        core, dynamic = persona.build_prompt_blocks(is_group=True, latest_text="在吗")
        assert "被夸→「才没有」" not in core
        assert not any(b["name"] == "examples" for b in dynamic)
        prompt = persona.build_system_prompt(is_group=True, latest_text="在吗")
        assert "被夸→「才没有」" not in prompt


class TestCachePrefixStability:
    """前缀缓存纪律（2026-08-15 用户要求：重复内容前置、变化内容后置）。"""

    @staticmethod
    def _freeze_time(monkeypatch, text):
        class _I:
            def strftime(self, fmt): return text

        class _DT:
            @staticmethod
            def now(): return _I()

        monkeypatch.setattr(persona, "datetime", _DT)

    def test_core_stable_across_minutes(self, monkeypatch):
        """分钟推进，核心段必须字节级不变；时间只许在 now 动态块里。"""
        self._freeze_time(monkeypatch, "2026-08-15 10:00 Saturday")
        core1, dyn1 = persona.build_prompt_blocks(is_group=True, latest_text="在吗")
        self._freeze_time(monkeypatch, "2026-08-15 10:01 Saturday")
        core2, dyn2 = persona.build_prompt_blocks(is_group=True, latest_text="在吗")
        assert core1 == core2, "时间变化打穿了核心段前缀"
        assert "当前时间" not in core1
        now1 = [b for b in dyn1 if b["name"] == "now"]
        now2 = [b for b in dyn2 if b["name"] == "now"]
        assert now1 and now2 and now1[0]["content"] != now2[0]["content"]
        assert now1[0]["required"] is True

    def test_reaction_hit_does_not_touch_core(self, monkeypatch):
        """keyword_reaction 命中与否，核心段字节级一致（命中走 reaction 动态块）。"""
        from junjun_core.config import get_global_config
        monkeypatch.setitem(get_global_config().raw, "keyword_reaction",
                            {"keyword_rules": [{"keywords": ["机器人"],
                                                "reaction": "被问是不是机器人"}]})
        core_hit, dyn_hit = persona.build_prompt_blocks(
            is_group=True, latest_text="你是不是机器人啊")
        core_miss, dyn_miss = persona.build_prompt_blocks(
            is_group=True, latest_text="在吗")
        # 核心段只允许差在「当前时间」上——冻结时间后必须完全一致
        import re
        strip_now = lambda s: re.sub(r"当前时间：[^\n<]*", "", s)
        assert strip_now(core_hit) == strip_now(core_miss)
        assert "机器人" not in core_hit
        r = [b for b in dyn_hit if b["name"] == "reaction"]
        assert r and "被问是不是机器人" in r[0]["content"]
        assert not any(b["name"] == "reaction" for b in dyn_miss)

    def test_legacy_prompt_time_after_rules(self):
        """非预算路径：当前时间在 <rules> 之后的 <state> 区。"""
        prompt = persona.build_system_prompt(is_group=True, latest_text="在吗")
        assert prompt.index("当前时间") > prompt.index("</rules>")

    def test_budget_path_addressed_in_latest_anchor(self):
        """必回提示并入最新消息锚点，不进 system 前缀（随消息翻转必穿缓存）。"""
        from junjun_agent.agent import _apply_context_budget
        msgs, sys_text, _ = _apply_context_budget(
            is_group=True, latest_text="在吗", mood_block="", memory_block="",
            relation_block="", background="「甲」: 在吗",
            latest_msg="「甲」: 君君在吗", addressed=True)
        assert "必须正面回应" not in sys_text
        last = msgs[-1]
        assert "你要回复的消息" in last.content and "必须正面回应" in last.content

    def test_budget_path_state_blocks_present(self):
        """预算路径：now/reaction 进 <state> 尾区，不被装配吞掉。"""
        from junjun_agent.agent import _apply_context_budget
        msgs, sys_text, _ = _apply_context_budget(
            is_group=True, latest_text="在吗", mood_block="心情：不错",
            memory_block="", relation_block="", background="",
            latest_msg="「甲」: 在吗", addressed=False)
        assert "当前时间" in sys_text
        assert sys_text.index("当前时间") > sys_text.index("</rules>")
        assert "心情：不错" in sys_text


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
