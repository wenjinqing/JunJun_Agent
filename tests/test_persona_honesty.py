"""persona 诚实锚回归测试（2026-08-04 实锤幻觉事件）：

昨晚生产实锤三类幻觉，system prompt 必须有对应约束：
1. 没调 TTS 却声称「语音马上就好」，还接受对不存在歌声的夸奖、编造「唱了两首」
2. 被拆穿后用「逗你玩的」嘴硬找补
3. 把 P 站作者链接说成「B 站 UP 主」、没查却给确定答案
"""

from junjun_agent.persona import build_system_prompt


def _prompt():
    return build_system_prompt(is_group=True, nickname="君君", latest_text="在吗")


class TestHonestyAnchors:
    def test_media_claim_requires_tool(self):
        p = _prompt()
        assert "没调工具或工具没成功" in p
        assert "不能接受别人对不存在作品的夸奖" in p

    def test_no_face_saving_when_caught(self):
        p = _prompt()
        assert "逗你玩的" in p  # 被拆穿照实认错，别拿这句找补
        assert "照实认错" in p

    def test_no_fabricated_facts(self):
        p = _prompt()
        assert "只能以工具真实返回为准" in p
        assert "臆造" in p

    def test_cannot_sing_boundary(self):
        p = _prompt()
        assert "不会唱歌" in p
        assert "play_music" in p  # 提供替代方案（放原曲）

    def test_search_rule_names_only_real_tools(self):
        """规则只许引用真实注册的工具（2026-08-07：mcp_search 是幻影工具，
        全仓无任何注册——prompt 教模型调不存在的工具等于教它编）。"""
        p = _prompt()
        assert "web_search" in p
        assert "mcp_search" not in p


class TestExistingAnchorsIntact:
    """既有防幻觉锚不因新增被挤掉。"""

    def test_subscription_receipt_rule(self):
        assert "拿到成功回执" in _prompt()

    def test_tool_error_map_rule(self):
        assert "[TOOL_ERROR kind=...]" in _prompt()


class TestSceneSpecializedR18Rule:
    """涩图规则按场景只给适用的一半（2026-08-06 实锤「分不清群聊私聊」：
    让弱模型自己组合「当前场景 + 双分支规则」不可靠）。"""

    def test_group_prompt_only_refusal_branch(self):
        p = build_system_prompt(is_group=True, nickname="君君", latest_text="在吗")
        assert "群里有人要涩图" in p
        assert "私聊里可以画" not in p      # 群场景不给授权分支，防误组合
        assert "绝对红线" in p              # 未成年红线任何场景都在

    def test_private_prompt_only_permission_branch(self):
        p = build_system_prompt(is_group=False, nickname="君君", latest_text="在吗")
        assert "私聊里可以画" in p
        assert "群里不行" not in p
        assert "绝对红线" in p


class TestAdminAnchorLast:
    """admin 安全段必须处于 system prompt 最后一块（2026-08-06 审查实锤：
    persona 重构后 <state> 动态块压到了安全段后面——记忆/情绪含用户输入
    衍生品，是注入攻击面，安全指令的近因位置不能被压过）。"""

    _MARKER = "【安全规则·最高优先级"

    def test_admin_after_state_legacy_path(self):
        from junjun_agent.persona import build_system_prompt
        p = build_system_prompt(is_group=True, nickname="君君",
                                latest_text="在吗", memory_block="【长期记忆】测试内容")
        assert self._MARKER in p
        assert "<state>" in p
        assert p.index(self._MARKER) > p.index("<state>"), \
            "安全段被动态块压到前面了"

    def test_admin_after_state_budget_path(self):
        from junjun_agent.agent import _apply_context_budget
        msgs, system_text, _ = _apply_context_budget(
            is_group=True, latest_text="在吗",
            mood_block="", memory_block="【长期记忆】测试",
            relation_block="", background="", latest_msg="在吗",
            addressed=True)
        assert self._MARKER in system_text
        assert "<state>" in system_text
        assert system_text.index(self._MARKER) > system_text.index("<state>")

    def test_core_blocks_no_admin(self):
        # build_prompt_blocks 的 core 不再夹带安全段（由组装方收尾追加）
        from junjun_agent.persona import build_prompt_blocks
        core, _ = build_prompt_blocks(is_group=True, latest_text="在吗")
        assert self._MARKER not in core
