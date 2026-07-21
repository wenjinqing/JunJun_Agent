"""L2 语义门单测（mock LLM，不打真实 API）。"""

import pytest

from junjun_agent.funnel import GateDecision, llm_gate, parse_gate_output


class TestParse:
    def test_clean_json(self):
        assert parse_gate_output('{"decision": "reply", "reason": "被提问"}') is GateDecision.REPLY

    def test_json_in_code_fence(self):
        raw = '好的，我的判断是：\n```json\n{"decision": "no_reply", "reason": "闲聊"}\n```'
        assert parse_gate_output(raw) is GateDecision.NO_REPLY

    def test_until_call(self):
        assert parse_gate_output('{"decision": "no_reply_until_call"}') is GateDecision.NO_REPLY_UNTIL_CALL

    def test_garbage_defaults_no_reply(self):
        assert parse_gate_output("我觉得应该回复！") is GateDecision.NO_REPLY

    def test_invalid_value_defaults_no_reply(self):
        assert parse_gate_output('{"decision": "maybe"}') is GateDecision.NO_REPLY

    def test_empty_defaults_no_reply(self):
        assert parse_gate_output("") is GateDecision.NO_REPLY


class _FakeModel:
    def __init__(self, content="", raise_exc=False):
        self.content = content
        self.raise_exc = raise_exc

    async def ainvoke(self, messages, config=None):
        if self.raise_exc:
            raise ConnectionError("api down")

        class R:
            content = self.content
        return R()


@pytest.mark.asyncio
async def test_gate_reply_path():
    model = _FakeModel('{"decision": "reply", "reason": "提问"}')
    assert await llm_gate("A: 君君会做饭吗", "君君", model=model) is GateDecision.REPLY


@pytest.mark.asyncio
async def test_gate_api_failure_defaults_no_reply():
    model = _FakeModel(raise_exc=True)
    assert await llm_gate("A: hi", "君君", model=model) is GateDecision.NO_REPLY


@pytest.mark.asyncio
async def test_gate_api_failure_private_defaults_reply():
    """私聊 API 故障兜底 reply（对齐原 Brain 语义，不吞用户消息）。"""
    model = _FakeModel(raise_exc=True)
    assert await llm_gate("A: hi", "君君", model=model, is_group=False) is GateDecision.REPLY


class _CaptureModel:
    """捕获 prompt 的 fake 模型。"""

    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages, config=None):
        self.messages = messages

        class R:
            content = '{"decision": "reply"}'
        return R()


@pytest.mark.asyncio
async def test_gate_prompt_contains_command_rule():
    """指令类消息（设提醒等）即使没叫名字也应 reply——2026-07-21 误判修复。"""
    model = _CaptureModel()
    await llm_gate("A: 设置12点下班", "君君", model=model)
    system = model.messages[0].content
    assert "设提醒" in system and "即使没叫你名字" in system


@pytest.mark.asyncio
async def test_gate_scene_private_biases_reply():
    model = _CaptureModel()
    await llm_gate("A: hi", "君君", model=model, is_group=False)
    assert "私聊" in model.messages[0].content
    assert "拿不准时 -> reply" in model.messages[0].content


@pytest.mark.asyncio
async def test_gate_scene_group_default():
    model = _CaptureModel()
    await llm_gate("A: hi", "君君", model=model, is_group=True)
    assert "群聊" in model.messages[0].content
