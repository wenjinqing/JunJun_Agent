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
