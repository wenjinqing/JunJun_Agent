"""坏 case 规则检测（P8-1）：检测器单测 + 回归语料回放。

工作流闭环：scripts/export_bad_traces.py 从 Langfuse 捞候选
-> 人工标注挑真坏 case 固化进 tests/regression_corpus/cases.jsonl
-> 本测试回放语料，保证检测器本身与规则语义不退化。
"""

import json
from pathlib import Path

from junjun_agent.loop.bad_cases import detect_bad_case

_CORPUS = Path(__file__).parent / "regression_corpus" / "cases.jsonl"


class TestDetectors:
    def test_percept_denial(self):
        trace = {"input": "[图片] 这是啥", "output": "我看不到图片"}
        assert "percept_denial" in detect_bad_case(trace)

    def test_denial_without_percept_no_flag(self):
        """纯文字聊天说「看不到希望」不误报（无感知标记）。"""
        trace = {"input": "人生好难", "output": "看不到希望也要加油"}
        assert "percept_denial" not in detect_bad_case(trace)

    def test_tool_error_storm(self):
        trace = {"input": "", "output": "[TOOL_ERROR kind=网络] x [TOOL_ERROR kind=网络] y [TOOL_ERROR kind=网络] z"}
        assert "tool_error_storm" in detect_bad_case(trace)

    def test_intent_missed(self):
        trace = {"input": "明天提醒我开会", "output": "好哒"}
        assert "intent_missed:set_reminder" in detect_bad_case(trace)

    def test_intent_called_no_flag(self):
        trace = {"input": "明天提醒我开会",
                 "output": "调用了 set_reminder 成功"}
        assert not any(k.startswith("intent_missed") for k in detect_bad_case(trace))

    def test_dict_payloads(self):
        """input/output 是 dict/list 时也能检测（REST API 真实形状）。"""
        trace = {"input": {"messages": [{"content": "[图片] 看"}]},
                 "output": {"text": "看不到"}}
        assert "percept_denial" in detect_bad_case(trace)


class TestCorpusReplay:
    def test_corpus_cases_match_expectations(self):
        assert _CORPUS.exists(), "回归语料缺失"
        n = 0
        for line in _CORPUS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            got = detect_bad_case(case["trace"])
            assert got == case["expect"], \
                f"语料「{case['case']}」期望 {case['expect']} 实际 {got}"
            n += 1
        assert n >= 3, "语料至少 3 条才有回归意义"
