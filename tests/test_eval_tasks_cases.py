"""golden_tasks.jsonl case 文件守卫（0 token，不跑 LLM）：

- 每条可解析、id 唯一
- 启用 case 引用的工具名必须存在于注册表（笔误 = 永远 FAIL 的假评测）
- expect 字段组合合法（submit_rejected 不与其他断言混用；approval 值域）
- 占位 case（enabled:false）豁免工具名校验（Phase 2 工具未上线）
"""

import json
from pathlib import Path

import pytest

CASES_FILE = Path(__file__).resolve().parent / "eval" / "golden_tasks.jsonl"


def _cases():
    return [json.loads(l) for l in
            CASES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def tool_names():
    from junjun_skills.registry import load_builtin, get_tools
    from junjun_skills.plugin_loader import load_plugins
    load_builtin()
    load_plugins()
    return {t.name for t in get_tools()} | {"llm_synthesize"}


class TestCaseFile:
    def test_parses_and_unique_ids(self):
        cases = _cases()
        assert len(cases) >= 30, "30 条草稿基准"
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids)), f"id 重复: {ids}"

    def test_every_case_has_input_and_expect(self):
        for c in _cases():
            assert c.get("input"), f"{c['id']} 缺 input"
            assert c.get("expect"), f"{c['id']} 缺 expect"

    def test_enabled_tool_names_exist(self, tool_names):
        for c in _cases():
            if not c.get("enabled", True):
                continue
            exp = c["expect"]
            refs = []
            for spec in exp.get("must_use", []):
                refs.extend(spec.split("|"))
            refs.extend(exp.get("must_not_use", []))
            refs.extend(exp.get("order", []))
            refs.extend((exp.get("step_action_status") or {}).keys())
            refs.extend((c.get("stub") or {}).keys())
            unknown = [r for r in refs if r not in tool_names]
            assert not unknown, f"{c['id']} 引用了不存在的工具: {unknown}"

    def test_submit_rejected_is_standalone(self):
        for c in _cases():
            exp = c["expect"]
            if exp.get("submit_rejected"):
                assert len(exp) == 1, f"{c['id']} submit_rejected 不应混搭其他断言"

    def test_approval_values(self):
        for c in _cases():
            ap = c.get("approval")
            assert ap in (None, "approve", "reject", "timeout"), f"{c['id']} approval={ap}"

    def test_stub_fail_times_int(self):
        for c in _cases():
            for tool, spec in (c.get("stub") or {}).items():
                if isinstance(spec, dict) and "fail_times" in spec:
                    assert isinstance(spec["fail_times"], int) and spec["fail_times"] > 0, \
                        f"{c['id']} stub.{tool}.fail_times 非法"
