"""golden dataset 完整性校验（离线，不调 LLM）——保证数据集本身不会烂掉。"""

import json
from pathlib import Path

CASES = Path(__file__).parent / "golden_cases.jsonl"

_REQUIRED = {"id", "scene", "input", "expect"}
_EXPECT_KEYS = {"must_call", "must_not_call", "silence", "reply_required",
                "must_contain", "must_not_contain"}


def _load():
    return [json.loads(l) for l in
            CASES.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_dataset_exists_and_parses():
    cases = _load()
    assert len(cases) >= 25, f"golden case 太少（{len(cases)}），评测没有统计意义"


def test_schema_valid():
    ids = set()
    for c in _load():
        assert _REQUIRED <= set(c), f"{c.get('id')}: 缺字段 {_REQUIRED - set(c)}"
        assert c["id"] not in ids, f"重复 id: {c['id']}"
        ids.add(c["id"])
        assert c["scene"] in ("group", "private"), c["id"]
        unknown = set(c["expect"]) - _EXPECT_KEYS
        assert not unknown, f"{c['id']}: 未知 expect 键 {unknown}"
        # 每条 case 至少要有一个可判定条件
        assert c["expect"], f"{c['id']}: expect 为空，无法判定"


def test_expectations_reference_real_tools():
    """must_call/must_not_call 里的工具名必须在注册表里（防 dataset 漂移）。"""
    from junjun_skills.registry import load_builtin, get_tools
    from junjun_skills.plugin_loader import load_plugins
    load_builtin()
    load_plugins()
    real = {t.name for t in get_tools()}
    for c in _load():
        for spec in c["expect"].get("must_call", []):
            alts = spec.split("|")
            # 备选组至少一个真实存在即可（mcp_search 等 MCP 工具仅生产环境注册）
            assert any(a in real for a in alts), \
                f"{c['id']}: 备选组 {spec} 全部不存在"
        for name in c["expect"].get("must_not_call", []):
            assert name in real, f"{c['id']}: 工具 {name} 不存在"


def test_covers_core_incidents():
    """关键事故场景必须有对应 case（防数据集退化）。"""
    ids = {c["id"] for c in _load()}
    core = {"subscribe-watch-up", "draw-compound-remind", "task-query-no-task",
            "addressed-must-reply", "draw-nsfw-private", "impossible-honesty"}
    assert core <= ids, f"缺少核心事故 case: {core - ids}"
