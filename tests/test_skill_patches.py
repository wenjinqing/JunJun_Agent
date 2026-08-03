"""技能补丁（P8-2 经验回放）：失败日志聚合、复盘产出候选、管理员门控、
description 注入/回滚、溢出合并、调度节流、/补丁 命令。

门控语义（doc）：补丁必须带失败依据（source_case）才允许生效——无依据激活被拒。
"""

import json
import time
from types import SimpleNamespace

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m
from junjun_skills import patches as pm


@pytest.fixture
def env(monkeypatch, tmp_path):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"evolution": {"enable": True, "review_interval_days": 7,
                           "min_failures": 3, "max_patches_per_tool": 3}})
    monkeypatch.setattr(pm, "_LOG_PATH", tmp_path / "tool_failures.jsonl")
    monkeypatch.setattr(pm, "_STATE_PATH", tmp_path / "patches_state.json")
    db = SqliteDatabase(":memory:")
    with db.bind_ctx([m.SkillPatch]):
        db.create_tables([m.SkillPatch])
        yield monkeypatch
    cfg_mod.global_config = old


def _fail(tool="pixiv_search", n=3, kind="网络", error="timeout", age_h=1):
    ts = time.time() - age_h * 3600
    pm._LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pm._LOG_PATH.open("a", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps({"ts": ts, "tool": tool, "kind": kind,
                                "error": error}, ensure_ascii=False) + "\n")


def _patch(tool="pixiv_search", patch="带审核状态过滤", source_case="3 次超时",
           status="candidate", version=1):
    return m.SkillPatch.create(tool=tool, patch=patch, source_case=source_case,
                               status=status, version=version,
                               created_at=time.time(), updated_at=time.time())


class _ReviewModel:
    """假复盘模型：按 prompt 里的工具名回固定补丁。"""

    def __init__(self, patch="搜索要带审核状态过滤", lesson="连续超时"):
        self.patch, self.lesson = patch, lesson

    async def ainvoke(self, msgs, config=None):
        content = json.dumps({"patch": self.patch, "lesson": self.lesson},
                             ensure_ascii=False)
        return SimpleNamespace(content=content)


class _EmptyModel:
    async def ainvoke(self, msgs, config=None):
        return SimpleNamespace(content='{"patch": "", "lesson": "只是网络抖动"}')


def _fake_tool(name="pixiv_search", desc="搜索插画"):
    return SimpleNamespace(name=name, description=desc, metadata={})


class TestFailureLog:
    def test_log_and_aggregate(self, env):
        pm.log_failure("web_search", "网络", "timeout")
        pm.log_failure("web_search", "参数", "bad arg")
        pm.log_failure("pixiv_search", "网络", "403")
        agg = pm.aggregate_failures(days=7)
        assert len(agg["web_search"]) == 2
        assert agg["web_search"][0]["kind"] == "网络"
        assert len(agg["pixiv_search"]) == 1

    def test_aggregate_filters_old(self, env):
        _fail(age_h=24 * 10)  # 10 天前
        assert pm.aggregate_failures(days=7) == {}
        assert len(pm.aggregate_failures(days=14)["pixiv_search"]) == 3

    def test_log_never_raises(self, env, monkeypatch):
        monkeypatch.setattr(pm, "_LOG_PATH", pm._LOG_PATH.parent / "x\x00bad")
        pm.log_failure("t", "k", "e")  # 静默吞掉


class TestReview:
    @pytest.mark.asyncio
    async def test_review_creates_candidate(self, env):
        _fail(n=3)
        n = await pm.review(model=_ReviewModel())
        assert n == 1
        row = m.SkillPatch.get()
        assert row.tool == "pixiv_search" and row.status == "candidate"
        assert row.patch == "搜索要带审核状态过滤"
        assert row.source_case == "连续超时"

    @pytest.mark.asyncio
    async def test_below_min_failures_skipped(self, env):
        _fail(n=2)  # min_failures=3
        assert await pm.review(model=_ReviewModel()) == 0
        assert m.SkillPatch.select().count() == 0

    @pytest.mark.asyncio
    async def test_empty_patch_no_candidate(self, env):
        """只是抖动的失败提炼不出教训 -> 不造候选。"""
        _fail(n=5)
        assert await pm.review(model=_EmptyModel()) == 0
        assert m.SkillPatch.select().count() == 0

    @pytest.mark.asyncio
    async def test_dedupe_similar_existing(self, env):
        _patch(patch="搜索要带审核状态过滤", status="active")
        _fail(n=3)
        assert await pm.review(model=_ReviewModel(patch="搜索要带审核状态过滤")) == 0
        assert m.SkillPatch.select().count() == 1

    @pytest.mark.asyncio
    async def test_broken_model_output_tolerated(self, env):
        _fail(n=3)

        class _Bad:
            async def ainvoke(self, msgs, config=None):
                return SimpleNamespace(content="我觉得吧……没有 JSON")

        assert await pm.review(model=_Bad()) == 0


class TestActivateRollback:
    def test_activate_injects_description(self, env, monkeypatch):
        from junjun_skills import registry
        tool = _fake_tool()
        monkeypatch.setitem(registry._registry, "pixiv_search", tool)
        row = _patch(status="candidate")
        out = pm.activate(row.id)
        assert "已启用" in out
        assert tool.description == "搜索插画\n【经验补丁 v1】带审核状态过滤"
        assert tool.metadata["_orig_desc"] == "搜索插画"

    def test_activate_requires_source_case(self, env):
        row = _patch(source_case="")
        out = pm.activate(row.id)
        assert "不能启用" in out
        assert m.SkillPatch.get_by_id(row.id).status == "candidate"

    def test_activate_missing_id(self, env):
        assert "不存在" in pm.activate(999)

    def test_rollback_restores_description(self, env, monkeypatch):
        from junjun_skills import registry
        tool = _fake_tool()
        monkeypatch.setitem(registry._registry, "pixiv_search", tool)
        row = _patch(status="candidate")
        pm.activate(row.id)
        assert "【经验补丁" in tool.description
        out = pm.rollback(row.id)
        assert "已回滚" in out
        assert tool.description == "搜索插画"
        assert m.SkillPatch.get_by_id(row.id).status == "rolled_back"

    def test_register_applies_existing_patches(self, env, monkeypatch):
        """新注册的工具也要吃到已激活补丁（重启恢复链路）。"""
        from junjun_skills import registry
        _patch(status="active")
        tool = _fake_tool()
        monkeypatch.setitem(registry._registry, "pixiv_search", tool)
        pm.apply_to_registry("pixiv_search")
        assert tool.description.endswith("【经验补丁 v1】带审核状态过滤")

    def test_apply_idempotent(self, env, monkeypatch):
        from junjun_skills import registry
        tool = _fake_tool()
        monkeypatch.setitem(registry._registry, "pixiv_search", tool)
        _patch(status="active")
        pm.apply_to_registry("pixiv_search")
        pm.apply_to_registry("pixiv_search")
        assert tool.description.count("【经验补丁") == 1


class TestMergeOverflow:
    @pytest.mark.asyncio
    async def test_overflow_merges_to_one(self, env, monkeypatch):
        from junjun_skills import registry
        tool = _fake_tool()
        monkeypatch.setitem(registry._registry, "pixiv_search", tool)
        for i in range(4):  # max_patches_per_tool=3 -> 溢出
            _patch(patch=f"教训{i}", status="active", version=i + 1)

        class _Merge:
            async def ainvoke(self, msgs, config=None):
                return SimpleNamespace(content="合并后的总教训")

        await pm._merge_overflow("pixiv_search", model=_Merge())
        rows = list(m.SkillPatch.select().order_by(m.SkillPatch.version))
        assert [r.status for r in rows[:4]] == ["merged"] * 4
        merged = rows[4]
        assert merged.status == "active" and merged.version == 5
        assert merged.patch == "合并后的总教训"
        assert "合并" in merged.source_case
        # 注入链路只剩一条
        assert tool.description.count("【经验补丁") == 1
        assert "合并后的总教训" in tool.description

    @pytest.mark.asyncio
    async def test_within_limit_no_merge(self, env):
        for i in range(3):
            _patch(patch=f"教训{i}", status="active", version=i + 1)

        class _Never:
            async def ainvoke(self, msgs, config=None):  # pragma: no cover
                raise AssertionError("不该触发合并")

        await pm._merge_overflow("pixiv_search", model=_Never())
        assert m.SkillPatch.select().where(
            m.SkillPatch.status == "active").count() == 3


class TestTick:
    @pytest.mark.asyncio
    async def test_disabled_no_review(self, env):
        cfg_mod.global_config.raw["evolution"]["enable"] = False
        _fail(n=5)
        await pm.review_tick()
        assert m.SkillPatch.select().count() == 0

    @pytest.mark.asyncio
    async def test_interval_throttle(self, env):
        _fail(n=3)
        await pm.review_tick(model=_ReviewModel())
        assert m.SkillPatch.select().count() == 1
        # 立刻再来一轮：周期未到，不重复复盘
        _fail(n=3)
        await pm.review_tick(model=_ReviewModel())
        assert m.SkillPatch.select().count() == 1
        assert json.loads(pm._STATE_PATH.read_text(encoding="utf-8"))["last_review"] > 0


class TestCommands:
    @pytest.mark.asyncio
    async def test_patches_cmd_flow(self, env, monkeypatch):
        from junjun_skills import registry
        from junjun_skills.builtin.capability_skills import patches_cmd
        tool = _fake_tool()
        monkeypatch.setitem(registry._registry, "pixiv_search", tool)
        ctx = lambda a: SimpleNamespace(args=a)  # noqa: E731

        out = await patches_cmd(ctx(""))
        assert "还没有技能补丁" in out

        row = _patch()
        out = await patches_cmd(ctx("list"))
        assert f"#{row.id}" in out and "pixiv_search" in out

        out = await patches_cmd(ctx(f"启用 {row.id}"))
        assert "已启用" in out and "【经验补丁" in tool.description

        out = await patches_cmd(ctx(f"回滚 {row.id}"))
        assert "已回滚" in out and tool.description == "搜索插画"

        out = await patches_cmd(ctx("启用 abc"))
        assert "用法" in out

    @pytest.mark.asyncio
    async def test_patches_cmd_admin_only(self, env):
        """/补丁 在源码里以 admin_only=True 注册（框架层权限门）。

        全套件下别的插件测试 fixture 会 clear_commands() 清掉全局命令表，
        不能依赖导入时的注册残留——直接查源码里的注册调用。
        """
        import inspect
        import re
        from junjun_skills.builtin import capability_skills as cs
        src = inspect.getsource(cs)
        hit = re.search(r'register_command\(\s*"patches".*?\)\s*\nasync def patches_cmd',
                        src, re.S)
        assert hit, "/补丁 命令注册丢失"
        assert "admin_only=True" in hit.group(0), "/补丁 必须是管理员命令"
