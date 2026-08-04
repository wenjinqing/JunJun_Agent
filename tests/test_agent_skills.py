"""Agent 技能手册（md skills）测试：加载/索引/工具/注入 prompt。

手册目录 junjun_skills/agent_skills/*.md 是真实资产——这里既测加载器逻辑，
也测资产本身的格式约束（每本必须有 name + when，frontmatter 之外必须有正文）。
"""

import pytest

from junjun_skills import skills_md


@pytest.fixture(autouse=True)
def _clear_cache():
    skills_md._cache = None
    yield
    skills_md._cache = None


class TestLoader:
    def test_real_skills_all_parse(self):
        skills = skills_md.load_skills()
        assert len(skills) >= 6, f"手册数量异常: {sorted(skills)}"
        for name, s in skills.items():
            assert s["when"], f"{name} 缺 when（索引会变成空话）"
            assert len(s["body"]) > 50, f"{name} 正文过短"
            assert "---" not in s["body"][:5], f"{name} frontmatter 没剥掉"

    def test_index_renders_name_and_when(self):
        idx = skills_md.skill_index()
        assert "use_skill" in idx
        for name, s in skills_md.load_skills().items():
            assert name in idx and s["when"] in idx

    def test_get_skill_and_missing(self):
        assert skills_md.get_skill("video-watching") is not None
        assert "watch_video" in skills_md.get_skill("video-watching")
        assert skills_md.get_skill("不存在的") is None

    def test_empty_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_md, "SKILLS_DIR", tmp_path)
        skills_md._cache = None
        assert skills_md.load_skills() == {}
        assert skills_md.skill_index() == ""


class TestUseSkillTool:
    def test_returns_body(self):
        from junjun_skills.builtin.skill_guide import use_skill
        out = use_skill.invoke({"name": "pixiv"})
        assert "pixiv_search_illusts" in out

    def test_unknown_lists_available(self):
        from junjun_skills.builtin.skill_guide import use_skill
        out = use_skill.invoke({"name": "xxx"})
        assert "video-watching" in out and "没有叫" in out

    def test_registered_in_builtin(self):
        from junjun_skills.registry import load_builtin, get_tools
        load_builtin()
        assert "use_skill" in {t.name for t in get_tools()}


class TestPromptInjection:
    def test_system_prompt_carries_skills_block(self, monkeypatch):
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={}))
        from junjun_agent.persona import build_system_prompt
        prompt = build_system_prompt(is_group=True, latest_text="帮我看看这个视频")
        assert "<skills>" in prompt and "video-watching" in prompt
        # 索引只放目录，不放全文（每轮 context 成本控制）
        assert "选哪条路" not in prompt
