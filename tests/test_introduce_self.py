"""introduce_self 自我介绍工具测试（0 token）：内容构成 + 泄露面负断言。

泄露面断言就是本工具的「误判回归」——披露面每加宽一次，这些负断言
必须仍然全绿（密钥/QQ 号/内网地址/路径/敏感插件名永不出现在输出里）。
"""

import junjun_skills.builtin.capability_skills as cap


def _fake_skills():
    return [
        {"name": "web_search", "plugin": "google_search", "enabled": True},
        {"name": "image_search", "plugin": "google_search", "enabled": True},
        {"name": "ai_draw", "plugin": "ai_draw", "enabled": True},
        {"name": "run_code", "plugin": "workspace", "enabled": True},
        {"name": "get_time", "plugin": "builtin", "enabled": True},
        {"name": "set_reminder", "plugin": "builtin", "enabled": True},
        # 敏感/内部件：即使启用也不许出现在简介里
        {"name": "netdisk_up", "plugin": "netdisk", "enabled": True},
        {"name": "topic_probe", "plugin": "topic_finder", "enabled": True},
        # 禁用件不出现
        {"name": "vrchat_move", "plugin": "vrchat_agent", "enabled": False},
    ]


class TestIntroduceSelf:
    def test_composition(self, monkeypatch):
        import junjun_skills.registry as reg
        monkeypatch.setattr(reg, "list_skills", lambda: _fake_skills())
        out = cap.introduce_self.invoke({})
        assert "我是君君" in out                       # 身份（conftest 假配置昵称）
        assert "联网搜索" in out                        # 分类映射命中
        assert "AI 画图" in out
        assert "工作区" in out
        assert "设提醒" in out                          # 内置基本功
        assert "技术栈" in out and "Python" in out
        # 架构是介绍重点（2026-08-15 用户拍板：型号不对外，讲架构与功能）
        assert "LangChain" in out and "LangGraph" in out
        assert "Docker" in out
        assert "get_capabilities" in out                # 指向完整清单

    def test_privacy_negative_assertions(self, monkeypatch):
        """泄露面：密钥/QQ/内网/路径/敏感插件名/接入平台名/模型型号永远不得出现。"""
        import junjun_skills.registry as reg
        monkeypatch.setattr(reg, "list_skills", lambda: _fake_skills())
        out = cap.introduce_self.invoke({})
        for bad in ("sk-", "http", "127.0.0.1", "localhost", "F:\\", "C:\\",
                    "netdisk", "topic_finder", "vrchat", "ADMIN_QQ",
                    "AI Ping", "DeepSeek", "混元"):  # 型号不公开（用户拍板）；混元=纯幻觉
            assert bad not in out, f"简介泄露了「{bad}」"

    def test_unknown_plugin_invisible(self, monkeypatch):
        """未登记的插件名不进简介（宁漏勿错——新插件要进简介先加映射表）。"""
        import junjun_skills.registry as reg
        monkeypatch.setattr(reg, "list_skills", lambda: [
            {"name": "mystery_tool", "plugin": "mystery_new", "enabled": True},
        ])
        out = cap.introduce_self.invoke({})
        assert "mystery" not in out
