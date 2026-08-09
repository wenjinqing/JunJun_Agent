"""extra_body 透传测试（2026-08-09 ***REMOVED*** 换模）：
硅基推理模型默认开思考（实测一句 hi 白烧 ~180 计费 reasoning tokens，
延迟 1.0s->3.8s），高频闲聊槽纯浪费 -> 条目级 extra_body
（GLM 认 thinking={"type":"disabled"}，不认 enable_thinking=false）
必须能从 toml 一路透传到 ChatOpenAI。
"""

import junjun_llm.models as models


def _write(tmp_path, body: str):
    toml = tmp_path / "model_config.toml"
    toml.write_text(body, encoding="utf-8")
    return toml


_STATIC_TOML = """
[task.agent]
temperature = 0.6
max_tokens = 2048
[[task.agent.models]]
base_url_env = "SF_LLM_BASE_URL"
model_env = "SF_GLM_MODEL"
api_key_env = "SILICONFLOW_API_KEY"
extra_body = { thinking = { type = "disabled" } }
[[task.agent.models]]
base_url_env = "DS_BASE_URL"
model_env = "DS_MODEL"
api_key_env = "DEEPSEEK_API_KEY"
"""


class TestExtraBody:
    def test_static_entry_carries_extra_body(self, tmp_path, monkeypatch):
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", _write(tmp_path, _STATIC_TOML))
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        monkeypatch.setenv("SF_GLM_MODEL", "zai-org/***REMOVED***")
        monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-x")
        monkeypatch.setenv("DS_BASE_URL", "https://api.***REMOVED***.com")
        monkeypatch.setenv("DS_MODEL", "***REMOVED***")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        models.reset_slots()
        specs = models._load_slots()["agent"].specs
        assert specs[0].extra_body == {"thinking": {"type": "disabled"}}
        assert specs[1].extra_body == {}  # 无 extra_body 的条目不受污染
        models.reset_slots()

    def test_build_chat_passes_extra_body(self):
        spec = models.ModelSpec(base_url="https://x/v1", model="m", api_key="k",
                                extra_body={"thinking": {"type": "disabled"}})
        chat = models._build_chat(spec)
        assert chat.extra_body == {"thinking": {"type": "disabled"}}

    def test_build_chat_empty_extra_body_is_none(self):
        """空 extra_body 传 None（不往请求体塞空对象，行为与旧版完全一致）。"""
        spec = models.ModelSpec(base_url="https://x/v1", model="m", api_key="k")
        assert models._build_chat(spec).extra_body is None

    def test_pool_legs_all_carry_extra_body(self, tmp_path, monkeypatch):
        """号池条目：每条腿都带 extra_body（GLM 号池链的实际形态）。"""
        toml = _write(tmp_path, """
[[task.agent.models]]
base_url_env = "SF_LLM_BASE_URL"
model_env = "SF_GLM_MODEL"
api_key_env = "SF_POOL"
extra_body = { thinking = { type = "disabled" } }
""")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        monkeypatch.setenv("SF_GLM_MODEL", "zai-org/***REMOVED***")
        monkeypatch.setattr("junjun_llm.key_pool.sf_pool.healthy_keys",
                            lambda: ["k1", "k2"])
        models.reset_slots()
        specs = models._load_slots()["agent"].specs
        assert len(specs) == 2
        assert all(s.extra_body == {"thinking": {"type": "disabled"}} for s in specs)
        models.reset_slots()
