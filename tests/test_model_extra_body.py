"""extra_body 透传测试（2026-08-09 换模）：
部分厂商的推理模型默认开思考（实测一句 hi 白烧 ~180 计费 reasoning tokens，
延迟 1.0s->3.8s），高频闲聊槽纯浪费 -> 条目级 extra_body
（有厂商认 thinking={"type":"disabled"}，不认 enable_thinking=false）
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
model_env = "SF_CHAT_MODEL"
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
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.provider-a.example/v1")
        monkeypatch.setenv("SF_CHAT_MODEL", "example-org/MODEL-X")
        monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-x")
        monkeypatch.setenv("DS_BASE_URL", "https://api.provider-b.example")
        monkeypatch.setenv("DS_MODEL", "demo-model")
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
        """号池条目：每条腿都带 extra_body（号池链的实际形态）。"""
        toml = _write(tmp_path, """
[[task.agent.models]]
base_url_env = "SF_LLM_BASE_URL"
model_env = "SF_CHAT_MODEL"
api_key_env = "SF_POOL"
extra_body = { thinking = { type = "disabled" } }
""")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.provider-a.example/v1")
        monkeypatch.setenv("SF_CHAT_MODEL", "example-org/MODEL-X")
        monkeypatch.setattr("junjun_llm.key_pool.sf_pool.healthy_keys",
                            lambda: ["k1", "k2"])
        models.reset_slots()
        specs = models._load_slots()["agent"].specs
        assert len(specs) == 2
        assert all(s.extra_body == {"thinking": {"type": "disabled"}} for s in specs)
        models.reset_slots()


class TestSlotTimeoutRetries:
    """槽级 timeout/max_retries 透传（2026-08-13 生产实锤：思考型 thinker 槽
    默认 60s×4 重试，规划两条腿 4×60s 全超时烧穿——思考槽必须能放宽单跳、
    减重试让给下一条腿）。"""

    def test_slot_timeout_applies(self, tmp_path, monkeypatch):
        toml = _write(tmp_path, """
[task.thinker]
temperature = 0.3
timeout = 180
max_retries = 1
[[task.thinker.models]]
base_url_env = "DS_BASE_URL"
model_env = "DS_MODEL"
api_key_env = "DEEPSEEK_API_KEY"
""")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("DS_BASE_URL", "https://api.provider-b.example")
        monkeypatch.setenv("DS_MODEL", "demo-model")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        models.reset_slots()
        spec = models._load_slots()["thinker"].specs[0]
        assert spec.timeout == 180.0 and spec.max_retries == 1
        chat = models._build_chat(spec)
        assert chat.request_timeout == 180.0 and chat.max_retries == 1
        models.reset_slots()

    def test_default_timeout_unchanged(self):
        """不配的槽维持 60s×3 旧行为（闲聊槽的防挂保护不动）。"""
        spec = models.ModelSpec(base_url="https://x/v1", model="m", api_key="k")
        chat = models._build_chat(spec)
        assert chat.request_timeout == 60.0 and chat.max_retries == 3

    def test_entry_level_overrides_slot(self, tmp_path, monkeypatch):
        """条目级覆盖槽级（同 temperature/max_tokens 的合并语义）。"""
        toml = _write(tmp_path, """
[task.thinker]
timeout = 180
[[task.thinker.models]]
base_url_env = "DS_BASE_URL"
model_env = "DS_MODEL"
api_key_env = "DEEPSEEK_API_KEY"
timeout = 30
""")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("DS_BASE_URL", "https://api.provider-b.example")
        monkeypatch.setenv("DS_MODEL", "demo-model")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        models.reset_slots()
        assert models._load_slots()["thinker"].specs[0].timeout == 30.0
        models.reset_slots()
