"""ai_draw 提示词工作室测试：写手+评审协作、降级链、VLM 验收重画。"""

import pytest

import junjun_skills.plugins.ai_draw.prompt_studio as studio
import junjun_skills.plugins.ai_draw.tools as draw_tools


class _FakeModel:
    """按调用顺序返回预设文本的 stub 模型。"""
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def ainvoke(self, msgs, config=None):
        self.calls.append(msgs)
        class R:
            content = self._outputs.pop(0) if self._outputs else ""
        return R()


def _patch_models(monkeypatch, utils_out, small_out):
    import junjun_llm
    models = {"utils": _FakeModel(utils_out), "utils_small": _FakeModel(small_out)}
    monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot: models[slot])
    return models


class TestCraftPrompt:
    @pytest.mark.asyncio
    async def test_writer_then_critic(self, monkeypatch):
        models = _patch_models(monkeypatch, ["写手稿"], ["评审修订稿"])
        monkeypatch.setattr(studio, "_cfg", lambda: {"prompt_critic": True})
        out = await studio.craft_prompt("画一只猫", "zimage")
        assert out == "评审修订稿"
        assert len(models["utils"].calls) == 1 and len(models["utils_small"].calls) == 1
        # 评审输入同时含原始需求与写手稿
        critic_input = models["utils_small"].calls[0][0].content
        assert "画一只猫" in critic_input and "写手稿" in critic_input

    @pytest.mark.asyncio
    async def test_critic_off_skips_review(self, monkeypatch):
        models = _patch_models(monkeypatch, ["写手稿"], ["评审修订稿"])
        monkeypatch.setattr(studio, "_cfg", lambda: {"prompt_critic": False})
        out = await studio.craft_prompt("画一只猫", "zimage")
        assert out == "写手稿"
        assert len(models["utils_small"].calls) == 0

    @pytest.mark.asyncio
    async def test_critic_failure_falls_back_to_draft(self, monkeypatch):
        import junjun_llm
        class _Boom:
            async def ainvoke(self, *a, **kw):
                raise RuntimeError("api down")
        monkeypatch.setattr(junjun_llm, "get_chat_model",
                            lambda slot: _FakeModel(["写手稿"]) if slot == "utils" else _Boom())
        monkeypatch.setattr(studio, "_cfg", lambda: {"prompt_critic": True})
        assert await studio.craft_prompt("画一只猫", "qwen") == "写手稿"

    @pytest.mark.asyncio
    async def test_anime_family_not_handled(self):
        assert await studio.craft_prompt("猫娘", "anime") == ""


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_zimage_uses_studio(self, monkeypatch):
        """zimage 家族走工作室（中文稿），不走旧英文扩写。"""
        monkeypatch.setenv("MODELSCOPE_API_KEY", "sk-test")
        monkeypatch.setattr(studio, "_cfg", lambda: {"prompt_critic": False})
        _patch_models(monkeypatch, ["中文结构化提示词"], [])
        seen = {}

        async def _fake_generate(prompt, model, negative=""):
            seen["prompt"] = prompt
            return "http://x/a.png"
        monkeypatch.setattr(draw_tools, "generate", _fake_generate)
        url, final = await draw_tools._draw_pipeline("一只橘猫在窗台晒太阳")
        assert url == "http://x/a.png"
        assert final == "中文结构化提示词"

    @pytest.mark.asyncio
    async def test_studio_failure_falls_back_to_legacy_expand(self, monkeypatch):
        monkeypatch.setenv("MODELSCOPE_API_KEY", "sk-test")

        async def _boom(prompt, family):
            raise RuntimeError("studio down")
        monkeypatch.setattr(studio, "craft_prompt", _boom)
        # craft_prompt 在 tools 里是函数内 import，打 studio 模块属性即可
        monkeypatch.setattr(studio, "_cfg", lambda: {})
        expanded = {}

        async def _legacy(prompt, *, anime=False):
            expanded["prompt"] = f"{prompt}，english detail"
            return expanded["prompt"]
        monkeypatch.setattr(draw_tools, "expand_prompt", _legacy)

        async def _fake_generate(prompt, model, negative=""):
            return "http://x/a.png"
        monkeypatch.setattr(draw_tools, "generate", _fake_generate)
        url, final = await draw_tools._draw_pipeline("一只橘猫")
        assert url and "english detail" in final

    @pytest.mark.asyncio
    async def test_vlm_review_retries_once(self, monkeypatch):
        """review_enable：首图验收不通过 -> 带意见重画一次；二次仍差也不再画。"""
        monkeypatch.setenv("MODELSCOPE_API_KEY", "sk-test")
        monkeypatch.setattr(studio, "_cfg",
                            lambda: {"prompt_critic": False, "review_enable": True})
        _patch_models(monkeypatch, ["稿一", "稿二"], [])
        reviews = []
        issues = ["主体画错了", "还是不对"]

        async def _review(url, origin):
            reviews.append(origin)
            return issues.pop(0)
        monkeypatch.setattr(studio, "review_image", _review)
        generated = []

        async def _fake_generate(prompt, model, negative=""):
            generated.append(prompt)
            return f"http://x/{len(generated)}.png"
        monkeypatch.setattr(draw_tools, "generate", _fake_generate)

        url, _ = await draw_tools._draw_pipeline("画一只橘猫")
        assert len(generated) == 2          # 重画一次且仅一次
        assert len(reviews) == 1            # 第二稿不再验收（_reviewed=True）
        assert url == "http://x/2.png"
        # 重画时写手拿到的需求带上了验收意见
        import junjun_llm
        writer_input = junjun_llm.get_chat_model("utils").calls[1][0].content
        assert "上一稿的问题" in writer_input and "主体画错了" in writer_input

    @pytest.mark.asyncio
    async def test_vlm_review_off_by_default(self, monkeypatch):
        monkeypatch.setenv("MODELSCOPE_API_KEY", "sk-test")
        monkeypatch.setattr(studio, "_cfg", lambda: {"prompt_critic": False})
        _patch_models(monkeypatch, ["稿"], [])
        called = []

        async def _review(url, origin):
            called.append(1)
            return "有问题"
        monkeypatch.setattr(studio, "review_image", _review)

        async def _fake_generate(prompt, model, negative=""):
            return "http://x/a.png"
        monkeypatch.setattr(draw_tools, "generate", _fake_generate)
        await draw_tools._draw_pipeline("画一只橘猫")
        assert not called
