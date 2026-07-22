"""插件加载器：启动扫描 junjun_skills/plugins/，按 _manifest.json 静态加载。

- manifest: {"name", "version", "module", "tools_attr", "available_for"}
- 依赖探测失败禁用该插件 WARN 不崩启动
- available_for: 会话白名单（chat_id 列表；空=全会话）
"""

import importlib
import json
from pathlib import Path

from junjun_core.observability import get_logger

logger = get_logger("skills.plugins")

PLUGINS_DIR = Path(__file__).resolve().parent / "plugins"


def load_plugins() -> int:
    """扫描并注册全部插件工具。返回注册数。"""
    from junjun_skills.registry import register
    if not PLUGINS_DIR.exists():
        return 0
    count = 0
    for manifest_path in sorted(PLUGINS_DIR.glob("*/_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            name = manifest["name"]
            module = importlib.import_module(manifest["module"])

            # 依赖探测（可选钩子）
            probe = getattr(module, "probe_available", None)
            if probe is not None and not probe():
                logger.warning(f"插件 [{name}] 依赖探测失败，禁用")
                continue

            tools = getattr(module, manifest.get("tools_attr", "TOOLS"))
            whitelist = set(manifest.get("available_for", []))
            gate = (lambda wl: (lambda session: not wl or session.chat_id in wl))(whitelist)
            # 拦截器在 import 时已注册（decorator side-effect），计数纳入日志
            from junjun_agent.interceptors import list_interceptors
            before_interceptors = len(list_interceptors())

            for t in tools:
                register(t, available_for=gate if whitelist else None, plugin=name,
                         admin_only=bool(manifest.get("admin_only", False)))
                count += 1
            interceptors_added = len(list_interceptors()) - before_interceptors
            suffix = f"（白名单 {len(whitelist)} 会话）" if whitelist else ""
            suffix += "（admin 门）" if manifest.get("admin_only") else ""
            if interceptors_added:
                suffix += f" + {interceptors_added} 拦截器"
            logger.info(f"插件 [{name}] v{manifest.get('version', '?')} 已加载 "
                        f"{len(tools)} 个工具{suffix}")
        except Exception as e:
            logger.warning(f"插件加载失败 [{manifest_path.parent.name}]（跳过）: {type(e).__name__}: {e}")
    return count
