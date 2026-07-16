"""VRChat agent skill 插件：原 plugins/vrchat_agent 迁移为 LangChain @tool。

anya_client / pose_library 原样复用（无框架依赖）。
可用性：仅白名单会话启用（config.toml [vrchat] available_for）。
"""

from typing import Optional

from langchain_core.tools import tool

from junjun_core.observability import get_logger

logger = get_logger("skills.vrchat")


def _client():
    from junjun_skills.plugins.vrchat_agent.anya_client import get_client
    return get_client()


@tool
def vrchat_list_poses() -> str:
    """查询君君在 VRChat 虚拟形象里能做哪些动作。用户让君君跳舞/打招呼/问能做什么动作时使用。"""
    from junjun_skills.plugins.vrchat_agent.pose_library import POSE_LIBRARY
    lines = ["可用动作："]
    for name, (desc, _builder) in POSE_LIBRARY.items():
        lines.append(f"- {name}: {desc}")
    lines.append("- stop: 停止当前动作")
    return "\n".join(lines)


@tool
def vrchat_play_pose(pose_name: str, duration: Optional[float] = None) -> str:
    """让君君在 VRChat 里做一个动作。

    Args:
        pose_name: 动作名（wave/nod/bow/clap/jump/walk_forward/tpose/standing 等，见 vrchat_list_poses）
        duration: 动作时长秒数，可省略用默认
    """
    from junjun_skills.plugins.vrchat_agent.pose_library import POSE_LIBRARY
    from junjun_skills.plugins.vrchat_agent.anya_client import (
        standing_frame_state, tpose_frame_state,
    )
    if pose_name not in POSE_LIBRARY:
        return f"未知动作: {pose_name}。可用: {', '.join(POSE_LIBRARY.keys())}"
    desc, builder = POSE_LIBRARY[pose_name]
    client = _client()
    if builder is None:  # 静态姿势
        client.hold(tpose_frame_state() if pose_name == "tpose" else standing_frame_state())
        return f"君君现在保持 {pose_name}（{desc}）。"
    kwargs = {"duration": float(duration)} if duration is not None else {}
    frames = builder(**kwargs)
    client.play_sequence(frames, then_hold=standing_frame_state())
    actual = frames[-1][0] if frames else 0.0
    logger.info(f"VRChat 动作: {pose_name} 时长 {actual:.1f}s")
    return f"君君开始做 {pose_name}（{desc}），约 {actual:.1f} 秒。"


@tool
def vrchat_stop_motion() -> str:
    """让君君停止 VRChat 里正在做的动作。"""
    _client().stop_motion()
    return "君君已停止动作。"


@tool
def vrchat_status() -> str:
    """查询君君 VRChat 形象的动作状态（是否在动、当前动作）。"""
    return f"当前状态: {_client().current_description()}"


VRCHAT_TOOLS = [vrchat_list_poses, vrchat_play_pose, vrchat_stop_motion, vrchat_status]


def probe_available() -> bool:
    """依赖探测：AnyaDance 客户端可构造即认为可用。"""
    try:
        _client()
        return True
    except Exception as e:
        logger.warning(f"VRChat 插件依赖不可用，禁用: {e}")
        return False
