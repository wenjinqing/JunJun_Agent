---
name: restart-deploy
description: 重启/部署君君生产实例（NapCat 看门狗 + adapter + bot 三件套），改代码或配置后生效、排查"机器人没反应"。用户说"重启机器人/部署/上线了没/怎么没反应"时使用。
---

# 重启部署

## 三件套（顺序：NapCat → adapter → bot）

| 件 | 启动 | 作用 |
|---|---|---|
| NapCat | `uv run python scripts/napcat_watchdog.py` | 拉起 QQ+NapCat、掉线自动重启、日志转发 |
| adapter | `uv run python scripts/run_adapter.py` | OneBot WS client，收发消息 |
| bot | `uv run python scripts/run_junjun.py` | 网关 + Agent 主程序 |

- 看门狗重复启动会自动接管旧实例；**taskkill 会杀本机所有 QQ.exe**（本机 QQ 专供 bot）。
- 看门狗状态机（踩坑沉淀）：get_status 不通 + WebUI(6099) 通 = 在等扫码/验证码，
  **此时不会重启**，需在 NapCat 窗口完成人工验证；免扫码凭证失效时密码兜底文件
  `data/napcat_quick_password.txt`（gitignored）。

## 什么改动要重启什么

- `config/bot_config.toml` 改动 → 重启 **bot**（adapter 不用）。
- junjun_agent/junjun_skills/junjun_memory 等代码 → 重启 **bot**。
- junjun_adapter_napcat 代码 → 重启 **adapter**。
- `.env` 密钥 → 两者都重启。

## 验证上线

1. 看日志（logs/ 目录）：网关启动行、插件装载行（`插件 [xxx] 已加载 N 个工具`）。
2. 私聊 bot 一句 / 或调 WebUI 任务面板确认回路通。
3. 上线后进入观察期：对照 `docs/生产验证清单_2026-08-03.md` 逐项验收（见 prod-verify skill）。

## 排查"没反应"顺序

NapCat 在线？(get_status) → adapter WS 连着？ → bot 进程活着？ →
网关日志有没有 inbound → echo guard / do_not_reply 是不是拦了（日志搜 echo/沉默）。
