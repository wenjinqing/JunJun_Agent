# Langfuse 启动排障（2026-07-16 诊断）

## 结论：不是 Langfuse 的问题，是 Docker Desktop 起不来

诊断链路：`docker ps` 报管道不存在 → 启动 Docker Desktop 后报 500 → 查 backend 日志找到根因：

```
engine linux/wsl failed to start: ... WSL2 is not supported with your current
machine configuration. HCS_E_HYPERV_NOT_INSTALLED
```

机器状态核对：
- Windows 11 **家庭版**
- BIOS 虚拟化**已开启**（VirtualizationFirmwareEnabled = True）✅
- Windows 虚拟机监控程序**未运行**（HypervisorPresent = False）❌
- WSL 程序本体已装（2.6.3.0）但**没有任何发行版**，且缺「虚拟机平台」组件，导致 Docker 的 WSL2 后端无法创建 VM
- 附带发现：`wsl --update` 走微软商店通道被 403（网络拦截），但不是主因

## 修复步骤（需要你本人操作，约 5 分钟 + 一次重启）

1. **管理员 PowerShell**（Win+X → 终端(管理员)）执行：
   ```powershell
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   ```
2. **重启电脑**
3. 重启后管理员 PowerShell：`wsl --update`
   - 若仍 403：浏览器直接下载离线包安装 https://github.com/microsoft/WSL/releases （最新 wsl.x.x.x.x.x64.msi）
4. 启动 Docker Desktop，等右下角鲸鱼图标变绿
5. 启动 Langfuse：
   ```
   cd E:\MaiM
   docker compose -f langfuse/docker-compose.yml up -d
   ```
6. 浏览器开 http://localhost:3000 → 注册账号 → 建组织/项目 → Settings 里拿 `pk-lf-...` / `sk-lf-...`
7. 填入 `E:\JunJun_Agent\.env`：
   ```
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-xxx
   LANGFUSE_SECRET_KEY=sk-lf-xxx
   ```
8. 验证：`.venv\Scripts\python.exe scripts\smoke_agent.py`，Langfuse 后台应出现 trace

## 备注

- 家庭版不支持完整 Hyper-V，但 WSL2 只需要「虚拟机平台」组件，家庭版可用
- Langfuse 未启动不影响君君运行（tracing 自动降级），只影响可观测验收项
