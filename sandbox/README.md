# junjun-sandbox：run_code 沙箱服务

君君「跑代码」能力的执行后端。FastAPI 编排层 + 一次性 docker 容器，
安全边界在容器不在本服务（见下「安全模型」）。

## 一次性部署

```bash
# 1. 构建镜像（首次 / Dockerfile 变更后）
docker build -t junjun-sandbox:latest sandbox/

# 2. 启动服务（先于 bot 启动；bot 的 run_code 工具运行时调它）
uv run uvicorn sandbox.server:app --host 127.0.0.1 --port 8100

# 3. 自检
curl http://127.0.0.1:8100/health     # {"docker":true,"image":"junjun-sandbox:latest",...}
curl -X POST http://127.0.0.1:8100/run -H "Content-Type: application/json" \
     -d '{"code":"print(1+1)","workdir":"smoke"}'
```

bot 侧配置（`config/bot_config.toml`）：

```toml
[sandbox]
base_url = "http://127.0.0.1:8100"   # 默认值，同机部署可不写
```

## 安全模型

隔离全部在容器，本服务只做编排：

| 面 | 措施 |
|---|---|
| 挂载 | 唯一挂载 `data/workspace/<会话子目录>` → `/workspace`；根 fs `--read-only`；`/tmp` 是 noexec tmpfs |
| 网络 | `--network=none`，容器出入全断（沙箱里的 requests 只能失败） |
| 用户 | 非 root（uid 10001 sandbox）；资源 `--memory=2g --cpus=2`；30s 硬杀 |

辅助防线（挡手滑，不是安全边界）：插件侧 ast 静态预检（禁 os/sys/subprocess
等）+ 门禁（管理员直跑，非管理员走内核人审）。

Linux 宿主注意：`data/workspace` 需对容器内 uid 10001 可写
（`chmod -R 777 data/workspace` 或 ACL 指定 10001）。Windows / Docker Desktop 无此问题。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| SANDBOX_WORKSPACE_ROOT | data/workspace | 工作区根（须与 bot 的 workspace 插件同一目录） |
| SANDBOX_IMAGE | junjun-sandbox:latest | 执行镜像名 |

## 运维

- 日志即容器履历：每次运行容器名 `jj-sandbox-<uuid8>`，`--rm` 自动销毁
- 镜像更新（加包）：改 Dockerfile 重新 build，重启本服务即可，bot 不用动
- 服务挂了的表象：run_code 工具返回「沙箱服务不可达」——bot 其余功能不受影响
