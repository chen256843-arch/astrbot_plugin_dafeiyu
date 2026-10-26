# 部署：QQ 小号 + SnowLuma + AstrBot + DeepSeek + 大肥鱼

```
QQ 小号（snowluma 容器里的 NTQQ）
   │ ptrace 注入
SnowLuma ──反向 WebSocket──▶ AstrBot :6199（aiocqhttp 适配器）
                                │
                                ├─ DeepSeek 提供商（对话）
                                └─ 大肥鱼插件（数据在 astrbot-data/plugin_data）
```

> **已经有 AstrBot 和 SnowLuma 在跑？** 不要用下面的 compose 重建容器，否则 QQ 要重新扫码，原有人格和插件也可能丢失。只要现有链路满足「SnowLuma 反向 WS 连到 AstrBot」「AstrBot 里配好了对话模型」，直接跳到 [第 6 步](#6-安装大肥鱼插件)。

## 0. 服务器准备

- Ubuntu 22.04 / 24.04。QQ 本身是 Chromium，建议至少 **2 核 4 GB**；内存小就加 2–4 GB swap。
- 安装 Docker 和 compose 插件（[官方文档](https://docs.docker.com/engine/install/ubuntu/)）。
- 云厂商安全组 **只放行 22 端口**。下面所有面板都通过 SSH 隧道访问，不要对公网开放 6081 / 5099 / 6185。
- 服务器用 SSH 密钥登录，不要把 root 密码写进聊天记录、日志或仓库。

## 1. 启动容器

```bash
mkdir -p ~/dafeiyu-bot && cd ~/dafeiyu-bot
curl -fsSLO https://raw.githubusercontent.com/chen256843-arch/astrbot_plugin_dafeiyu/main/deploy/docker-compose.yml
docker compose up -d
```

[`deploy/docker-compose.yml`](../deploy/docker-compose.yml) 里所有管理端口都只绑定 `127.0.0.1`，SnowLuma 通过 Docker 内部网络连到 `astrbot:6199`，不需要暴露 OneBot 端口。

## 2. 在本地电脑打开 SSH 隧道

```bash
ssh -L 6081:127.0.0.1:6081 -L 5099:127.0.0.1:5099 -L 6185:127.0.0.1:6185 用户名@服务器IP
```

隧道开着的时候，本地浏览器访问 `localhost:6081` / `localhost:5099` / `localhost:6185` 就是服务器上的面板。

## 3. 登录 QQ 小号（SnowLuma）

1. 查远程桌面密码：
   ```bash
   docker logs snowluma 2>&1 | grep -E "远程桌面密码:|remote desktop password:" | tail -n 1
   ```
2. 浏览器打开 `http://localhost:6081/`，输入密码。QQ 会自动启动，用手机 QQ（小号）扫码登录。
3. 查 SnowLuma WebUI 初始密码，打开 `http://localhost:5099/` 登录：
   ```bash
   docker logs snowluma 2>&1 | grep -E "临时密码|initial credentials" | tail -n 1
   ```

第三方 QQ 客户端有封号风险，请只用小号。

## 4. 让 SnowLuma 连到 AstrBot（反向 WebSocket）

先生成一个随机 token，SnowLuma 和 AstrBot 两边填同一个：

```bash
openssl rand -hex 16
```

在 SnowLuma WebUI 的「连接配置」里为小号添加一个 **WebSocket 客户端**（反向 WS）：

| 字段 | 值 |
|---|---|
| url | `ws://astrbot:6199/ws` |
| role | `Universal` |
| accessToken | 上面生成的 token |
| messageFormat | `array` |

如果 WebUI 里没有这一项，也可以直接改 SnowLuma 数据目录下的 `config/onebot.json`（每个号可以用 `config/onebot_<QQ号>.json` 单独覆盖），在 `networks.wsClients` 里加：

```json
{
  "name": "astrbot",
  "enabled": true,
  "url": "ws://astrbot:6199/ws",
  "role": "Universal",
  "reconnectIntervalMs": 5000,
  "accessToken": "上面生成的 token",
  "messageFormat": "array",
  "reportSelfMessage": false
}
```

改完 `docker restart snowluma`。

## 5. 配置 AstrBot

1. 查 AstrBot 初始密码（用户名 `astrbot`），打开 `http://localhost:6185/` 登录，按提示改密码：
   ```bash
   docker logs astrbot 2>&1 | grep "Initial password"
   ```
2. **模型提供商** → 新增 DeepSeek，填 API Key，选一个对话模型。
3. **机器人 / 消息平台** → 新增「OneBot v11（aiocqhttp）」：反向 WebSocket 主机 `0.0.0.0`、端口 `6199`，Token 填第 4 步的 token。
4. 看是否连上：
   ```bash
   docker logs -f astrbot 2>&1 | grep -i aiocqhttp
   ```
5. 想要回复逐句分条发送，在 AstrBot 配置里开启「分段回复」。
6. 「主动回复」（长期记忆设置下的 active_reply）保持关闭，群聊是否插话交给大肥鱼，避免一条消息回两次。

## 6. 安装大肥鱼插件

AstrBot WebUI → 插件 → 安装插件，任选一种：

- 从链接安装：`https://github.com/chen256843-arch/astrbot_plugin_dafeiyu`
- 从文件安装：上传 [Releases](https://github.com/chen256843-arch/astrbot_plugin_dafeiyu/releases) 里的 zip
- 手动：
  ```bash
  git clone https://github.com/chen256843-arch/astrbot_plugin_dafeiyu ~/dafeiyu-bot/astrbot-data/plugins/astrbot_plugin_dafeiyu
  docker restart astrbot
  ```

依赖（`httpx`、`numpy`、`tzdata`）由 AstrBot 按 `requirements.txt` 自动安装。

## 7. 配置插件

AstrBot WebUI → 插件 → 大肥鱼 → 配置：

| 配置 | 填什么 |
|---|---|
| 主人 QQ 号 | **你自己的大号**（不是机器人小号） |
| 主人常用昵称 | 可留空。填了之后，昵称和它相似的其他人会被标记为疑似冒充 |
| 人设 / 群聊人设 | 可留空用默认人设；群聊人设留空时沿用私聊人设 |
| 逐句分条发送 | 默认「仅私聊」 |
| 对话模型来源 | `使用 AstrBot 里配置的提供商` |
| 使用的提供商 | 选第 5 步的 DeepSeek |
| 允许参与的群号 | 想让她出现的群，留空 = 所有群 |
| 时区 | `Asia/Shanghai` |

保存后，**用大号私聊机器人小号说一句话**。这一步让插件记下和你的私聊会话，之后她才能主动给你发消息。

## 8. 验证

用大号私聊机器人：

- `/自检`：各项为 ✅；日记、巩固在第一次到点前显示 ⏳ 是正常的；最后一行应为「已记录主人会话」。
- `/状态`：天气后面没有「（模拟）」，说明 Open-Meteo 访问正常。
- 随便聊两句，回复应由模型生成。如果总是「我在屋子的客厅发呆呢」这类固定句式，说明没调到模型，查看 AstrBot 日志。
- 过 10 分钟后 `/记忆 关键词` 能查到刚才说的话。
- 拉进群后：普通消息偶尔插话、@ 必回；在群里问别人私聊过的事，她不会说出来。

## 日常运维

```bash
# 看日志
docker logs -f --tail 100 astrbot

# 备份大肥鱼的全部状态（记忆、驱力、小岛、金币、关系）
tar czf dafeiyu-backup-$(date +%F).tgz -C ~/dafeiyu-bot/astrbot-data plugin_data/astrbot_plugin_dafeiyu

# 更新镜像（不要删 volume，否则 QQ 要重新扫码）
docker compose pull && docker compose up -d
```

- 掉线时先看 `docker logs snowluma`；QQ 登录失效就重新走第 3 步扫码。
- DeepSeek 没有向量接口，插件默认用本地哈希向量做语义检索。想要更准的召回，可以在插件配置里填百炼 `text-embedding-v4` 的 Key，或在 AstrBot 里配一个嵌入提供商并填它的 ID。
- 升级插件前先备份 `plugin_data/astrbot_plugin_dafeiyu/`；配置保存在 `config/astrbot_plugin_dafeiyu_config.json`，升级不会覆盖。
