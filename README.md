# mahjong-eval

四川麻将识别项目的 Phase 0 评测基础设施与测试集工具链。

## 当前状态

- 已完成任务 1 的基础脚手架
- 已提供 `docs/source_curation.md` 与 `docs/annotation_guide.md`
- 已提供 `scripts/fetch_videos.py` 的 B 站可运行版本，用于任务 1.5 联调

## 目录说明

- `configs/`: 类别、数据源与标注相关配置
- `data/`: 本地视频、抽帧和测试集数据目录
- `docs/`: 人工操作文档
- `scripts/`: CLI 工具脚本目录
- `eval/`: 评测脚本与报告模板目录

## 任务 1.5：抓取视频

### 先决条件

- Python 3.10+
- 已安装 `yt-dlp`
- 建议准备 `configs/cookies/bilibili.txt`（Netscape 格式 Cookie 文件）
- 如需走抖音降级方案，准备 `configs/cookies/douyin.txt` 以及一个 URL 清单 JSON

### 为什么推荐 Cookie 文件而不是直接读 Chrome

当前 Windows + Chrome 环境下，`yt-dlp --cookies-from-browser chrome` 可能因为浏览器数据库锁定而失败。为了稳定联调，建议优先使用**导出的 Cookie 文件**：

1. 在 Chrome 登录 B 站
2. 用浏览器扩展导出当前站点 Cookie 为 **Netscape 格式**
3. 保存到 `configs/cookies/bilibili.txt`
4. 运行脚本时传 `--cookies configs/cookies/bilibili.txt`

如果你后面希望继续尝试浏览器直读，也可以保留：

```bash
python scripts/fetch_videos.py --browser chrome
```

但当前机器上，这条链路已经实测会失败，所以**文件 Cookie 更稳**。

### 抖音当前方案

当前仓库已支持两种模式：

- `bili`：使用 `yt-dlp` 直接从 UP 主页抓取
- `dy`：优先预留专用后端接入口；如果后端暂不可用，使用 **URL 清单降级方案** 继续推进流程

#### 抖音 Cookie 获取建议

1. 浏览器登录抖音网页版
2. 打开开发者工具 `F12`，进入 `Network`
3. 刷新页面，任选一个 `douyin.com` 请求
4. 复制请求头中的 `Cookie`
5. 保存到 `configs/cookies/douyin.txt` 备查（当前降级模式主要用于你后续借助第三方下载器批量下载）

#### URL 清单降级方案

当抖音专用下载后端不稳定时，可先整理一个 `JSON` 清单交给脚本管理后续状态。结构示例：

```json
{
  "dy_douyin_user_id": [
    {
      "id": "video_001",
      "title": "四川麻将实战夜局",
      "url": "https://www.douyin.com/video/1234567890",
      "upload_date": "20260101",
      "uploader": "某博主"
    }
  ]
}
```

运行 dry-run 枚举：

```bash
python scripts/fetch_videos.py --platform dy --dry-run --douyin-manifest configs/douyin_urls.json --browser ""
```

说明：

- `key` 必须是 `platform_uid`，例如 `dy_douyin_user_id`
- 脚本会继续执行 `include_keywords` / `exclude_keywords` 过滤
- 脚本会维护 `download_state.json` 与 `fetch_report.json`
- 当前仓库还**不会自动下载抖音视频文件本体**，但不会阻塞你先做链接整理、筛选与归档

### 运行示例


- 最小联调（只跑 1 个博主，先不真正下载）：

```bash
python scripts/fetch_videos.py --platform bili --limit-authors 1 --dry-run --cookies configs/cookies/bilibili.txt
```

- 实际下载：

```bash
python scripts/fetch_videos.py --platform bili --limit-authors 1 --cookies configs/cookies/bilibili.txt
```

- 任务 2.5：对已下载网络视频做粗筛并生成预览页：

```bash
python scripts/screen_web_videos.py --input-root data/raw_videos --report data/web_screen/screen_report.json --preview data/web_screen/preview.html --clips-root data/web_clips --skip-ffmpeg
```

说明：

- 该脚本默认按 `1 fps` 采样，输出视频级判定、有效片段区间和 HTML 预览页
- 若本机已安装 `ffmpeg`，去掉 `--skip-ffmpeg` 即可把保留片段切到 `data/web_clips/`
- 当前版本优先保证粗筛链路可跑通；后续可再接入你提供的旧 YOLO11 模型，替换当前启发式含牌检测

### 当前已支持能力

- 读取 `configs/sources.yaml`
- B 站主页抓取（UID / URL）
- `include_keywords` / `exclude_keywords` 标题过滤
- 增量下载状态 `data/raw_videos/download_state.json`
- 每次运行输出 `data/raw_videos/fetch_report.json`
- 每个视频保留 `.info.json`
- 随机 3~8 秒间隔、失败重试

### 当前限制

- 抖音后端尚未接入；下一步会按规格接专用工具或降级成 URL 清单方案
- 当前机器实测 `--cookies-from-browser chrome` 失败，因此 B 站联调建议改用导出的 Cookie 文件

## 下一步

1. 导出 B 站 Cookie 到 `configs/cookies/bilibili.txt`
2. 运行 `scripts/fetch_videos.py` 验证该博主的最小抓取
3. 继续补任务 1.5 的抖音后端与降级方案
