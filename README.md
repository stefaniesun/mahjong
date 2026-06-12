# mahjong-eval

四川麻将识别项目的 Phase 0 评测基础设施与测试集工具链。

## 当前状态

- 已完成任务 1 的基础脚手架
- 已提供 `docs/source_curation.md` 与 `docs/annotation_guide.md`
- 已提供 `scripts/fetch_videos.py` 的 B 站可运行版本，用于任务 1.5 联调
- 已完成任务 4 的 `configs/cvat_labels.json`、`scripts/make_cvat_tasks.py` 与 CVAT 操作说明

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

## 任务 4：X-AnyLabeling 标注环境与预标注流程

### 当前状态

- 任务 4 的正式目标已切换为 **X-AnyLabeling** 工作流
- 当前仓库里的 `configs/cvat_labels.json` 与 `scripts/make_cvat_tasks.py` 属于旧版 CVAT 遗留产物，不再作为主流程推荐
- 当前仓库已补齐：`configs/xanylabel_classes.txt`、`scripts/make_prelabel.py`、`scripts/export_to_coco.py`

### 你会用到的文件

- `docs/annotation_guide.md`：四川麻将标注规范与两遍法操作建议
- `configs/classes.yaml`：全项目统一类别定义
- `configs/xanylabel_classes.txt`：供 X-AnyLabeling 导入的 29 类类别清单
- `scripts/make_prelabel.py`：为待标注图片批量生成同名 X-AnyLabeling JSON 预标注
- `scripts/export_to_coco.py`：把校正后的 X-AnyLabeling JSON 转成标准 COCO
- `scripts/validate_coco.py`：标注转成 COCO 后的自动校验工具

### 1. 安装 X-AnyLabeling

X-AnyLabeling 是桌面端标注工具，不需要 Docker、自部署或单独起 Web 服务。

建议直接从其官方发布页下载适合当前系统的版本，解压后即可运行。首次启动后，先确认软件能正常打开图片目录和标注文件。

### 2. 准备标注目录

当前 Phase 0 的目标流程是：

1. 从 `data/frames_selected/` 准备待标注图片
2. 使用 `scripts/make_prelabel.py` 生成与图片同名的 X-AnyLabeling 预标注 JSON
3. 在 X-AnyLabeling 中直接打开该目录进行校正
4. 校正完成后再统一转换为 COCO，供 `validate_coco.py` 与评测脚本使用

也就是说，主流程不再是“上传 zip 到平台”，而是“本地目录 + 同名 JSON”模式。

### 3. 生成预标注 JSON

在已有旧模型权重的前提下，可以先批量生成预标注：

```powershell
python scripts/make_prelabel.py --input-root data/frames_selected --model weights/legacy.pt --classes configs/classes.yaml --conf 0.25
```

说明：

- 脚本会对 `input-root` 下的每张图片生成一个同名 `.json`
- 输出格式为 X-AnyLabeling 可直接打开的 `rectangle` 标注
- 如果旧模型预测出了不在 29 类清单里的类别，会自动映射成 `unknown`

### 4. 在 X-AnyLabeling 中进行校正

推荐按下面顺序操作：

1. 启动 X-AnyLabeling
2. 打开待标注图片目录
3. 确认图片旁的同名 JSON 预标注能正常显示
4. 按 `docs/annotation_guide.md` 的规则检查并校正框与类别
5. 保存修改后的标注结果

推荐继续使用“两遍法”：

- 第一遍：先把所有牌框完整过一遍
- 第二遍：统一检查并修改类别

### 5. 转换为 COCO 后再做校验

标注目录整理完成后，先转成标准 COCO：

```powershell
python scripts/export_to_coco.py --input-dir data/frames_selected --output data/test_set_v1/annotations/instances_default.json --classes configs/classes.yaml
```

然后再运行：

```powershell
python scripts/validate_coco.py --annotations data/test_set_v1/annotations/instances_default.json --images-root data/test_set_v1/images --report data/test_set_v1/validation_report.json
```

`validate_coco.py` 会检查：

- COCO 结构是否合法
- 类别 ID / 名称是否与 `configs/classes.yaml` 一致
- 是否存在极小框、异常长条框、重复框、越界框
- 随机抽样可视化预览图，供人工复核

### 6. 推荐检查点

第一次从零操作时，按下面检查最稳：

- X-AnyLabeling 能正常打开待标注目录
- 图片对应的预标注 JSON 能显示并可编辑
- 按 `docs/annotation_guide.md` 能顺利完成一小批校正
- 转成 COCO 后，`validate_coco.py` 能正常输出报告与预览图

### 当前已支持能力

- `configs/xanylabel_classes.txt` 已提供 29 类类别清单
- `scripts/make_prelabel.py` 已支持生成 X-AnyLabeling 预标注 JSON
- `scripts/export_to_coco.py` 已支持把 X-AnyLabeling JSON 转成 COCO
- COCO 标注质量校验与预览图输出
- 旧 `CVAT` 工具链仍保留在仓库中，作为历史兼容内容


