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

- 一次抓取 `sources.yaml` 里全部博主（增量、可中断续传）：

```bash
python scripts/fetch_videos.py --platform bili --cookies configs/cookies/bilibili.txt --browser ""
```

### B 站 412 限流与下载提速

B 站对 UP 主空间列表接口（WBI 签名）有反爬限流，**最常见的坑是空间视频很多时翻遍全部分页**会瞬间发起几十次请求直接被封 `412 Request is blocked by server`。脚本已针对性处理：

- `--list-limit N`（默认 40）：拉空间列表时只取**最新 N 个**视频，一页搞定，从根本上避免深翻分页触发 412
- `--candidates-cache PATH`（默认 `data/raw_videos/_candidates_cache.json`）：每个博主的列表成功拉取一次即落盘缓存，重跑直接复用、不再请求；加 `--refresh-list` 可强制刷新
- 配额已满的博主（state 里已记录 ≥ `max_videos`）直接跳过、连列表都不请求
- 列表请求命中 412 时按指数退避重试（`--list-backoff-base` 30s 起，`--list-backoff-max` 上限 300s，`--list-retries` 次）
- 不同博主之间留 `--author-gap-min/max`（默认 10~20s）间隔

若仍被限流，等几分钟让冷却结束再重跑即可（已下完的博主会被状态跳过）。

> 下载提速：B 站常把单连接限到约 100~150 KB/s。若安装了 `aria2c`，可让 yt-dlp 走多连接显著提速；当前机器未安装 `aria2c` / `ffmpeg`（缺 `ffmpeg` 时音视频流不会被合并，但纯视频帧提取不受影响）。

## 任务 2：智能抽帧

`scripts/extract_frames.py` 会从 `data/raw_videos/` 递归查找视频，按指定采样率抽帧，并过滤模糊、欠曝、过曝帧。

### 运行示例

```powershell
python scripts/extract_frames.py --input-root data/raw_videos --output-root data/frames_candidate --report data/frames_candidate/extract_report.json
```

常用参数：

```powershell
python scripts/extract_frames.py --fps 0.5 --blur-threshold 100 --min-brightness 30 --max-brightness 225
```

说明：

- 默认每秒抽 `0.5` 帧，即约每 `2` 秒取一帧
- 输出命名为 `{视频文件名}_f{帧号:06d}.jpg`
- 会保留原始分辨率
- 会保留原博主目录结构，例如输出到 `data/frames_candidate/bili_1014433798/`
- 报告写入 `extract_report.json`，包含每个视频的采样数、保存数、模糊丢弃数、欠曝丢弃数、过曝丢弃数

### 模糊阈值怎么校准

`--blur-threshold` 使用灰度图的 Laplacian 方差，数值越低通常越糊。默认值是 `100`，适合作为第一轮粗筛起点。

推荐校准流程：

1. 先抽 1~2 个视频小样本：

```powershell
python scripts/extract_frames.py --input-root data/raw_videos --output-root data/frames_candidate_probe --report data/frames_candidate_probe/extract_report.json --limit-videos 2 --fps 0.5 --blur-threshold 100
```

2. 人眼检查 `data/frames_candidate_probe/`：
   - 如果明显糊帧还很多，把阈值提高到 `150` 或 `200`
   - 如果可用帧被误删太多，把阈值降低到 `50` 或 `80`
3. 确认阈值后，再对全量视频运行正式抽帧。

曝光阈值默认过滤平均亮度 `<30` 的欠曝帧和 `>225` 的过曝帧；如果视频整体偏暗，可适当降低 `--min-brightness`。

## 任务 3：去重与多样性采样

`scripts/dedup_filter.py` 会从 `data/frames_candidate/` 中按博主均衡选出最终待标注图片到 `data/frames_selected/`。

### 运行示例

```powershell
python scripts/dedup_filter.py --input-root data/frames_candidate --output-root data/frames_selected --report data/frames_selected/selection_report.json --total 500 --per-video-cap 8 --min-gap-sec 30 --clean-output
```

说明：

- 使用 `imagehash.phash` 做感知哈希去重，默认汉明距离 `<=8` 视为重复
- 同一重复组优先保留 Laplacian 方差更高的清晰帧
- 默认按博主目录名（如 `bili_1014433798`）均衡分配配额
- 单博主占比默认不超过 `25%`
- 单视频默认最多选 `8` 张
- 同一视频内优先选择间隔至少 `30` 秒的帧
- 输出文件名带来源前缀：`{source}__{video_id}__f000001.jpg`
- 报告写入 `selection_report.json`，包含每个博主、每个视频的候选数、去重后数量、选中数量和入选时间戳

只看报告不复制图片：

```powershell
python scripts/dedup_filter.py --input-root data/frames_candidate --output-root data/frames_selected --report data/frames_selected/selection_report.json --total 500 --dry-run
```

如果源视频不是 `30 fps`，可用 `--source-fps` 调整从帧号估算时间戳的比例。



```bash
python scripts/screen_web_videos.py --input-root data/raw_videos --report data/web_screen/screen_report.json --preview data/web_screen/preview.html --clips-root data/web_clips --skip-ffmpeg
```

说明：

- 该脚本默认按 `1 fps` 采样，输出视频级判定、有效片段区间和 HTML 预览页
- 若本机已安装 `ffmpeg`，去掉 `--skip-ffmpeg` 即可把保留片段切到 `data/web_clips/`
- 当前版本可通过任务 0.5 的预标注器 ONNX 替换启发式含牌检测：模型路径登记在 `configs/paths.yaml` 的 `prelabeler_onnx`

## 任务 0.5：训练预标注模型

任务 0.5 已补齐以下文件：

- `docs/train_prelabeler.md`：训练、真实域验证、ONNX 导出和下游使用手册
- `configs/prelabel_map.yaml`：Roboflow 27 类到本项目标签的映射表，`B=条`、`C=万`、`D=筒`
- `configs/paths.yaml`：登记 `prelabeler_onnx`、`prelabeler_pt` 和源数据路径
- `scripts/train_prelabeler.py`：跨平台训练/预测/导出入口
- `scripts/train_prelabeler.sh`：bash 包装入口

常用命令：

```powershell
py scripts/train_prelabeler.py --mode train
py scripts/train_prelabeler.py --mode predict --source data/frames_selected --conf 0.25
py scripts/train_prelabeler.py --mode export
py -m pytest tests/test_prelabel_map.py
```

生成 X-AnyLabeling 预标注时可直接使用登记路径：

```powershell
py scripts/make_prelabel.py --input-root data/frames_selected --paths configs/paths.yaml --prelabel-map configs/prelabel_map.yaml --conf 0.25
```



### 当前已支持能力

- 读取 `configs/sources.yaml`
- B 站主页抓取（UID / URL）
- `include_keywords` / `exclude_keywords` 标题过滤
- 增量下载状态 `data/raw_videos/download_state.json`
- 每次运行输出 `data/raw_videos/fetch_report.json`
- 每个视频保留 `.info.json`
- 随机 3~8 秒间隔、失败重试
- B 站 412 限流处理：限量列表（`--list-limit`）、列表缓存（`--candidates-cache`）、配额跳过、指数退避重试

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

在已有任务 0.5 预标注器 ONNX 的前提下，可以先批量生成预标注：

```powershell
python scripts/make_prelabel.py --input-root data/frames_selected --paths configs/paths.yaml --prelabel-map configs/prelabel_map.yaml --conf 0.25
```

说明：

- 脚本会对 `input-root` 下的每张图片生成一个同名 `.json`
- 默认从 `configs/paths.yaml` 读取 `prelabeler_onnx`，也可以用 `--model` 显式覆盖
- 输出格式为 X-AnyLabeling 可直接打开的 `rectangle` 标注
- `configs/prelabel_map.yaml` 会把预标注器 27 类映射到本项目的万/条/筒类别
- 如果模型预测出了不在映射表或 29 类清单里的类别，会自动映射成 `unknown`


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


