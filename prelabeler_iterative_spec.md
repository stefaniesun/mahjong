# 需求规格:预标注器滚动迭代训练管线(Prelabeler Iterative Loop)

> 本文档交给 AI 编码助手(opencode / codex / Claude Code)执行。
> 全部代码用 Python 3.10+,依赖写入 requirements.txt,每个脚本支持 `--help`。
> **本管线是反复使用的工具**:每当我新标注一批真实数据,就重跑一次,产出更强的预标注器,用它再去标更多数据,如此循环。请把"可重复运行、版本化、配比可控"作为第一设计原则。

---

## 背景与目标(供 AI 理解上下文)

四川麻将识别项目,两阶段架构,检测器为**单类 `tile_face`**(只检测牌面,牌背不检测当背景)。

我需要一个**预标注器**(辅助标注用的检测模型):用它在 X-AnyLabeling 里把框先画好,我只校正。预标注器越准,我标注越省力。

**现状与痛点**:我现有一个 Roboflow 训练的旧模型(27 类游戏域),它只能辅助识别清晰的手牌,**牌池(中距离、小、斜、叠、糊的牌)识别很差**——因为它没见过真实牌池分布。我已精标了约 70 张真实牌局图(含牌池),想用"真实数据 + Roboflow 数据"重训一个更强的预标注器,解决牌池问题。

**滚动迭代愿景**:
```
70张真实 + Roboflow数据 → 训预标注器 v1 → 用 v1 辅助标注 +100张
→ 170张真实 + Roboflow数据 → 训预标注器 v2(牌池更准)→ 辅助标注 +100张
→ 270张 → v3 → ... 每轮预标注器更强,标注越来越省力,数据越滚越多
```

**这套脚本要支持上面整个循环,每轮我只改一个数据版本号/路径就能重跑。**

**运行环境:训练在云算力平台(如 AutoDL)的 GPU 实例上进行**,本地只做标注与校正。因此整套管线必须为"云上跑、按量计费、跑完即关"优化:路径与机器解耦、支持增量上传(每轮只传新增的真实标注,不重传 Roboflow 大数据集)、产物可一键打包下载。具体见任务 6。

### 关键数据源

1. **我的真实标注**(逐轮增长):X-AnyLabeling 格式(图 + 同名 .json),28 类细标(w/t/b + unknown),可能含早期误标的 back(丢弃)。位置:`D:\mahjong\data\labeled\`(每轮我会往里加新博主的标注)
2. **Roboflow 数据集**:YOLO 格式,27 类(1B~9D = 条/万/筒),游戏域。位置:`D:\mahjong\Mahjong.v1i.yolov11\`(固定不变)

### 类别处理(本管线训练的是单类检测器)

- 我的真实标注:所有 w/t/b/unknown 框 → 压平为 `tile_face`(类别 0);back 标签丢弃
- Roboflow 数据:所有 27 类 → 压平为 `tile_face`(类别 0)
- 训练目标:单类检测,只学"哪里有牌面"

### 仓库结构(任务 1 创建)

```
mahjong-prelabeler/
├── configs/
│   ├── paths.yaml            # 路径配置:支持 local/cloud 两套 profile(见任务6)
│   └── train_iter.yaml       # 迭代训练参数(配比、轮数等,见任务3)
├── scripts/
│   ├── prepare_data.py       # 两数据源转单类 + 按配比组装混合集
│   ├── train_prelabeler.py   # 训练 + 版本化
│   ├── eval_prelabeler.py    # 在留出真实验证集上评测(对比上一版)
│   ├── export_prelabeler.py  # 导出 ONNX + 接入预标注的说明
│   └── make_prelabel.py      # 用最新预标注器对新图生成 X-AnyLabeling 预标注
├── datasets/                 # gitignore,每轮组装的混合集(清单式,不复制图)
│   └── mix_v{N}/
├── runs/                     # gitignore,每轮训练产物
│   └── prelabeler_v{N}/
├── docs/
│   └── iteration_howto.md    # 给我的"每轮怎么操作"手册
├── requirements.txt
└── README.md
```

---

## 任务 1:脚手架 + 迭代操作手册

**做什么**:建结构、写 `paths.yaml`,写 `docs/iteration_howto.md`。

**`iteration_howto.md` 要写清"每一轮"我的操作步骤**(这是反复用的核心):
1. 把新标注的博主图放进 `labeled/`(累加,不覆盖旧的)
2. 跑 `prepare_data.py`(自动统计真实数据新增量,按配比组装混合集 mix_v{N})
3. 跑 `train_prelabeler.py`(训出 prelabeler_v{N})
4. 跑 `eval_prelabeler.py`(看新版 vs 旧版在留出真实验证集上的提升,尤其牌池/小目标召回)
5. 跑 `export_prelabeler.py`(导出 ONNX)
6. 跑 `make_prelabel.py`(用新预标注器预标注下一批新图)→ 回 X-AnyLabeling 校正
7. 循环回第 1 步
并写明:每轮大约多少数据、什么时候提升会变缓(收益递减)、何时可以停止迭代(预标注已足够省力)。

**验收**:手册可照做;结构完整。

---

## 任务 2:数据准备与配比组装 `scripts/prepare_data.py`(本管线核心)

**做什么**:把两个数据源转成单类,**按可控配比**组装成混合训练集 + 一个固定的真实验证集。

**功能需求**:

1. **真实数据转换**:扫描 `labeled/`,X-AnyLabeling JSON → 单类 YOLO 标注;back 丢弃;非法 label 列异常清单(不静默丢);记录每张图来源博主(从路径/文件名)

2. **留出固定真实验证集(关键,先于配比)**:从真实数据中**按博主分层抽样**留出一份验证集 `real_val`(默认每博主留 15%,且固定 seed,**跨轮保持稳定**——同一张图一旦进 val 就一直在 val,新增数据只扩充不打乱旧划分)。这是评测预标注器"在真实域(尤其牌池)进步多少"的标尺,**绝不参与训练**

3. **Roboflow 数据转换**:YOLO 27 类 → 单类;全部可用于训练(它不是真实域,不留验证)

4. **按配比组装混合训练集(核心可调)**:
   - 配比参数在 `train_iter.yaml`,**以"真实数据有效占比"为控制目标**,而非简单堆数量
   - 默认策略:**真实数据上采样**(重复采样)到与 Roboflow 数据达到目标比例。推荐默认 `real_effective_ratio: 0.35`(即混合集中真实样本的有效采样占比约 35%)
   - 原因写进注释:真实数据是教模型学牌池的关键,占比太低(如 70 张淹没在几千张 Roboflow 里)模型学不到牌池;占比太高(纯 70 张)又会过拟合、牌面多样性不足。35% 左右是平衡点,且**随真实数据增多,这个比例可逐轮提高**(见下)
   - **自适应配比建议**:脚本根据当前真实图数量,在报告里给出推荐 ratio——真实图越多,越可以提高真实占比、降低对 Roboflow 的依赖。给一张参考表(写进文档):
     | 真实图数量 | 建议 real_effective_ratio | 说明 |
     |---|---|---|
     | <150 | 0.30~0.35 | 真实数据少,靠 Roboflow 补多样性,真实上采样 |
     | 150~400 | 0.40~0.55 | 真实数据渐成主力 |
     | 400~800 | 0.60~0.75 | Roboflow 退为辅助 |
     | >800 | 0.80~1.0 | 可考虑基本脱离 Roboflow,真实数据为主 |
   - 组装用**符号链接 + 重复采样清单**实现上采样,不物理复制图像

5. **输出**:
   - `datasets/mix_v{N}/`(train 清单 + data.yaml,单类)与固定的 `real_val/`
   - `prepare_report.json`:真实图总数/新增数、各博主分布、Roboflow 图数、**实际有效配比**(真实:Roboflow 的采样占比)、real_val 规模、异常 label 清单、各尺寸桶框数分布
   - 版本号 N 自增,记录本轮用的配比与数据快照

**验收标准**:对当前 70 张 + Roboflow 运行,正确输出混合集与固定 real_val;报告中"实际有效配比"与配置目标一致(±3%);real_val 划分可复现且跨轮稳定(再次运行同一张图仍在 val);异常 label 清单准确。

---

## 任务 3:迭代训练 `scripts/train_prelabeler.py`

**做什么**:用混合集训单类检测器,版本化产出。

**功能需求**:
1. Ultralytics YOLO11,单类;`train_iter.yaml` 配置:model(默认 yolo11s.pt,小数据够用且快)、imgsz(默认 960)、epochs(默认 80)、batch(自动)、patience(默认 20 早停)
2. **小目标增强**:加 P2 检测头配置可选(`use_p2: true`,默认开)——预标注器也要尽量框到牌池小牌,P2 有帮助
3. 防过拟合(真实数据少时重要):标准增广 + 适度 mosaic;**禁水平翻转**(条/万翻转语义变)——加注释与单测
4. 训练中在 `real_val` 上评测(不是 Roboflow 的 val,要看真实域表现),每 epoch 记录 real_val 的 mAP50 与小目标召回
5. 版本化:`runs/prelabeler_v{N}/` 含配置快照、混合集版本号、best.pt、训练曲线;**自动续接上一版**(可选 `--from-prev`:从 prelabeler_v{N-1} 权重微调而非从头训,数据多时更快收敛)
6. 打印 best.pt 路径与 real_val 关键指标

**验收标准**:训练跑通收敛;real_val 指标曲线正常;禁翻转有单测;版本目录自包含。

---

## 任务 4:版本对比评测 `scripts/eval_prelabeler.py`

**做什么**:量化"这一版预标注器比上一版强多少",尤其牌池能力。

**功能需求**:
1. 在固定 `real_val` 上评测指定版本(默认最新);指标:整体 mAP50/recall + **按尺寸分桶召回(<20px / 20~40px / >40px)**——牌池主要是小桶,这是核心进步信号
2. **自动对比上一版**:并排表 + 提升量(本版 vs v{N-1}),重点高亮小目标桶的变化
3. 可视化:在 real_val 抽 15 张,把预测框画出来(尤其展示牌池区域),输出对比图,让我直观看到"牌池现在框得怎么样了"
4. 输出 HTML 报告 + 一句结论:"相比上版,牌池召回 +X%,建议:继续迭代 / 已够用可降低标注校正强度"

**验收标准**:对比表与提升量正确;小目标桶单独可见;可视化能看出牌池改善。

---

## 任务 5:导出 + 预标注 `scripts/export_prelabeler.py` + `make_prelabel.py`

**export_prelabeler.py**:
1. 最新 best.pt 导出 ONNX(imgsz 960);精度对齐检查(.pt vs onnx argmax/框一致)
2. 产出 `prelabeler_latest.onnx`(固定名,软链到最新版),下游 make_prelabel 始终用这个,无需每轮改路径

**make_prelabel.py**(反复用的预标注工具):
1. 加载 `prelabeler_latest.onnx`,对指定新图目录推理(conf 默认 0.25,宁多勿漏)
2. 输出 **X-AnyLabeling 兼容预标注 JSON**(每图同名,矩形 shape,label 一律先给 `tile_face`——因为是单类检测器,类别我在 X-AnyLabeling 第二遍手动改成 28 细类)
3. **可选分类建议(锦上添花,默认关)**:`--cls-hint` 开启时,对每个框裁切喂给 Roboflow 老模型(27类)预测类别建议,把建议 label 写进预标注 JSON(映射 1B→t1 等,见映射表),省我手标类别;关闭则全标 tile_face
4. 输出预标注统计:每图框数、平均置信度

**验收标准**:ONNX 对齐通过;预标注 JSON 在 X-AnyLabeling 正常显示;`--cls-hint` 开关都能跑。

---

## 任务 6:云算力平台适配(本管线在云上跑,必做)

**做什么**:让整套迭代在云 GPU 实例(以 AutoDL 为参考,其它平台同理)上顺畅运行,把"开机→同步数据→训练→下载→关机"做成最少步骤、最低成本。

**功能需求**:

1. **路径双 profile**:`paths.yaml` 支持 `local` 与 `cloud` 两套路径配置,用环境变量或 `--env cloud` 切换。
   - 本地:`labeled/` 在 `D:\mahjong\data\labeled`,产物在本地
   - 云端:数据与产物根目录统一放数据盘 `/root/autodl-tmp/mahjong/`(AutoDL 数据盘,关机不丢、空间大);**绝不放系统盘**
   - 脚本内部一律用 paths.yaml 的键,不写死任何绝对路径

2. **增量数据同步(省时省钱核心)**:Roboflow 数据集(几 GB,固定不变)只需上传一次、常驻云数据盘;**每轮只需上传新增的真实标注**(几十张图+json,几 MB)。
   - 提供 `scripts/sync_to_cloud.py`(或一段 rsync/scp 封装):比对本地 `labeled/` 与云端,只传新增/变更文件;输出本轮上传了多少新图
   - 提供反向 `scripts/sync_from_cloud.py`:训练后只拉回产物(best.pt、onnx、eval 报告、prepare/train 报告),不拉回庞大的混合集/数据集
   - 也支持用 Roboflow API 在云端直接重新下载 Roboflow 数据(免上传那几 GB):云端首次执行 `roboflow download` 落到数据盘

3. **一键编排脚本 `scripts/run_iteration_cloud.sh`**(在云实例上跑):依次执行 `prepare_data → train_prelabeler → eval_prelabeler → export_prelabeler`,全程无人值守;结束后把本轮所有产物打包成 `runs/prelabeler_v{N}_bundle.zip`(含 onnx、best.pt、两份报告),方便我一次性下载。脚本结尾打印醒目提示:"训练完成,产物已打包;请下载 bundle 后到 AutoDL 控制台**关机停止计费**"

4. **环境自举 `scripts/setup_cloud_env.sh`**:在新开的云实例上一键装好依赖(`pip install -r requirements.txt`)、校验 GPU 可用(`torch.cuda.is_available()` 必须 True,否则醒目报错提示换 CUDA 版 torch)、校验数据盘路径存在。新开实例或换实例时跑一次即可

5. **成本与耗时提示**:`run_iteration_cloud.sh` 开头按当前混合集规模 + GPU 型号,粗估本轮训练时长与花费(给个量级即可),让我决定是否值得这一轮就训;`--from-prev` 开启时提示"增量微调,耗时约为全量的 1/2~1/3"

6. **断点保护**:训练支持 resume(实例被抢占/掉线后,重开能从最近 checkpoint 续训,不浪费已花的机时)

**验收标准**:
- 在一台干净云实例上:`setup_cloud_env.sh` → 上传新增标注(或 Roboflow API 下载)→ `run_iteration_cloud.sh` 一条龙跑通,产出 bundle.zip
- 第二轮:只上传新增的几十张真实标注(几 MB,秒级),不重传 Roboflow
- 全程不碰系统盘;关机后再开机,数据盘内容与产物仍在
- 成本/耗时提示合理(与实际量级相符)

**我(人类)在云上每轮的操作(应被压缩到极简)**:
1. AutoDL 开机(数据盘里 Roboflow 与历史数据都在)
2. (本地)`sync_to_cloud.py` 传新增标注 —— 几 MB,秒级
3. (云端)`run_iteration_cloud.sh` —— 挂机 20~40 分钟
4. (本地)`sync_from_cloud.py` 拉回 bundle
5. AutoDL 关机(停止计费)
6. 本地用拉回的 onnx 跑 `make_prelabel.py` 预标下一批

---

## 全局技术约束

- Python 3.10+;依赖:ultralytics, torch, opencv-python, pyyaml, numpy, matplotlib, jinja2, tqdm, onnx, onnxruntime
- **可重复运行是第一原则**:每个脚本幂等;版本号自增不覆盖;real_val 划分跨轮稳定(用确定性 hash 或固定清单,不能每轮重新随机)
- 配比/路径全部走 configs,不写死
- 上采样用清单不复制图;实验目录自包含
- pytest:X-AnyLabeling→单类转换、配比采样(验证实际有效占比)、real_val 稳定性(同输入同划分)、禁翻转——必须有单测
- **机器无关**:不在代码里写死本地或云的绝对路径;同一套代码靠 paths.yaml 的 local/cloud profile 在两端都能跑
- 云端默认数据根 `/root/autodl-tmp/mahjong/`;所有大文件(数据集、混合集、权重)只落数据盘,系统盘只放代码

## 执行顺序(每一轮都这样,云端版)

**首次(一次性)**:开实例 → `setup_cloud_env.sh` → 上传 Roboflow 数据(或云端 API 下载)到数据盘

**每轮**:开机 → (本地)`sync_to_cloud.py` 传新增标注 → (云端)`run_iteration_cloud.sh`(内部依次 prepare_data→train→eval→export,产出 bundle.zip)→ (本地)`sync_from_cloud.py` 拉回 bundle → **关机** → (本地)`make_prelabel.py` 用新 onnx 预标下一批 → 我在 X-AnyLabeling 校正 → 下一轮

> 说明:prepare_data / make_prelabel 这类轻量步骤也可在本地跑(make_prelabel 必须本地跑,因为要喂给本地的 X-AnyLabeling);只有 train 这步必须在云 GPU 上。eval/export 跟随 train 在云上一起跑完最省事。

## 我(人类)负责的部分

- 每轮往 `labeled/` 添新标注的博主数据
- 看 eval 报告判断:牌池够好了吗?继续迭代还是降低投入?
- 调 `real_effective_ratio`(参考自适应表,或直接用脚本推荐值)
- 用 make_prelabel 的结果在 X-AnyLabeling 校正(框 + 改类别)

## 设计要点强调(给 AI)

1. **真实/Roboflow 配比是这套脚本的灵魂**:必须以"有效采样占比"为控制量,通过对真实数据上采样实现,而不是简单合并目录。报告里要能看到实际占比,且能随数据增长调整。
2. **real_val 必须跨轮稳定**:否则每轮指标不可比,看不出真实进步。用固定划分(同一图永远在同一侧)。
3. **不要为单轮精度过度优化**:这是迭代管线,单轮够用即可,价值在循环累积。
4. **预标注器是工具不是终点**:它和最终正式检测器(Phase 2)是同架构(单类),将来数据足够时,这套训练产物可平滑升级为正式检测器,不浪费。
