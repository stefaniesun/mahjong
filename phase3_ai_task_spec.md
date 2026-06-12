# Phase 3 需求规格:麻将识别项目 — 29 类牌面分类器训练

> 本文档交给 AI 编码助手(opencode / codex / Claude Code)执行。
> 请严格按任务顺序实现,每个任务有明确的输入、输出和验收标准。
> 全部代码用 Python 3.10+,依赖写入 requirements.txt,每个脚本支持 `--help`。
> 本 Phase 依赖:Phase 0 `mahjong-eval`(冻结测试集 + eval_classification.py)、Phase 1 `cls_synth_v{N}`(合成 crop 集)、Phase 2 导出的检测器 ONNX(及其规格说明)。
> 本 Phase 可与 Phase 2 的后几轮迭代并行进行。

---

## 项目背景(供 AI 理解上下文)

四川麻将识别系统,两阶段架构。本 Phase 做第二阶段:**对检测器输出的牌面 crop 做 29 类分类**(w1~w9 / t1~t9 / b1~b9 / back / unknown)。

**最终验收目标(在真实 crop 冻结测试集上,见任务 2)**:
- 总体 top-1 ≥99%(剔除 unknown 后的 27+back 类)
- 最差单混淆对错误率 <0.5%,重点盯:4条vs5条、6条vs9条、2条vs3条、万字组内
- 置信度可用:经校准后,设阈值能做到"接受样本准确率 ≥99.5%、弃权率 ≤5%"(弃权样本交给 Phase 4 时序投票消化)
- 推理:96×96 输入,CPU 单 crop <3ms / GPU 批量 <0.5ms(为 Phase 5 留余量)

**技术路线**:MobileNetV3-Small(基线)与 EfficientNet-B0(备选)双跑取优;合成 crop 预训练 → 真实 crop(检测器收割 + 人工文件夹分拣)混合微调;置信度校准 + 弃权机制。

### 数据纪律

- 真实 crop 的来源视频必须经 `get_leaked_videos()` 排除冻结测试集来源(复用 Phase 1 实现)
- 分类器的冻结测试 crop 集由 test_set_v1 的 GT 框直接生成(任务 2),与检测冻结集同源同纪律:只做里程碑评测、强制留痕

### 仓库目录结构(任务 1 中创建)

```
mahjong-cls/
├── configs/
│   ├── paths.yaml             # 指向 mahjong-eval/synth/det 各产物
│   └── train_base.yaml        # 训练超参配置
├── scripts/
│   ├── build_frozen_crops.py  # 从冻结测试集生成分类冻结评测集
│   ├── harvest_crops.py       # 用检测器从视频收割真实 crop + 预分类
│   ├── clean_labels.py        # 分拣后标签噪声检测
│   ├── train.py               # 训练入口
│   ├── run_eval.py            # 一键评测(对接 Phase 0)
│   ├── calibrate.py           # 置信度校准与弃权阈值选择
│   └── export_model.py        # ONNX 导出 + 对齐
├── runs/                      # 实验目录(gitignore),规范同 Phase 2
├── data/                      # gitignore
│   ├── frozen_crops_v1/       # 分类冻结评测集(按类分文件夹)
│   ├── real_cls_candidate/    # 待我分拣的预分类 crop
│   ├── real_cls_v1/ v2/ ...   # 我分拣完成的真实 crop 库
│   └── mixed_v1/ ...          # 混合训练集(清单式,不复制文件)
├── docs/
│   └── sorting_guide.md       # 给我的分拣操作指引
├── requirements.txt
└── README.md
```

---

## 任务 1:项目脚手架 + 分拣操作指引

**做什么**:建目录、写 paths.yaml、写 `docs/sorting_guide.md`。

**`sorting_guide.md` 内容**:
- 分拣方式:文件管理器大图标模式逐文件夹扫视,把放错类的 crop 拖到正确文件夹;预期速度 2000~3000 张/小时
- **必须人工重点复核的文件夹清单**:t4 / t5 / t6 / t9 / t2 / t3(条子组,旋转和数数易错)、w 组全部(字形)、unknown(边界最模糊)
- unknown 的判定标准(与 Phase 0 标注规范一致并细化到 crop 视角):侧棱 / 可见面 <30% / 糊到人眼 1 秒内认不出 / 字牌花牌
- 不确定就放 unknown 的原则:分类训练宁可少一张样本,不能进一张错标
- 每完成一批,把整个候选目录重命名为 `real_cls_v{N}`,不要在旧版本上追加(版本可追溯)

**验收**:文档可直接执行;目录完整。

---

## 任务 2:分类冻结评测集 `scripts/build_frozen_crops.py`

**做什么**:从 `test_set_v1` 的 GT 标注生成分类器专用冻结评测集,这是本 Phase 所有最终指标的标尺。

**功能需求**:
1. 读取冻结测试集 COCO,按 GT 框(用 GT 而非检测器输出,保证评的是纯分类能力)crop,外扩 8% 边距,保持原始分辨率不缩放(评测时再 resize,保留真实小牌的退化信息)
2. 按 GT 的 29 类标签分文件夹输出 `frozen_crops_v1/`;每张 crop 文件名编码来源(图名+框id+尺寸桶+source 博主)
3. 同时生成"尺寸分桶清单":<20px / 20~40px / >40px,供评测分桶出指标
4. 统计报告:每类数量(某类 <30 张要警告——意味着冻结测试集该类覆盖不足,提示我未来 v2 补)
5. 与检测冻结集相同的留痕纪律:评测脚本访问此目录时记录日志(在任务 5 实现)

**验收标准**:crop 与 GT 标注严格对应(抽 30 张人工核对);分桶清单数量对得上;报告完整。

---

## 任务 3:真实 crop 收割与预分类 `scripts/harvest_crops.py`

**做什么**:用 Phase 2 检测器从博主视频批量收割真实 crop,预分类后交我分拣,把人工从"标注"降为"挑错"。

**功能需求**:
1. 排除泄漏视频;对视频按 1 fps 抽帧(复用质量过滤),用 Phase 2 ONNX 检测器推理(按其规格文档解码,conf ≥0.5)
2. crop 规则与任务 2 一致(外扩 8%、保留原始分辨率)
3. **预分类**:用当前最优分类器(第一轮用纯合成训练版)预测,按预测类别放入 `real_cls_candidate/{类名}/`;同时按预测置信度在文件名加前缀(`hi_` ≥0.9 / `mid_` 0.6~0.9 / `lo_` <0.6),让我分拣时优先看 mid/lo
4. **配额与多样性**:目标每轮收割 2~5 万张;phash 去重(同一牌在连续帧反复出现);每类配额均衡;小尺寸桶(<25px)强制占比 ≥30%(分类器最需要的恰是这些难样本);博主分布均衡
5. tile_back 检出的 crop 进 back 候选;检测置信度 0.5~0.65 的"可疑检出"单独放 `_review/` 文件夹(可能是误检,也是 unknown 类的好素材)
6. 输出收割报告:各类/各置信度段/各尺寸桶数量

**验收标准**:候选文件夹抽查,hi_ 前缀样本预分类正确率 ≥97%(意味着我的分拣工作量主要集中在 mid/lo);尺寸与博主分布达标。

**人工配合**:我分拣,首轮约 1.5~2 万张重点张数,5~8 小时;后续轮次模型更准,只看 mid/lo,工作量递减。

---

## 任务 4:标签噪声检测 `scripts/clean_labels.py`

**做什么**:我分拣完成后,自动复查可能被我放错的样本,双保险。

**功能需求**:
1. 对 `real_cls_v{N}` 用 cleanlab(交叉验证 + 当前模型概率)找出"标签可疑"样本,输出可疑清单 HTML(缩略图 + 现标签 + 模型认为的标签 + 可疑分)
2. 我确认后,脚本按我编辑过的清单(csv 标记 keep/move/drop)批量执行修正
3. 修正后输出 `real_cls_v{N}` 的最终 MANIFEST(各类数量、尺寸分布、来源分布、噪声修正记录)

**验收标准**:对故意放错 50 张的测试目录,可疑清单召回 ≥80%;批量修正操作正确无误。

**人工配合**:看可疑清单做 keep/move/drop 决定,通常 <30 分钟/轮。

---

## 任务 5:训练与评测管线 `scripts/train.py` + `scripts/run_eval.py`

**做什么**:分类器训练入口与一键评测。

**train.py 功能需求**:
1. 模型:timm 加载 mobilenetv3_small_100 与 efficientnet_b0,输入 96×96,29 类输出;`--arch` 切换
2. 数据:混合集清单式组装(合成:真实可配比,默认预训练阶段纯合成、微调阶段 真实:合成=1:2);类别均衡采样
3. **训练侧增广(与 Phase 1 合成增广衔接,注释说明分工)**:核心是"先缩小到 12~40px 再放回 96"的退化增广(真实数据侧仍要做,概率 0.5);随机裁切偏移(模拟检测框抖动 ±10%);RandAugment 轻量配置;**禁止水平翻转**(条子/万字翻转后语义会错)
4. 损失:CrossEntropy + label smoothing 0.1;**混淆对强化**:实现一个可配置的"难对采样器",对 configs 里声明的混淆对(t4/t5、t6/t9、t2/t3、万组)按 1.5~2 倍频率采样
5. 两段预设 `--preset synth_pretrain / real_finetune`;实验目录规范同 Phase 2(自包含快照)
6. 训练中按 epoch 在一个 dev crop 集(从 real_cls 中切 5% 留出,脚本自动管理且与训练划分固定)上报指标

**run_eval.py 功能需求**:
1. `--dataset dev|frozen|<path>`,底层调 Phase 0 eval_classification.py;输出总体/每类/尺寸分桶准确率、29×29 混淆矩阵、Top10 混淆对
2. frozen 评测加确认提示 + `runs/frozen_eval_log.csv` 留痕(同 Phase 2 纪律)
3. 双架构对比模式:两个权重同集评测,输出并排对比表

**验收标准**:两架构在合成集上训练跑通且 dev 曲线正常;固定 seed 可复现;禁翻转等关键约束有单测防回归(随机抽训练 batch 检查无翻转样本)。

---

## 任务 6:置信度校准与弃权机制 `scripts/calibrate.py`(Phase 4 投票质量的地基)

**做什么**:让分类器的置信度"说真话",并选出弃权阈值。

**功能需求**:
1. 在 dev crop 集上做 **temperature scaling** 校准,输出校准前后的可靠性曲线(reliability diagram)与 ECE 指标
2. **弃权阈值扫描**:校准后,扫描置信度阈值,输出"接受准确率 vs 弃权率"曲线;自动标出满足"接受准确率 ≥99.5%"的最低弃权率工作点
3. 分尺寸桶分别给出建议阈值(小牌桶允许更高弃权率,反正有多帧投票兜底)
4. 校准温度值与建议阈值写入一个 `calibration.json`,随模型导出物一起交付 Phase 4

**验收标准**:校准后 ECE 显著下降(报告呈现);工作点选择逻辑有单测;calibration.json 字段齐全有文档。

---

## 任务 7:模型导出 `scripts/export_model.py`

**做什么**:导出 ONNX 给 Phase 4/5。

**功能需求**:
1. ONNX(opset 12+,固定 96×96,支持动态 batch——分类是批量推一帧内所有 crop,动态 batch 必须保留);可选 FP16
2. 精度对齐:1000 张 crop 上 .pt 与 ONNX 的 argmax 一致率 ≥99.9%、概率最大差 <0.005
3. 调 Phase 0 benchmark_fps.py 出 CPU/GPU 延迟报告(batch 1/8/16/32)
4. 交付物目录:onnx + calibration.json + 输入预处理规格(resize 方式/归一化均值方差/颜色通道顺序,精确到数值)+ 29 类标签顺序表 + 对齐与速度报告

**验收标准**:对齐达标;规格文档完整到下游零沟通集成。

---

## 全局技术约束

- Python 3.10+;核心依赖:torch, timm, opencv-python, albumentations, cleanlab, onnx, onnxruntime, pyyaml, tqdm, scikit-learn, matplotlib
- 单卡 24GB 上限;分类训练很轻,单轮全量训练应 <2 小时
- 实验自包含规范、复用 Phase 0/1/2 代码(经 paths.yaml)、长任务断点续跑——均同 Phase 2 要求
- pytest:数据清单组装、难对采样器、禁翻转约束、校准与阈值选择逻辑必须有单测

## 执行顺序与依赖

任务 1 → 任务 2 → 任务 5(纯合成 synth_pretrain + dev 评测)→ 任务 3(首轮收割,用合成版模型预分类)→(我分拣)→ 任务 4 →任务 5(real_finetune 双架构)→ 任务 6 → run_eval frozen 里程碑 → 不达标则:任务 3 二轮收割(更准的预分类)→ 循环 → 达标后任务 7 导出

## 我(人类)负责的部分,你不要尝试做

- crop 分拣(首轮 5~8 小时,后续递减)与 clean_labels 可疑清单裁决
- 重点混淆对文件夹的二次人工复核
- frozen 评测放行确认;双架构最终选型决定
- 最终验收:对照开头指标,决定分类器是否毕业进入 Phase 4
