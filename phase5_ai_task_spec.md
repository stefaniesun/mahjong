# Phase 5 需求规格:麻将识别项目 — 量化与端侧部署

> 本文档交给 AI 编码助手(opencode / codex / Claude Code)执行。
> 请严格按任务顺序实现,每个任务有明确的输入、输出和验收标准。
> 本 Phase 依赖:Phase 2/3 的 ONNX 交付目录、Phase 4 的 `mahjong-rt` 参考实现(含确定性录制数据与视频评测集)、Phase 0 评测工程。
> 注意:本 Phase 涉及真机,部分任务 AI 只能交付"可在真机执行的工具与流程",实际插线跑机由我完成。

---

## 项目背景(供 AI 理解上下文)

四川麻将识别系统已在 PC 上达标(检测 + 分类 + 时序融合,Phase 4 参考实现)。本 Phase 把它部署到端侧。

**部署形态(双轨,按硬件实测决定主用哪条)**:
- **A 轨(首选验证):眼镜采集 + 手机推理**。眼镜仅做视频流推送,Android 手机跑完整流水线。工程最稳,先打通它作为保底
- **B 轨(理想形态):眼镜端原生推理**。取决于眼镜 SoC 的 NPU 能力,在 A 轨跑通后评估

**最终验收目标**:
- 端到端延迟(摄像头曝光→事件输出)≤100ms(p95)
- 流水线吞吐 ≥15fps(端侧目标,低于 PC 的 30fps,靠 Phase 4 的降频调度参数适配)
- 量化精度回退:冻结测试集端到端准确率相比 PC FP32 下降 ≤0.5%;Phase 4 视频级四项指标全部仍 PASS
- 稳定性:连续运行 30 分钟无崩溃、无热降频导致的 fps 腰斩;手机整机功耗增量 <3W

**硬件信息由我提供(任务 1 的前置输入)**:眼镜型号/SoC/系统、可用的视频输出方式(RTSP/UVC/私有 SDK)、目标手机型号/SoC。AI 实现时按我提供的具体硬件选择后端,但**代码结构必须后端可插拔**,不许写死单一厂商。

### 仓库目录结构(任务 1 中创建)

```
mahjong-edge/
├── configs/
│   ├── paths.yaml
│   ├── device_profile.yaml    # 我填写的目标硬件档案
│   └── deploy.yaml            # 端侧流水线参数(由 Phase 4 pipeline.yaml 派生)
├── quant/
│   ├── build_calib_set.py     # 量化校准集构建
│   ├── quantize.py            # PTQ 量化(多后端)
│   └── verify_quant.py        # 量化精度回归
├── backends/                  # 推理后端适配层
│   ├── ort/                   # ONNX Runtime(基线,必做)
│   ├── ncnn/                  # NCNN(Android 通用,必做)
│   └── vendor/                # QNN/SNPE/RKNN 等厂商后端(按 device_profile 选做)
├── cpp/                       # 后处理核心 C++ 移植
│   ├── include/ src/          # tracker/voter/state_machine/zones
│   ├── tests/                 # 含金标文件交叉验证
│   └── bindings/              # Python 绑定(pybind11,用于在 PC 上跑同一套 C++ 代码做验证)
├── android/                   # A 轨参考 App(Android Studio 工程)
├── scripts/
│   ├── bench_device.py        # 端侧基准测试编排(adb 驱动)
│   ├── soak_test.py           # 30 分钟稳定性烤机
│   └── eval_on_device.py      # 端侧精度回归编排
├── docs/
│   ├── deploy_architecture.md
│   └── device_runbook.md      # 给我的真机操作手册
├── requirements.txt
└── README.md
```

---

## 任务 1:硬件档案 + 部署架构设计

**做什么**:定义 `device_profile.yaml` 模板(SoC/NPU/内存/系统版本/视频接入方式/厂商 SDK 可用性等字段,我来填),然后写 `docs/deploy_architecture.md`。

**deploy_architecture.md 必须说清**:
1. A 轨链路:眼镜 → 视频流(按 device_profile 确定协议,优先低延迟方案,目标采集传输 ≤40ms)→ 手机 App:解码 → 推理流水线(C++ 核心)→ 叠加显示/事件输出;各环节延迟预算分解表(总预算 100ms 如何分)
2. B 轨可行性评估清单:眼镜 NPU 算力、可用运行时、内存上限,对照两个模型的算量(给出 GFLOPs 估算)得出结论模板,A 轨完成后填数
3. 后端抽象接口:`IDetector` / `IClassifier` 的 C++ 纯虚接口(load/infer/spec 查询),ort/ncnn/vendor 三种实现共用;后处理与推理后端完全解耦
4. **模型适配注意**:确认 P2 头、SiLU、解码层等算子在目标后端的支持情况;不支持的算子列出替换方案(如解码移出模型在 C++ 后处理做——推荐默认就这么做,导出"裸输出"版 ONNX)

**验收**:我填完 device_profile 后,文档能直接推导出本项目实际采用的后端与链路,无悬而未决项。

---

## 任务 2:量化校准集与 PTQ `quant/`

**做什么**:INT8 量化两个模型,并建立严格的精度回归门禁。

**功能需求**:
1. **`build_calib_set.py`**:从真实数据构建校准集——检测模型用 300~500 张真实帧(博主均衡、尺寸场景均衡、含暗光/模糊样本;从 real_det_v* 训练集抽,绝不用冻结测试集);分类模型用 2000 张真实 crop(类均衡 + 尺寸桶均衡);输出校准集 MANIFEST
2. **`quantize.py`**:对两个模型执行 PTQ,按后端分别产出:ORT 静态量化(QDQ 格式)、NCNN int8(table 生成)、厂商后端(按 device_profile,调用其转换工具链,封装成一条命令);支持逐层敏感度分析(`--sensitivity`):逐层回退 FP16 测精度,输出"哪几层量化伤害最大"报告,需要时混合精度(敏感层保留 FP16)
3. **`verify_quant.py`**:量化模型 vs FP32 原模型,(a) 数值层面:同输入输出余弦相似度;(b) 任务层面:调 Phase 0 评测在冻结测试集跑端到端,**门禁:下降 ≤0.5% 才放行**;不达标自动建议(敏感度报告 + 混合精度重量化)
4. QAT 预案:若混合精度仍不达标,文档化 QAT 流程(回 Phase 2/3 训练仓库,插入量化感知微调 10~15 epochs),本期先不实现代码,写清触发条件与操作步骤

**验收标准**:两模型三后端的 INT8 产物齐备;门禁流程跑通且留痕(沿用 frozen_eval_log 纪律);敏感度分析报告可读。

---

## 任务 3:后处理核心 C++ 移植 `cpp/`(本 Phase 工程核心)

**做什么**:把 Phase 4 的 tracker(含 GMC)/ voter / state_machine / zones / 事件协议移植为无依赖 C++17 库(只依赖 OpenCV core/video + nlohmann_json),供 Android 与未来眼镜端共用。

**功能需求**:
1. 模块与接口一一对应 Phase 4 Python 实现;pipeline.yaml 的参数体系原样支持(yaml 解析)
2. **金标文件交叉验证(本任务的灵魂)**:Phase 4 的确定性录制数据(原始帧 + 每帧检测/分类中间结果 + 事件流输出)作为金标——C++ 实现读取同样的中间结果输入,产出的事件流必须与 Python 版逐事件一致(时间戳字段允许 ±1 帧容差,类别/状态/track 归属必须全等)。把 ≥5 段录制做成自动化金标测试,进 CI
3. pybind11 绑定:PC 上可用 Python 调 C++ 实现,直接复用 Phase 4 的 eval_video.py 验证 C++ 版视频级指标
4. CMake 构建:Linux(PC 验证)与 Android NDK(arm64-v8a)双目标;无任何平台特定代码散落,平台差异收敛到适配层
5. 性能:C++ 后处理全链路(40 目标)单帧 <3ms(arm 大核单核)

**验收标准**:金标测试全绿;eval_video.py 跑 C++ 绑定版,四项视频指标与 Python 版一致;Android NDK 编译产物在真机可加载(任务 5 联调确认)。

---

## 任务 4:推理后端适配层 `backends/`

**做什么**:实现 IDetector/IClassifier 的多后端版本。

**功能需求**:
1. **ORT 后端**(必做,基线):加载 QDQ INT8 模型,XNNPACK/NNAPI EP 可配;裸输出解码(YOLO 解码 + NMS)在 C++ 实现并单测(对照 Phase 2 导出规格文档)
2. **NCNN 后端**(必做,Android 主力候选):模型转换脚本(onnx→ncnn,含量化 table)、Vulkan GPU 开关、解码实现复用同一套 C++ 解码代码
3. **厂商后端**(按 device_profile 选做一个):QNN/SNPE(高通)或 RKNN(瑞芯微)的转换 + 运行时封装,NPU 委托
4. **逐后端数值对齐测试**:固定 50 张输入,各后端输出与 ORT FP32 基准比对,框级一致率 ≥98%(量化容差内),分类 argmax 一致率 ≥99.5%;不达标的后端禁止进入集成
5. 统一的后端选择配置与运行时回退链(vendor 失败→ncnn→ort)

**验收标准**:PC 上 ORT/NCNN 对齐测试通过;真机上目标后端对齐测试通过(配合任务 5);任一后端可通过 deploy.yaml 一行切换。

---

## 任务 5:A 轨 Android 参考 App `android/`

**做什么**:打通"眼镜推流 → 手机全流水线"的最小可用 App。

**功能需求**:
1. 视频接入:按 device_profile 实现(RTSP/UVC/SDK 其一),硬解码(MediaCodec),解码后零拷贝进推理线程;同时支持手机自身摄像头作为开发调试源(没眼镜也能开发)
2. 集成任务 3/4 产物:JNI 封装,流水线调度沿用 Phase 4 架构(检测隔帧 + 分类按需),参数读 deploy.yaml
3. UI:实时预览 + 叠加(确认牌的中文名/zone 着色,与 Phase 4 可视化一致)+ 性能面板(各阶段耗时/fps/丢帧/机身温度读数);事件流写本地 jsonl(供 eval 与上层应用)
4. **诊断模式**:一键录制 30 秒"原始帧+中间结果"包,格式与 Phase 4 录制兼容——线上问题可拉回 PC 用 replay.py 复现(这是 Phase 6 badcase 闭环的入口,接口先留好)
5. App 不做任何联网上传;录制仅存本地由我手动导出

**验收标准**:手机摄像头源下全链路真机运行,叠加显示正确;眼镜推流接入后(我操作)端到端延迟实测 ≤100ms(用拍秒表屏幕法测,操作写进 runbook);诊断包可被 PC replay.py 加载。

---

## 任务 6:端侧基准与稳定性 `scripts/bench_device.py` + `soak_test.py`

**做什么**:把真机测试标准化成可重复执行的编排脚本(adb 驱动)。

**功能需求**:
1. **bench_device.py**:推送固定测试视频到手机 → App 以确定性模式处理 → 拉回逐阶段耗时与事件流 → 生成报告:各阶段 p50/p95、端到端延迟分布、fps、与 PC 基准的对照表;支持多配置批跑(后端×分辨率×检测间隔 N 的矩阵)
2. **soak_test.py**:30 分钟连续运行,每 30 秒采样 fps/各阶段耗时/电池温度/CPU 频率(检测热降频)/内存占用,输出时序曲线报告;判定:fps 衰减 >20% 或崩溃即 FAIL,并给出瓶颈归因(温度曲线 vs 频率曲线对照)
3. **eval_on_device.py**:把冻结测试集与视频评测集喂给真机流水线(确定性模式),结果回传 PC 调 Phase 0/4 评测脚本,产出"端侧最终精度报告",对照验收红线逐项 PASS/FAIL

**验收标准**:三个脚本一条命令出报告;报告含与 PC FP32 基准的全量对照;数据足以支撑"哪个配置组合上线"的决策。

**人工配合**:我负责插线、装 App、跑测试时盯设备;秒表法延迟实测;按 runbook 操作即可。

---

## 任务 7:B 轨可行性评估 + 发布打包

**做什么**:
1. **B 轨评估**:按任务 1 的清单,用实测数据(模型在眼镜 SoC 同档芯片上的 bench,若眼镜可直接跑则直接测)填写结论:眼镜端原生推理在当前模型规模下 可行/需进一步压缩(给出压缩路线:更小输入分辨率/检测间隔加大/模型蒸馏)/不可行(维持 A 轨);输出一页决策备忘录
2. **发布打包**:`release_v{N}/` 目录 = APK + 全部模型产物(各后端)+ deploy.yaml + calibration.json + 事件协议文档 + 端侧精度/性能/烤机三份报告 + 完整版本溯源(各 Phase 数据与模型版本号、git hash)——拿这个目录可以完整复现与回滚
3. `docs/device_runbook.md` 终稿:从拿到新手机到跑通验收的全部人工步骤

**验收标准**:发布目录自包含;runbook 我能照做;B 轨备忘录给出明确结论与下一步。

---

## 全局技术约束

- C++17 + CMake;Android NDK r25+,minSdk 29;Kotlin 做 App 壳,核心逻辑全在 C++ 层
- Python 侧依赖:onnxruntime, ncnn 转换工具, opencv-python, pyyaml, adb 编排用 pure python(不引 Appium 等重框架)
- 金标交叉验证与后端对齐测试进 CI(PC 可跑部分);真机测试以脚本编排保证可重复
- 所有量化/转换产物带 MANIFEST(源模型 hash、校准集版本、工具链版本)
- 安全与隐私:App 无网络权限申请;录制数据不离机,导出靠 adb 手动

## 执行顺序与依赖

任务 1(我填硬件档案)→ 任务 2 与任务 3 并行 → 任务 4 → 任务 5(先手机摄像头源,后接眼镜流)→ 任务 6(基准/烤机/精度回归)→ 不达标则回任务 2(混合精度/QAT)或调 deploy.yaml 调度参数 → 任务 7

## 我(人类)负责的部分,你不要尝试做

- 填写 device_profile.yaml(眼镜与手机的硬件信息、视频接入方式),提供厂商 SDK/文档
- 全部真机操作:装 App、连眼镜、跑 bench/烤机、秒表法延迟实测
- 实际佩戴体验评估(画面延迟体感、发热体感)——指标 PASS 不等于能用,体感我把关
- 最终验收与 A/B 轨路线决策
