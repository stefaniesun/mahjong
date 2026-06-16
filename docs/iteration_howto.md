# 预标注器滚动迭代操作手册

本手册用于每一轮训练更强的麻将牌面预标注器。目标不是一次性训练到最优，而是让“真实标注数据 + Roboflow 数据”反复滚动，逐轮提升牌池、小目标、斜牌、遮挡牌的框选能力。

## 目录与配置

核心配置：

- `configs/paths.yaml`：真实标注、Roboflow、数据集、训练、导出、评测产物路径。
- `configs/train_iter.yaml`：当前轮版本号、真实数据有效占比、训练参数。

关键目录：

- `data/labeled/`：你持续累加的 X-AnyLabeling 精标图片与同名 `.json`。
- `datasets/mix_v{N}/`：第 `N` 轮组装出的 YOLO 单类混合集。
- `datasets/real_val/`：固定真实验证集，跨轮稳定，只扩充不打乱。
- `runs/prelabeler_v{N}/`：第 `N` 轮训练产物。
- `output/prelabeler_eval/`：版本对比评测报告。
- `output/prelabeler_exports/`：ONNX 与最新模型入口。
- `output/prelabels/`：给下一批图片生成的 X-AnyLabeling 预标注。

## 每一轮标准流程

### 1. 追加真实标注

把新标注好的博主图片和同名 `.json` 放进：

```powershell
D:\mahjong\data\labeled\
```

要求：

- 只追加，不覆盖旧数据。
- 保留能识别博主/来源的目录或文件名，方便后续按博主分层留出验证集。
- X-AnyLabeling 标签可以是 `w1~w9`、`t1~t9`、`b1~b9`、`unknown`；早期误标 `back` 会被丢弃，不参与训练。

### 2. 组装本轮混合集

确认 `configs/train_iter.yaml` 里的 `version` 是本轮版本号，例如第 1 轮：

```yaml
version: 1
real_effective_ratio: 0.35
```

然后运行：

```powershell
py scripts/prepare_data.py --paths configs/paths.yaml --config configs/train_iter.yaml
```

它会做这些事：

- 将真实标注压平成单类 `tile_face`。
- 将 Roboflow 27 类压平成单类 `tile_face`。
- 按博主稳定留出 `real_val`，旧图划分不变，新增图只增量进入 train 或 val。
- 用清单重复采样真实图，让真实数据有效占比接近 `real_effective_ratio`。
- 输出 `datasets/mix_v{N}/data.yaml` 与 `prepare_report.json`。

检查报告重点：

- 真实图总数、新增数是否合理。
- 异常 label 清单是否为空；如果有异常，先回 X-AnyLabeling 修正。
- 实际有效配比是否在目标值 ±3% 内。
- `real_val` 数量是否稳定增长。

### 3. 训练本轮预标注器

```powershell
py scripts/train_prelabeler.py --paths configs/paths.yaml --config configs/train_iter.yaml
```

训练产物在：

```text
runs/prelabeler_v{N}/
```

重点看：

- `weights/best.pt` 是否生成。
- 训练 loss 是否下降。
- `real_val` 的 mAP50、recall、小目标召回是否正常。

如果显存不足：

1. 优先降低 `batch` 或设为更小整数。
2. 再考虑把 `imgsz` 从 `960` 降到 `768`。
3. 小目标牌池很多时，尽量不要过早降低 `imgsz`。

### 4. 对比新版与上一版

```powershell
py scripts/eval_prelabeler.py --paths configs/paths.yaml --config configs/train_iter.yaml
```

报告会在固定 `real_val` 上比较本版与上一版，重点看：

- 整体 recall / mAP50。
- `<20px`、`20~40px`、`>40px` 尺寸桶召回。
- 小目标桶是否提升；牌池主要落在小目标桶。
- 可视化图里牌池是否明显少漏框。

判断建议：

- 小目标召回明显提升：继续按当前策略迭代。
- 整体提升但小目标没动：提高 `imgsz`、增加真实牌池图、适当提高 `real_effective_ratio`。
- 新版比旧版退化：降低真实占比或从上一版权重微调重训。

### 5. 导出 ONNX

```powershell
py scripts/export_prelabeler.py --paths configs/paths.yaml --config configs/train_iter.yaml
```

导出后会生成：

```text
output/prelabeler_exports/prelabeler_v{N}.onnx
output/prelabeler_exports/prelabeler_latest.onnx
```

`prelabeler_latest.onnx` 是固定入口，后续预标注命令不用每轮改模型路径。

### 6. 给下一批图片生成预标注

把下一批未标注图片放到一个新目录，例如：

```text
data/to_label/batch_002/
```

运行：

```powershell
py scripts/make_prelabel.py --input-root data/to_label/batch_002 --model output/prelabeler_exports/prelabeler_latest.onnx --conf 0.25
```

输出是每张图片旁边的同名 `.json`，可直接用 X-AnyLabeling 打开。

预标注策略：

- 默认 label 是 `tile_face`，你在 X-AnyLabeling 里第二遍改成 `w/t/b/unknown` 细类。
- 阈值 `conf=0.25` 偏保守，宁可多框一点，人工删掉；不要漏牌池小牌。

### 7. 回到第 1 步

校正完这一批预标注后，把它们作为新的真实标注追加回 `data/labeled/`，然后把 `configs/train_iter.yaml` 的 `version` 加 1，进入下一轮。

## 每轮大约需要多少数据

推荐节奏：

| 阶段 | 真实图数量 | 每轮新增建议 | 重点 |
|---|---:|---:|---|
| v1 起步 | 70~150 | 50~100 | 补足牌池、小牌、遮挡、斜牌 |
| v2~v4 | 150~400 | 100~150 | 多博主、多画质、多桌面样式 |
| v5+ | 400~800 | 100~200 | 针对漏检场景补困难样本 |
| 稳定期 | >800 | 按需 | 只补新增域或失败案例 |

## 真实数据有效占比怎么调

`real_effective_ratio` 是混合训练采样中真实数据的目标占比，不是简单图片数量占比。

| 真实图数量 | 建议 `real_effective_ratio` | 说明 |
|---:|---:|---|
| <150 | 0.30~0.35 | 真实数据少，靠 Roboflow 补外观多样性，同时上采样真实牌池 |
| 150~400 | 0.40~0.55 | 真实数据逐渐成为主力 |
| 400~800 | 0.60~0.75 | Roboflow 退为辅助 |
| >800 | 0.80~1.0 | 可考虑基本脱离 Roboflow，以真实域为主 |

调参原则：

- 牌池漏检多：提高真实占比，补更多牌池标注。
- 手牌清晰牌退化：降低真实占比或保留更多 Roboflow。
- 过拟合明显：降低真实占比、增加真实来源多样性。

## 什么时候提升会变慢

通常会在这些时候出现收益递减：

- 同类场景已覆盖充分，新增图片与旧图片高度相似。
- 主要错误来自极小、强糊、严重遮挡，已经接近图片信息上限。
- `real_val` 小目标召回连续 2~3 轮提升低于 1~2 个百分点。
- 可视化里剩余漏检需要人眼也反复确认。

这时不要盲目堆重复数据，应改为补“失败案例”：夜局、糊帧、斜牌、远距离牌池、遮挡叠牌、不同博主画质。

## 什么时候可以停止迭代

满足以下多数条件即可暂时停止：

- X-AnyLabeling 中大部分牌面框已自动给出，你主要只是在改类别和少量删补框。
- 固定 `real_val` 上整体 recall 稳定，尤其 `<20px` 与 `20~40px` 小目标召回连续两轮提升很小。
- 新增一批 100 张图片时，人工补漏框数量明显少于改类别数量。
- 继续训练带来的标注省时不足以抵消训练和检查成本。

停止不是永久停止。后面遇到新博主、新画质、新桌面样式或明显漏检场景，再追加数据开启下一轮即可。
