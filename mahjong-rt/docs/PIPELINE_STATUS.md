# 麻将视频识别流水线：现状与产出格式

> 本文给外部工具开发者使用。目标是让你不看流水线源码,也能读懂产出、写出独立的验证工具。
> 所有格式均已对照实际文件核实。路径以仓库根目录 `D:/mahjong/` 为基准。

---

## 1. 系统是什么

第一人称视角(AI 眼镜)的**四川麻将**实时识别。两段式模型 + 时序融合:

```
视频帧
  ├─ 检测器 (YOLO11s, 单类 tile_face)     → 牌面框
  ├─ 分类器 (EfficientNet-B0, 27 类)       → 每个框是几万/几条/几筒
  ├─ 跟踪器 (ByteTrack + 全局运动补偿)     → 跨帧认出"这是同一张牌"
  ├─ 投票器 (滑窗加权投票 + 滞回)          → 多帧融合成稳定判定
  ├─ 状态机 (TENTATIVE→CONFIRMED→LOST)     → 对外只暴露已确认结果
  └─ 区域划分 (纯几何启发式)               → 每张牌属于哪家
                                            ↓
                                       事件流 / 帧快照
```

**牌背不检测**(牌墙、立牌一律当背景),**字牌花牌不在类别体系内**。

### 关键设计约束

* 牌是静止的,**相机一直在动**(头部转动)。跟踪器因此做全局运动补偿:用稀疏光流估计
  帧间单应矩阵,先把所有 track 的预测框按全局运动变换,再做 IoU 关联。
* 关联同时使用**外观相似度**:密集排列的牌框大量重叠,纯 IoU 分不清谁是谁。
  外观描述子是分类器的 27 维类别概率分布。
* 对外类别由**状态机持有**,单帧结果永远不直接暴露。这是"不闪烁"的实现基础。

---

## 2. 当前实测指标

在试点评测集(3 段 × 30 秒,21 个人工标注检查点)上:

| 指标 | 实测 | 目标 | 状态 |
|---|---|---|---|
| 检查点类别准确率 | 98.07% | ≥99.5% | 未达标 |
| 检查点召回 | 98.60% | — | — |
| **闪烁率(改判次数/牌)** | **0.0197** | <0.05 | **达标** |
| ID 切换率 | 17.2% | — | 偏高 |
| 确认延迟 | 未测(缺时间戳真值) | ≤0.5s | — |
| 吞吐 | 0.16 fps (CPU) | ≥30 fps | 硬件限制 |

单帧模型自身的权威指标(在各自的冻结测试集上,与上表不是一回事):

| 模型 | 指标 |
|---|---|
| 检测器 | 小牌(<20px)召回 98.4%,整体精确率 97.7% |
| 分类器 | 27 类 top-1 **99.51%**;小牌 98.84%,大牌 99.90% |

### 已知局限(写验证工具时必须知道)

1. **评测跑在 stride=3(等效 10fps)**。真实 30fps 下帧间运动小 3 倍,跟踪指标会明显好转。
   当前所有 ID 切换数字都是在这个放大难度下测的。
2. **只有 3 段片段**,单段准确率从 93.7% 到 98.8%,统计噪声大。
3. **评测片段的分类准确率偏乐观**:分类器训练时用过这些视频的部分帧。分类的权威数字看
   冻结测试集的 99.51%,不看评测集的。
4. **检测器有合并框问题**:相邻两张牌被框成一个,占全部漏检的 30.6%。这是检测层的问题,
   时序融合救不了(每帧都会同样合并)。
5. **区域划分**:在 923 个人工标注框上图级 5 折交叉验证,整体 **97.9%**,
   逐区域召回 手牌 100 / 上家 100 / 牌池 97.6 / 下家 96.0 / **对家 92.3**。
   对家曾长期卡在 71%,靠"把碰牌当整组判"解决(见 `docs/OPEN_PROBLEM_zones.md`)。
   注意这是**单帧静态图**上的数字;视频上还叠了一层按 track 生命期的加权投票。

---

## 3. 模型文件

| 文件 | 说明 |
|---|---|
| `output/eval_real_v1/best.pt` | 检测器,Ultralytics YOLO11s,单类 `tile_face` |
| `output/eval_real_v1/best.onnx` | 同上的 ONNX(opset 12,固定 imgsz=960) |
| `output/cls_final_v2/best.pt` | 分类器 PyTorch 权重 |
| `output/cls_final_v2/best.onnx` | 分类器 ONNX(输入 `images` (N,3,96,96),输出 `logits` (N,27)) |
| `output/cls_final_v2/meta.json` | 分类器的类别表与预处理参数 |

`meta.json`:

```json
{
  "classes": ["b1","b2",...,"w9"],   // 27 个,顺序即输出通道顺序
  "imgsz": 96,
  "arch": "efficientnet_b0",
  "mean": [0.485, 0.456, 0.406],
  "std":  [0.229, 0.224, 0.225]
}
```

**分类器预处理**:按框外扩 8% 裁剪 → resize 到 96×96 → BGR 转 RGB → /255 →
减 mean 除 std → CHW。输出 logits 需自行 softmax。

**牌名编码**:`w1`~`w9` = 一万~九万,`t1`~`t9` = 一条~九条,`b1`~`b9` = 一筒~九筒。
注意 **`t` 是条、`b` 是筒**(不是 bamboo)。类别表里没有 `unknown`,也没有牌背。

---

## 4. 产出文件格式

### 4.1 事件流 `events_{clip}.jsonl`

每行一个 JSON 对象,三种事件类型 + 每帧快照。**这是最主要的验证对象。**

```jsonc
// 一张牌首次被确认
{"track_id":1, "label":"w6", "confidence":1.0,
 "bbox":[785.8, 415.0, 79.8, 89.5],        // [x, y, w, h] 像素,原始帧坐标系
 "zone":"my_hand", "frame_idx":2, "ts":0.202, "type":"tile_confirmed"}

// 已确认的牌改判(滞回条件满足后才会发生;这个数越少越好)
{"track_id":80, "label":"b6", "previous_label":"b7", "confidence":0.824,
 "bbox":[216.9,182.2,59.5,20.8], "zone":"seat_left",
 "frame_idx":113, "ts":11.41, "type":"tile_updated"}

// 牌消失(超出 track buffer)
{"track_id":.., "label":"..", "last_bbox":[..], "frames_tracked":N,
 "frame_idx":.., "ts":.., "type":"tile_lost"}

// 每帧全量快照(可关闭)
{"frame_idx":0, "ts":0.0,
 "tiles":[{"track_id":1,"label":"w6","confidence":1.0,
           "bbox":[..],"zone":"my_hand","state":"CONFIRMED",
           "visible":true}, ...],      // visible=false: 在桌上但这帧被挡住,别丢
 "counts_by_zone":{"my_hand":13,"river":33,...},
 "stats":{"detections":79,"tracks":79,"gmc_ok":0,"gmc_fail":1},
 "type":"frame_summary"}
```

**字段语义**

* `track_id`:同一张物理牌在其生命周期内保持不变。ID 变了就意味着跟踪断过。
* `bbox`:**xywh**,不是 xyxy。像素坐标,对应原始帧尺寸(见 MANIFEST 的 fps/尺寸)。
* `ts`:秒,片段内相对时间(不是原视频时间)。
* `zone`:`my_hand` / `river` / `seat_left` / `seat_across` / `seat_right` /
  `meld_area` / `opponent_wall` / `unknown_zone`。
  座位按麻将轮转:**下家在右、对家在对面、上家在左**。
* `frame_idx`:片段内帧号。注意评测跑 stride 时,只有被处理的帧才有事件。

`frame_summary` 里 `tiles` 含 **CONFIRMED 与 OCCLUDED** 两种状态,TENTATIVE 不在其中。

**`visible` 字段很重要**:`false` 表示"这张牌还在桌上,只是当前这一帧被挡住了"。
**消费方应该保留它,不要当成消失**。麻将牌放下就不动了,手挥过去挡一下不等于牌被拿走——
真正离场只有碰杠取牌和一局结束。之前这里的实现把被遮挡的牌直接从世界状态里剔除,表现就是
"有人伸手过桌面时牌成片闪没",实测输出骤降 28 次;改成保留后降到 2 次,ID 切换也从
28.3% 降到 17.2%。

### 4.2 模型输出录制 `recordings2/{clip}.npz`

一次模型推理的完整输出,用于免推理重放。numpy 压缩格式,扁平数组 + 索引:

| 键 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `counts` | (F,) | int32 | 每帧的检测框数,用于切分下面的扁平数组 |
| `frame_index` | (F,) | int32 | 片段内帧号 |
| `timestamp` | (F,) | float32 | 秒 |
| `homography` | (F,3,3) | float32 | 与上一记录帧之间的全局运动矩阵 |
| `boxes` | (N,4) | float32 | **xyxy** 像素(注意与事件流的 xywh 不同) |
| `scores` | (N,) | float32 | 检测置信度 |
| `labels` | (N,) | int16 | 类别索引,查 `meta[1]` 的类别表 |
| `confidences` | (N,) | float32 | 分类置信度(即 probs 的最大值) |
| `probs` | (N,27) | float16 | 完整类别概率分布,也用作外观描述子 |
| `meta` | (6,) | object | `[clip名, "b1\|b2\|...\|w9", 宽, 高, fps, stride]` |

切分方式:`offsets = concat([[0], cumsum(counts)])`,第 i 帧的框是 `boxes[offsets[i]:offsets[i+1]]`。

**分类结果是按检测框记录的,不是按 track**。这样改跟踪参数时不必重跑模型。

### 4.3 评测集 `output/video_testset_pilot/`

```
MANIFEST.json          片段与检查点索引
checkpoints/           21 张检查点图 + 同名人工标注 JSON
clips_full/            1280x720 片段视频(mp4v 编码,浏览器放不了)
recordings2/           上面 4.2 的录制
events_gt.json         出现事件时间戳(当前为空,未标注)
eval_run2/             评测输出
```

`MANIFEST.json`:

```jsonc
{
  "clips": [{
    "name": "clip01_7507945925261200",
    "video": "..\\data\\raw_videos\\...\\原视频.mp4",   // 注意可能是乱码,别依赖
    "clip_file": "clips_full/clip01_....mp4",
    "start": 285.0, "end": 315.0,      // 在原视频中的秒区间
    "fps": 30.0,
    "checkpoints": [{
      "file": "clip01_..._f000000.jpg",
      "clip_frame": 0,          // 片段内帧号 —— 与事件流的 frame_idx 对齐用这个
      "clip_time": 0.0,
      "source_frame": 8427,     // 原视频帧号
      "prelabel_boxes": 41
    }, ...],
    "harvested_overlap": 0,     // 与训练用帧的重叠数,0 表示无泄漏
    "scan_score": 2.795
  }],
  "checkpoint_seconds": 5.0,
  "checkpoint_total": 21,
  "leakage": {"note": "...", "clips_with_harvested_overlap": 0}
}
```

**检查点标注**(X-AnyLabeling 格式,人工校正过,是评测真值):

```jsonc
{
  "version": "2.3.6", "flags": {},
  "shapes": [{
    "label": "w6",
    "points": [[785.47, 415.60], [865.68, 504.72]],   // 两个对角点 xyxy
    "group_id": null, "shape_type": "rectangle", "flags": {}
  }],
  "imagePath": "clip01_..._f000000.jpg",
  "imageData": null,
  "imageHeight": 720, "imageWidth": 1280
}
```

牌背未标注;看不清的标 `unknown`。

### 4.4 评测报告 `eval_run2/summary.json`

```jsonc
{
  "checkpoint": {"gt_tiles":1150, "recall":0.967, "precision":0.976,
                 "class_accuracy":0.9617, "matched":.., "class_correct":..,
                 "missed":.., "spurious":..,
                 "by_bucket": {"lt20":{"n":444,"recall":..,"class_accuracy":..},
                               "20to40":{...}, "gt40":{...}}},
  "latency": {"n_events":0, "n_matched":0, "p50":0.0, "p95":0.0, ...},
  "flicker": {"confirmed_tracks":328, "total_updates":36,
              "updates_per_tile":0.1098, "worst_tracks":[...]},
  "tracking": {"comparisons":.., "id_switches":.., "switch_rate":..},
  "performance": {"fps":0.16},
  "clips": [...],                    // 逐段同结构
  "annotated_checkpoints": 18,
  "acceptance": [{"name":"..","value":..,"target":"..","pass":true/false}]
}
```

尺寸分桶按**框的较短边**:`lt20` <20px,`20to40` 20~40px,`gt40` >40px。

### 4.5 演示页数据 `output/demo_state.json`

供可视化用的精简版(字段名缩写以省体积):

```jsonc
{
  "clip":"clip02_...", "fps":30.0, "stride":3, "w":1280, "h":720,
  "frames":[{"t":10.097, "idx":300,
             "tiles":[{"id":1,"l":"w6","z":"my_hand","b":[785,416,80,89]}]}],  // b 是 xywh
  "events":[{"t":0.202,"type":"tile_confirmed","id":1,"label":"w6",
             "prev":null,"zone":"my_hand"}]
}
```

---

## 5. 配置

`mahjong-rt/configs/pipeline.yaml` 是所有后处理参数的唯一来源。关键项:

```yaml
tracker:
  high_thresh: 0.5          # 高分检测进一级关联
  low_thresh: 0.1           # 低分检测只进二级关联,用于救糊帧
  match_thresh: 0.7         # 一级关联 IoU 门限
  match_thresh_low: 0.4
  track_buffer: 30          # 丢失后保留帧数
  gmc_enabled: true         # 全局运动补偿
  appearance_weight: 0.3    # 外观约束权重,0=纯 IoU
  appearance_momentum: 0.7

voter:
  window: 7                 # 投票滑窗
  min_conf: 0.5             # 低于此值的分类结果弃权,不入窗
  min_effective: 3          # 至少这么多次有效观测才下结论
  majority_ratio: 0.6       # 最高票占比阈值
  hysteresis: 4             # 已确认的牌要连续这么多次才允许改判

state:
  occluded_after: 2
  lost_after: 30

zones:
  enabled: true
  hand_size_ratio: 1.4      # 手牌:尺寸 ≥ 帧内中位数的 1.4 倍
  hand_min_ny: 0.60         #       且在画面下方 60% 以下
  left_max_nx: 0.28         # 上家:横向在牌池范围之外(左)
  right_min_nx: 0.78        # 下家:横向在牌池范围之外(右)
  across_max_ny: 0.28       # 对家:靠更上 + 更小区分
  across_max_size_ratio: 0.9
```

区域阈值来自 899 个人工标注框的网格搜索,不是拍脑袋定的。

---

## 6. 复现命令

```bash
cd mahjong-rt

# 跑流水线出标注视频与事件流
python scripts/run_pipeline.py --source 视频.mp4 \
  --det ../output/eval_real_v1/best.pt --cls ../output/cls_final_v2/best.onnx \
  --config configs/pipeline.yaml --out-video out.mp4 --events out.jsonl --headless

# 录制模型输出(之后改参数不必再推理)
python scripts/record_clips.py --testset ../output/video_testset_pilot \
  --det ../output/eval_real_v1/best.pt --cls ../output/cls_final_v2/best.onnx \
  --out ../output/video_testset_pilot/recordings2 --stride 3 --det-conf 0.05

# 视频级评测
python scripts/eval_video.py --testset ../output/video_testset_pilot \
  --det ../output/eval_real_v1/best.pt --cls ../output/cls_final_v2/best.onnx \
  --out ../output/eval_new --stride 3

# 单张图端到端可视化
python ../scripts/predict_image.py --image 图.jpg \
  --det ../output/eval_real_v1/best.pt --cls ../output/cls_final_v2/best.pt --out 结果.jpg
```

CPU 上检测约 2.2s/帧、分类约 1.4s/批,一段 30 秒片段跑完约 20 分钟。

---

## 7. 给验证工具的建议

如果你要写独立工具验证这条流水线,以下几条是最值得查的,按价值排序:

1. **牌数守恒**:每种牌全局最多 4 张。把 `frame_summary` 里所有区域的牌统计一遍,
   任何一类超过 4 就是确定的错误。这是最强的自动校验,不需要人工真值。
2. **手牌数量**:`my_hand` 区域应恒为 13 或 14 张(未碰杠时)。偏离即异常。
3. **track 连续性**:同一 `track_id` 的 `bbox` 在相邻帧应连续变化。跳变说明关联出错。
4. **改判合理性**:`tile_updated` 里 `previous_label` → `label` 的混淆对,应集中在
   已知易混对(w3/w2、w5/w9、b2/b4、t6/t7)。出现离谱的改判(如 w1→b9)说明有 bug。
5. **区域一致性**:同一 track 的 zone 在生命周期内不应频繁跳变(状态机已做多数投票,
   跳变说明该牌在边界附近)。
6. **与检查点真值比对**:21 个检查点有人工标注,用 IoU≥0.5 + 类别匹配即可算准确率。
   注意**每段第一个检查点在第 0 帧,那时投票器还没确认任何牌**,必须跳过(见
   `eval_video.py --warmup-frames`)。

**容易踩的坑**

* 事件流 `bbox` 是 **xywh**,录制 npz 里是 **xyxy**,别搞混。
* `frame_idx` 是片段内帧号;若跑了 stride,帧号不连续。
* `ts` 是片段内相对时间,不是原视频时间。原视频时间 = `ts` + MANIFEST 里的 `start`。
* `clips_full/` 里的视频是 mp4v 编码,**浏览器播不了**,需要 ffmpeg 转 H.264。
* 类别表里 **`t` 是条、`b` 是筒**。

---

## 8. 区域判定:不要自己重写

区域判定已经实现在 `mahjong_rt/zones.py`,阈值来自 923 个人工标注框、图级 5 折交叉验证,
整体 **97.9%**。**自己拍阈值几乎必然更差** —— 本项目前三版就是手调的,最好只有 39.7%,
连"把所有牌都判成牌池"这个笨基线(53.0%)都不如。失败原因在 `docs/tuning_findings.md`。

判定用的全是**相对量**,不用像素:

```python
size_ratio = 框的短边 / 该帧所有框短边的中位数
nx, ny     = 框中心的归一化坐标 (0~1)
nn         = 最近邻距离 / 自身宽度            # 牌池是密集的
gap        = 到"密集区"的距离 / 自身宽度       # 定缺牌单张但远离牌堆

if size_ratio >= 1.4 and ny >= 0.55:                      → my_hand     我的手牌
elif nx <= 0.30 and (nn <= 0.45 or gap >= 2.0):           → seat_left   上家
elif nx >= 0.78:                                           → seat_right  下家
elif ny <= 0.28 and (nn >= 0.55 or gap >= 2.0):           → seat_across 对家
else:                                                      → river       牌池

# 之后两道后处理:
#   1) 簇内多数投票(小簇跟随大多数)
#   2) 碰/杠整组覆盖 —— 同类相邻 >=3 张且排得整齐(组内 y 标准差/平均高 <= 0.12)
#      整组判给某一家。这条需要分类结果,是对家从 72% 提到 92% 的关键。
```

用相对尺寸而非绝对像素是刻意的:牌的尺寸就是深度线索(短边 15.9px→81.7px 对应 ny
0.39→0.81),比成中位数之后,换分辨率、换坐姿远近都不失效。v1 死在固定像素阈值上。

### 必须做时序投票,不能逐帧独立判

这一条最容易被忽略,也是"区域一团糟"最常见的原因。

单帧判定**会因镜头运动而抖**:头一转,牌的归一化坐标就变,靠近边界的牌就会翻。实测逐帧
翻转率:

| 片段 | 相邻帧同位置区域翻转 |
|---|---|
| clip01 | 11.2% |
| clip02 | 8.0% |
| clip03 | 3.3% |

正确做法是**按 track 生命期对区域做加权投票**(状态机已实现):

```python
# river 是"什么都没证明出来"时的兜底 —— 包括碰牌少检到一张、间距算不出来的那些帧。
# 判给某一家则需要正面证据。等同计票会让"没看见"的帧压过"看见了"的帧。
weight = 1.0 if zone in ('river', 'unknown_zone') else 2.0
tile.zone_votes[zone] += weight
tile.zone = max(tile.zone_votes.items(), key=lambda kv: kv[1])[0]
```

权重取值的依据(用标注图模拟带噪多帧,每张 15 个视角、漏检 15%、框抖动、2% 分类错):

| | 准确率 |
|---|---|
| 单帧独立判 | 96.7% |
| 等权多数投票 | 97.4% |
| **权重 2** | **98.4%** |
| 权重 3 | 98.5%(但漏检率高时会反转) |

真实录制上实测:1.1% 的牌从牌池移到了对家。

注意投票救不了**系统性**错误:同一张牌若因几何位置被持续判错,每一帧都会同样判错,
投票只会把错误固化。它救的是**边界抖动**——实测 81.4% 的 track 全程判定一致(投票无事可做),
18.6% 出现分歧,其中分歧几乎全是 river ↔ seat_across。

投票之后,区域变过的 track 降到 **2~8%**。如果你的工具逐帧独立判区域,复现的就是上面那个
3~11% 的原始抖动。

### 已知的弱项

逐区域召回:手牌 97.0 / 牌池 95.3 / 下家 97.8 / **上家 73.5 / 对家 71.0**。

上家和对家弱是因为它们和牌池**横向真的重叠**(对家的牌河和公共牌池在画面上挨着),属于本质
歧义。阈值再调也难大幅改善;要突破得换手段(用那 899 个标注训个几 KB 的小分类器)。

### 阈值改了怎么验证

真值在 `output/zone_annotation/zone_labels.json`(899 个框,含人工标注与启发式预测)。
回归测试 `tests/test_zones.py` 会检查准确率不低于 90%、且必须超过"全判牌池"的基线 15%。
改完跑一遍就知道变好还是变坏,不用猜。
