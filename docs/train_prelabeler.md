# 训练预标注模型手册

本手册用于训练 Phase 0 的临时预标注器。它只负责在 X-AnyLabeling 里先画出大部分牌面框，方便人工校正；它不是最终线上检测器。

## 1. 数据与类别

本地 Roboflow 数据集路径：

```text
E:/360MoveData/Users/Administrator/Desktop/Mahjong.v1i.yolov11/data.yaml
```

类别为 27 类：

- `1B~9B`：Bamboo，映射为条 `t1~t9`
- `1C~9C`：Character，映射为万 `w1~w9`
- `1D~9D`：Dot，映射为筒 `b1~b9`

注意：数据集没有牌背 `back`。预标注阶段漏掉的牌背由人工在 X-AnyLabeling 中补标为 `back`。

## 2. 安装依赖

```powershell
py -m pip install -r requirements.txt
```

如果只想单独安装训练依赖：

```powershell
py -m pip install ultralytics PyYAML
```

## 3. 核对映射表

训练前先确认 `configs/prelabel_map.yaml` 没写反，尤其是：

- `B = 条 = t*`
- `C = 万 = w*`
- `D = 筒 = b*`

可运行：

```powershell
py -m pytest tests/test_prelabel_map.py
```

## 4. 训练基线模型

推荐先用默认参数：

```powershell
py scripts/train_prelabeler.py --mode train
```

等价于：

```powershell
yolo detect train data="E:/360MoveData/Users/Administrator/Desktop/Mahjong.v1i.yolov11/data.yaml" model=yolo11s.pt epochs=80 imgsz=960 batch=16 name=prelabel_v1
```

如果显存不足，优先降低 `batch`，再降低 `imgsz`：

```powershell
py scripts/train_prelabeler.py --mode train --batch 8
py scripts/train_prelabeler.py --mode train --batch 4 --imgsz 768
```

训练完成后，核心产物为：

```text
runs/detect/prelabel_v1/weights/best.pt
```

## 5. 真实域快速验证

准备几张来自 `data/frames_selected/` 或博主视频的真实截图，然后运行：

```powershell
py scripts/train_prelabeler.py --mode predict --source data/frames_selected --conf 0.25
```

验收口径：手牌区和牌河的大部分牌能被框出即可；远处小牌漏检、牌背漏检可接受，后续人工补。

## 6. 导出 ONNX

```powershell
py scripts/train_prelabeler.py --mode export
```

等价于：

```powershell
yolo export model=runs/detect/prelabel_v1/weights/best.pt format=onnx imgsz=960
```

导出后确认文件存在：

```text
runs/detect/prelabel_v1/weights/best.onnx
```

并在 `configs/paths.yaml` 中登记：

```yaml
prelabeler_onnx: runs/detect/prelabel_v1/weights/best.onnx
```

## 7. 生成 X-AnyLabeling 预标注

后续任务 4 可直接使用登记的 ONNX 路径和映射表：

```powershell
py scripts/make_prelabel.py --input-root data/frames_selected --paths configs/paths.yaml --prelabel-map configs/prelabel_map.yaml --conf 0.25
```

输出为每张图片旁边的同名 `.json`，可由 X-AnyLabeling 直接打开。
