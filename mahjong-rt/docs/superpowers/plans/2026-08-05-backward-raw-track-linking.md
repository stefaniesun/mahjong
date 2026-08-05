# Backward Raw Track Linking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对稳态差分确认的新落地牌，使用录制中的原始检测和完整类别概率向前回溯，跨越短遮挡及轨迹 ID 断裂，获得移动前史并修正事件时间和置信度。

**Architecture:** 保留 `reconstruct_events` 作为候选生成器，新建独立 raw-link 模块消费 `Recording` 和稳定事件。模块先把 raw box 中心映射到世界坐标，再从事件落地时间附近的目标类检测向过去做动态规划；边代价联合类别概率、检测分数、时间间隔、空间速度和尺寸变化。回溯只做证据增强和时间修正，评测入口新增 `backtrack` 方法，与 `stable` 和 `online` 公平对照。

**Tech Stack:** Python 3.12、NumPy、pytest、现有 `Recording`、`GameEvent`。

---

## 文件结构

- Create: `mahjong_rt/raw_event_backtrack.py`：原始检测节点构建、向后动态规划、路径证据和事件精化。
- Create: `tests/test_raw_event_backtrack.py`：跨断轨、类别非 top-1、遮挡、静止干扰和时间修正测试。
- Modify: `mahjong_rt/offline_game_events.py`：在事件元数据中保留稳定落点世界坐标，供回溯使用。
- Modify: `scripts/eval_game_events.py`：新增 `--method backtrack`。
- Modify: `docs/OPEN_PROBLEM_events.md`：记录真实结果，不以真值调参。

### Task 1: 原始检测向后连接

- [ ] 先写失败测试：目标类别仅为非 top-1 仍可连接；允许短帧缺失和 ID 不存在；静止在落点的旧牌不产生移动证据。
- [ ] 运行 `python -m pytest tests/test_raw_event_backtrack.py -q`，确认因模块缺失失败。
- [ ] 实现 `BacktrackConfig`、`BacktrackEvidence` 和 `trace_landing_backwards(recording, event, landing, config)`；节点使用 `probs[:, class_index]`，不依赖 tracker ID。
- [ ] 重跑测试至通过。

### Task 2: 精化事件

- [ ] 先写失败测试：有足够位移的路径将时间修正到到达落点时刻，并提升置信度；无路径时按配置保留或过滤事件。
- [ ] 实现 `refine_events_with_raw_tracks`，保持 `GameEvent` 协议兼容和连续 `seq`。
- [ ] 运行新旧事件测试。

### Task 3: 真实评测

- [ ] 新增 `--method backtrack` 并传入 `Recording`。
- [ ] 运行 clip01 的 `stable`、`backtrack`、`online` 对照。
- [ ] 记录匹配、随机期望、误报和牌正确；若回溯没有改善，明确保留失败结论。

### Task 4: 回归与提交

- [ ] 运行 `python -m pytest tests -q`。
- [ ] 运行 `git diff --check` 和代码审查。
- [ ] 提交并推送当前功能分支。
