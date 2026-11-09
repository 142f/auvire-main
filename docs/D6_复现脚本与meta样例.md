# D6 复现脚本与 meta.json 样例

> 目标（验收 [6]）：**一条命令 + 一个 meta.json** 能复现表里的每个数字。
> 所有路径均为远端 `/3240608030/auvire-main*` 的绝对路径；本地仓库为唯一版本库。

## 1. 目录布局（远端）

| 目录 | 用途 | 代码版本 |
|---|---|---|
| `/3240608030/auvire-main` | **基线臂（R1）**：HEAD=`5e21901`，不含 MAQS | 已 md5 对齐 |
| `/3240608030/auvire-main-maqs` | **创新臂（R2/R3）**：`0def479`，含 MAQS；`data` 为符号链接复用主目录数据 | 已 md5 对齐 |
| `/3240608030/auvire-main-sub` | **泛化臂（R4）**：5k/1k/1k 子集训练（数据量 1/15.7） | 同 `0def479` |

## 2. 一条命令跑一臂

```bash
cd /3240608030/auvire-main-maqs
export GIT_COMMIT=$(git -C <local> rev-parse --short HEAD)   # 远端无 git，靠 md5 兜底
export AUVIRE_LABEL_MODE=continuous     # 连续标签（基线口径，已 +32.6pp）
export AUVIRE_KEEP_CKPT=1               # 保留 .pth，供离线诊断
export AUVIRE_EPOCHS=32                 # 与历史 C_cont 记录同预算（该记录在 32 epoch 结束）
export AUVIRE_Q_ALPHA=0.8               # MAQS 解码融合指数，α=1.0 即"融合关"
export ARM=r2_maqs
bash scripts/run_arm.sh
```

`run_arm.sh` 会：跑 `scripts/train.py -d lavdf` → 写 `logs/<arm>/<ts>.log`（训练日志）与 `<ts>.gpu`（每 5 s 采样显存/利用率）→ 调 `scripts/write_meta.py` 生成 `<ts>.meta.json`。

## 3. 单变量切换（同一份代码）

| 臂 | 命令差异 |
|---|---|
| A 基线 | `AUVIRE_QUALITY=0`（不建质量头、composition 去掉 quality） |
| B MAQS | 默认（`model.quality.enabled=True`，composition 含 `quality`） |
| B1 融合关 | 同一 checkpoint，评测时 `AUVIRE_Q_ALPHA=1.0` |
| B2 融合强度 | 同一 checkpoint，`AUVIRE_Q_ALPHA∈{0.9,0.8,0.5,0.2}` |
| C 置换对照 | 同一 checkpoint，`AUVIRE_Q_SHUFFLE=1`（质量通道跨样本打乱） |
| D 分级校准（历史对照） | `AUVIRE_LEVEL_SCALE="1,1,0.9,0.8"` |
| E oracle 排序上限 | `AUVIRE_ORACLE_RANK=1`（把分数换成"与任意 GT 的最大 IoU"） |

## 4. 单次评测（3k 子集 / 全量）

```bash
cd /3240608030/auvire-main-maqs
JC=$(ls ckpt/r2_maqs/*.json); PC=$(ls ckpt/r2_maqs/*.pth)
# 3k 子集（与历史 83.15 同口径）
AUVIRE_Q_ALPHA=0.8 python3 scripts/diag_boundary.py --json "$JC" --ckpt "$PC" \
  --split test --limit 3000 --out results/diag/R2_a08_3000.json
# 全量 test（26,100）
AUVIRE_Q_ALPHA=0.8 python3 scripts/diag_boundary.py --json "$JC" --ckpt "$PC" \
  --split test --limit 0 --out results/diag/R2_a08_full.json
```

## 5. meta.json 样例（真实字段）

```json
{
  "arm": "r2_maqs",
  "command": "python3 scripts/train.py -d lavdf",
  "env": {
    "AUVIRE_LABEL_MODE": "continuous",
    "AUVIRE_KEEP_CKPT": "1",
    "AUVIRE_EPOCHS": "32",
    "AUVIRE_Q_ALPHA": "0.8",
    "AUVIRE_FOLDER": "ckpt/r2_maqs",
    "GIT_COMMIT": "0def479"
  },
  "exit_code": 0,
  "wall_clock_seconds": 11880,
  "git_commit": "0def479",
  "source_md5": {
    "src/models.py": "…", "src/losses.py": "…", "src/eval.py": "…",
    "src/training.py": "…", "src/datasets.py": "…", "src/config.py": "…"
  },
  "versions": {"python": "3.10.12", "torch": "2.8.0+cu128", "cuda": "12.8",
               "gpu": "NVIDIA GeForce RTX 4090 D"},
  "peak_memory_mib": 8123,
  "mean_utilization_pct": 71.4,
  "test_metrics": {"tloss": 0.0, "tap@0.5": 0.0, "tap@0.75": 0.0,
                   "tap@0.95": 0.0, "tar@100": 0.0}
}
```

> 说明：`test_metrics` 由脚本从训练日志末尾的 JSON 块直接解析，**不是手工填写**；`source_md5` 是"远端代码 = 哪个本地 commit"的唯一证据（远端无 git）。

## 6. 单变量性证明（可复跑）

```bash
# 本地：在同一 seed 下比较 HEAD 与 MAQS-closed 的 state_dict / 前向 / 损失
git worktree add .tmp/base_worktree 5e21901 --detach
python .tmp/equiv_check.py E:/Project/auvire-main/.tmp/base_worktree
python .tmp/equiv_check.py E:/Project/auvire-main
# 期望：n_params / state_dict_sha / out_sha / z_sha / loss 四行完全相同
```

## 7. 诊断复算（D3 的所有数字）

```bash
AUVIRE_BOUNDARY_DIAG=$PWD/results/diag/C_all_3000.json \
  python3 scripts/diag_boundary.py --json "$JC" --ckpt "$PC" --split test --limit 3000 --out /tmp/x.json
# C_all_3000.json 里的 unreachable_095 / near_miss_08_095 / err_by_len / best_iou_hist
# 即 D3 表格的全部证据；fps=25，容忍 0.85 帧 = IoU 0.95 对应的最大边界误差 0.0251·L
```
