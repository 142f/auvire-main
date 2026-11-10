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
| A 基线 | `AUVIRE_SOFT_CLS=0`（硬目标，= 关掉创新点） |
| S1 主创新 | 默认（`criterion.params.soft_classification=True`） |
| S2 地板（已否证） | `AUVIRE_SOFT_FLOOR=0.5` |
| 分级校准（历史对照） | `AUVIRE_LEVEL_SCALE="1,1,0.9,0.8"` |

> 2026-11-11：MAQS（质量读出头）、MAQS-R（排序项）、膨胀/池化变体、多头集成与 oracle/置换诊断**代码已从仓库移除**（见 `feat/s1-only` 的 `chore` 提交），对应的 `AUVIRE_QUALITY` / `AUVIRE_Q_*` / `AUVIRE_RANK_LOSS` / `AUVIRE_INIT_CKPT` / `AUVIRE_LR` 开关不再存在；这些臂的历史命令与数字只保留在 git 历史与 D5 台账里作否定空间。

## 4. 单次评测（3k 子集 / 全量）

```bash
cd /3240608030/auvire-main-maqs
JC=$(ls ckpt/s1_softcls/*.json); PC=$(ls ckpt/s1_softcls/*.pth)
# 3k 子集（与历史 83.148 同口径）
python3 scripts/diag_boundary.py --json "$JC" --ckpt "$PC" \
  --split test --limit 3000 --out results/diag/S1_3000.json
# 全量 test（26,100）
python3 scripts/diag_boundary.py --json "$JC" --ckpt "$PC" \
  --split test --limit 0 --out results/diag/S1_full.json
```

> `--out` 写的 JSON 只含 `metrics` / `limit` / `ckpt`（原来的边界误差谱与召回分层拆解随诊断代码一并移除，见 §7）。

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
# 本地：在同一 seed 下比较 HEAD 与基线 worktree（AUVIRE_SOFT_CLS=0）的 state_dict / 前向 / 损失
git worktree add .tmp/base_worktree 5e21901 --detach
python .tmp/equiv_check.py E:/Project/auvire-main/.tmp/base_worktree
python .tmp/equiv_check.py E:/Project/auvire-main
# 期望：n_params / state_dict_sha / out_sha / z_sha / loss 四行完全相同
```

## 7. 已移除的诊断（D3 的证据来源）

> **2026-11-11 状态**：原 `AUVIRE_BOUNDARY_DIAG` / `AUVIRE_RANK_DIAG` 两个只读诊断
> （D3 的 unreachable_095 / near_miss_08_095 / err_by_len / best_iou_hist 与召回分层
> FP 拆解）**代码已移除**，D3 的全部数字只能从 git 历史 `4f91ef2` 复算；D3 台账本身保留。
## 8. 已移除的第二轮命令（质量读出头 + 集成）

> 该臂的代码（`src/models.py::Model.quality`、`losses.quality/ranking`、`scripts/final_eval.sh`、`run_final_heads.sh`、`r3_sweep.sh`、`ensemble_eval.py`）已于 2026-11-11 删除，命令不再可执行；原文见 git 历史 `a49c8c8`。数字（+0.91 全量 / +0.56±0.44 多头均值）作为否定空间保留在 D5 台账。
## 9. 主创新 S1 的一条命令（最终交付）

```bash
cd /3240608030/auvire-main-maqs
export AUVIRE_LABEL_MODE=continuous AUVIRE_KEEP_CKPT=1 AUVIRE_EPOCHS=32 \
       AUVIRE_SOFT_CLS=1 AUVIRE_SEED=1234567891 \
       AUVIRE_FOLDER=ckpt/s1_softcls ARM=s1_softcls GIT_COMMIT=<local commit>
bash scripts/run_arm.sh
# → logs/s1_softcls/<ts>.meta.json（env/md5/显存/耗时/test_metrics 全在内）
# 评测：python3 scripts/diag_boundary.py --json ckpt/s1_softcls/*.json --ckpt ckpt/s1_softcls/*.pth --split test --limit 0
```

**关掉创新点 = 基线**：同一命令把 `AUVIRE_SOFT_CLS` 去掉（或置 0），训练路径与 R1 逐位相同（`losses.focal` 里的软目标分支不执行），结果即 83.405。
## 10. 关于"默认开关"的更正（2026-11-10）

S1 原先只能由 `AUVIRE_SOFT_CLS=1` 打开，这违反任务书淘汰线里的"默认关闭 → 不算创新"。已把 `soft_classification: True` 写进 `src/config.py` 的 `DEFAULT_LAVDF` / `DEFAULT_AVDEEPFAKE1M`，即：

| 臂 | 命令 |
|---|---|
| **主方法（S1，默认）** | `AUVIRE_LABEL_MODE=continuous AUVIRE_QUALITY=0 AUVIRE_EPOCHS=32 python3 scripts/train.py -d lavdf` |
| 基线（关掉创新点） | 同上 + `AUVIRE_SOFT_CLS=0` |

两者除该目标外**代码路径完全相同**（`losses.focal` 第 140 行的分支），因此"关掉即回基线"是构造性的，不依赖另一个实验。
