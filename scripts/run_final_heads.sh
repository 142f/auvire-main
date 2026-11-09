#!/usr/bin/env bash
# Train the read-out quality heads on the FINAL baseline checkpoint (R1, epoch 32).
#   bash scripts/run_final_heads.sh
set -uo pipefail
cd /3240608030/auvire-main-maqs
R1P=$(ls /3240608030/auvire-main/ckpt/r1_baseline/*.pth | head -1)
cp "$R1P" /tmp/r1_final.pth
echo "warm start: $R1P -> /tmp/r1_final.pth ($(stat -c%s /tmp/r1_final.pth) bytes)"
mkdir -p logs

common_env() {
  export GIT_COMMIT=wip
  export AUVIRE_LABEL_MODE=continuous
  export AUVIRE_KEEP_CKPT=1
  export AUVIRE_EPOCHS=4
  export AUVIRE_INIT_CKPT=/tmp/r1_final.pth
  export AUVIRE_FREEZE_BACKBONE=1
  export AUVIRE_LR=0.001
  export AUVIRE_Q_ALPHA=0.8
  export AUVIRE_Q_OFFSETS=1
}

(
  common_env
  export AUVIRE_FOLDER=ckpt/f_a
  export ARM=f_a
  bash scripts/run_arm.sh
) > logs/runner_f_a.out 2>&1 &
echo "A pid $!"

(
  common_env
  export AUVIRE_FOLDER=ckpt/f_b
  export ARM=f_b
  export AUVIRE_RANK_LOSS=1
  bash scripts/run_arm.sh
) > logs/runner_f_b.out 2>&1 &
echo "B pid $!"
