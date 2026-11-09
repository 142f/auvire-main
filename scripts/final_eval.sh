#!/usr/bin/env bash
# Final 3k-subset evaluation for the paper table (baseline / head arms / controls).
set -uo pipefail
cd /3240608030/auvire-main-maqs
mkdir -p results/diag
LIMIT=${1:-3000}
R1J=$(ls /3240608030/auvire-main/ckpt/r1_baseline/*.json | head -1)
R1P=$(ls /3240608030/auvire-main/ckpt/r1_baseline/*.pth | head -1)

echo "=== baseline (R1 checkpoint, 3 columns) ==="
python3 scripts/diag_boundary.py --json "$R1J" --ckpt "$R1P" --split test --limit $LIMIT --out results/diag/F_base.json 2>&1 | tail -1

for arm in f_a f_b; do
  J=$(ls ckpt/$arm/*.json 2>/dev/null | head -1); P=$(ls ckpt/$arm/*.pth 2>/dev/null | head -1)
  [ -z "$J" ] && continue
  for a in 1.0 0.5 0.3; do
    echo "=== $arm alpha=$a ==="
    AUVIRE_Q_ALPHA=$a python3 scripts/diag_boundary.py --json "$J" --ckpt "$P" --split test --limit $LIMIT --out results/diag/F_${arm}_a${a}.json 2>&1 | tail -1
  done
  echo "=== $arm shuffle (alpha=0.5) ==="
  AUVIRE_Q_ALPHA=0.5 AUVIRE_Q_SHUFFLE=1 python3 scripts/diag_boundary.py --json "$J" --ckpt "$P" --split test --limit $LIMIT --out results/diag/F_${arm}_shuf.json 2>&1 | tail -1
  echo "=== $arm oracle ==="
  AUVIRE_Q_ALPHA=0.5 AUVIRE_ORACLE_RANK=1 python3 scripts/diag_boundary.py --json "$J" --ckpt "$P" --split test --limit $LIMIT --out results/diag/F_${arm}_orac.json 2>&1 | tail -1
done
