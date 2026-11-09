#!/usr/bin/env bash
# R3 ablation sweep: one checkpoint, several decode settings (no retraining).
#   bash scripts/r3_sweep.sh <ckpt_dir> <tag> [limit]
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT_DIR=${1:?ckpt dir}
TAG=${2:?tag}
LIMIT=${3:-3000}
mkdir -p results/diag
JC=$(ls $CKPT_DIR/*.json | head -1)
PC=$(ls $CKPT_DIR/*.pth | head -1)
echo "json=$JC"; echo "pth=$PC"
run () {  # name, env assignments...
  local name=$1; shift
  env "$@" python3 scripts/diag_boundary.py --json "$JC" --ckpt "$PC" --split test \
      --limit $LIMIT --out results/diag/$name.json 2>&1 | tail -2
}
run ${TAG}_a10  AUVIRE_Q_ALPHA=1.0
run ${TAG}_a09  AUVIRE_Q_ALPHA=0.9
run ${TAG}_a08  AUVIRE_Q_ALPHA=0.8
run ${TAG}_a05  AUVIRE_Q_ALPHA=0.5
run ${TAG}_a02  AUVIRE_Q_ALPHA=0.2
run ${TAG}_shuf AUVIRE_Q_ALPHA=0.8 AUVIRE_Q_SHUFFLE=1
run ${TAG}_oracle AUVIRE_Q_ALPHA=0.8 AUVIRE_ORACLE_RANK=1
echo "--- summary ---"
python3 - <<'PYEOF'
import glob, json, os
for path in sorted(glob.glob('results/diag/*.json')):
    try:
        d = json.load(open(path))
    except Exception:
        continue
    if 'metrics' not in d:
        continue
    ap, ar = d['metrics']['ap'], d['metrics']['ar']
    print(f"{os.path.basename(path):32s} AP@0.5 {100*ap['0.5']:6.2f}  @0.75 {100*ap['0.75']:6.2f}  @0.95 {100*ap['0.95']:6.2f}  AR@100 {100*ar['100']:6.2f}")
PYEOF
