#!/usr/bin/env bash
# One command per experiment arm (D6): trains, then writes logs/<arm>/<ts>.meta.json.
#   ARM=maqs AUVIRE_EPOCHS=32 AUVIRE_LABEL_MODE=continuous bash scripts/run_arm.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ARM=${ARM:?set ARM}
TS=$(date +%Y%m%d_%H%M%S)
OUT="logs/$ARM"
mkdir -p "$OUT"
LOG="$OUT/${TS}.log"
GPU="$OUT/${TS}.gpu"
START=$(date +%s)
( while :; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU"; sleep 5; done ) &
POLL=$!
python3 scripts/train.py -d lavdf > "$LOG" 2>&1
RC=$?
kill $POLL 2>/dev/null
END=$(date +%s)
python3 scripts/write_meta.py --arm "$ARM" --log "$LOG" --gpu "$GPU" --start "$START" --end "$END" --rc "$RC" --out "$OUT/${TS}.meta.json"
exit $RC
