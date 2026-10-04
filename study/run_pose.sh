#!/bin/sh
# Pose for every downloaded fight that doesn't have it yet, three at a time, live frames only (needs board/).
W="${MMA_WORK:-work}"
cd "$W"
for f in mcgregor/raw/*.mp4; do id=$(basename "$f" .mp4)
  [ -f "mcgregor/pose/$id.npz" ] && continue
  pgrep -f "pose.py $id" >/dev/null && continue
  [ -f "mcgregor/board/$id.npz" ] && echo "$id"
done | xargs -P 3 -I{} sh -c 'venv/bin/python deep-sports-analysis/mma/study/mcgregor/pose.py {} --live > mcgregor/pose/{}.log 2>&1; tail -1 mcgregor/pose/{}.log'
echo POSE-PASS-DONE
