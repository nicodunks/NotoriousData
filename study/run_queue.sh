#!/bin/sh
# Pose for the listed fights, one at a time (one process runs ~25 frames/s; several at once thrash).
W="${MMA_WORK:-work}"
cd "$W"
for id in "$@"; do
  [ -f "mcgregor/pose/$id.npz" ] && { echo "have $id"; continue; }
  pgrep -f "pose.py $id" >/dev/null && { echo "running elsewhere $id"; continue; }
  flag=--live; [ "$id" = "2014_poirier" ] && flag=""
  venv/bin/python deep-sports-analysis/mma/study/mcgregor/pose.py "$id" $flag > "mcgregor/pose/$id.log" 2>&1
  grep -E "DONE|Traceback|Error" "mcgregor/pose/$id.log" | tail -1
done
echo QUEUE-DONE
