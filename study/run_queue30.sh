#!/bin/sh
# Every-frame pose (POSE_RATE=30) for the listed fights, one at a time; three of these run side by side.
W="${MMA_WORK:-work}"
cd "$W"
for id in "$@"; do
  [ -f "mcgregor/pose/${id}_30.npz" ] && continue
  pgrep -f "pose.py $id " >/dev/null && continue
  flag=--live; [ "$id" = "2014_poirier" ] && flag=""
  POSE_RATE=30 venv/bin/python deep-sports-analysis/mma/study/mcgregor/pose.py "$id" $flag > "mcgregor/pose/${id}_30.log" 2>&1
  grep -E "DONE|Traceback|Error" "mcgregor/pose/${id}_30.log" | tail -1
done
echo QUEUE30-DONE
