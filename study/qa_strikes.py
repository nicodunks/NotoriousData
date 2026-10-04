"""Check by eye: is each detected punch a punch, and is its set-up label right? For a random sample of events,
a strip of 5 frames (onset - 400 ms, onset - 200 ms, onset, halfway, peak), cropped around both fighters, with
the label, kind and timing printed.

    python qa_strikes.py [who=mcg] [n=20] [seed]     -> results/qa/strikes_<who>.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402

who = sys.argv[1] if len(sys.argv) > 1 else 'mcg'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
W = F.windows()
evs = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['who'] == who]
pick = np.random.default_rng(seed).choice(len(evs), min(n, len(evs)), replace=False)
rows = []
for i in sorted(pick, key=lambda i: (evs[i]['fight'], evs[i]['t_onset'])):
    e = evs[i]; vid = W[e['fight']].get('video', e['fight'])
    cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4'))
    ts = [e['t_onset'] - .4, e['t_onset'] - .2, e['t_onset'], (e['t_onset'] + e['t_peak']) / 2, e['t_peak']]
    tiles = []
    for tt in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000); ok, im = cap.read()
        if not ok: im = np.zeros((360, 640, 3), np.uint8)
        h, w = im.shape[:2]
        tiles.append(cv2.resize(im[int(.05 * h):, int(.1 * w):int(.9 * w)], (256, 170)))
    strip = np.hstack(tiles)
    label = f"{e['fight']} {e['t_peak']:.1f}s  {e['kind']}  {e.get('setup') or ''}  {e['duration_ms']:.0f}ms  v{e['peak_speed']:.0f}"
    cv2.rectangle(strip, (0, 0), (strip.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(strip, label, (4, 13), 0, .45, (0, 255, 255), 1)
    rows.append(strip)
(HERE / 'results' / 'qa').mkdir(parents=True, exist_ok=True)
for p in range(0, len(rows), 10):
    cv2.imwrite(str(HERE / 'results' / 'qa' / f'strikes_{who}_{p // 10}.jpg'), np.vstack(rows[p:p + 10]))
print(len(rows), 'strips')
