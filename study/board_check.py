"""One image per video to check the clock detector by eye: the live-score timeline, and the clock box on frames
scored live (top row) and not live (bottom row)."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path(os.environ.get('MMA_WORK', 'work') + '/mcgregor')
LIVE = .6

for vid in sys.argv[1:]:
    b = np.load(DATA / 'board' / f'{vid}.npz')
    t, s, (x0, y0, x1, y1) = b['t'], b['score'], b['bbox'].astype(int)
    cap = cv2.VideoCapture(str(DATA / 'raw' / f'{vid}.mp4'))
    rng = np.random.default_rng(0)
    rows = []
    for mask in (s > LIVE, s < .3):
        idx = np.flatnonzero(mask)
        pick = np.sort(rng.choice(idx, min(5, len(idx)), replace=False)) if len(idx) else []
        tiles = []
        for i in pick:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t[i]) * 1000); ok, im = cap.read()
            if not ok: continue
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 255), 3)
            im = cv2.resize(im, (320, 180)); cv2.putText(im, f'{t[i]:.0f}s {s[i]:.2f}', (5, 20), 0, .6, (0, 255, 255), 2)
            tiles.append(im)
        tiles += [np.zeros((180, 320, 3), np.uint8)] * (5 - len(tiles))
        rows.append(np.hstack(tiles))
    line = np.full((80, 1600, 3), 255, np.uint8)
    xs = (t / max(t[-1], 1) * 1599).astype(int); ys = (75 - np.clip(s, 0, 1) * 70).astype(int)
    for a, c in zip(xs, ys): line[c, a] = (0, 0, 0)
    cv2.line(line, (0, int(75 - LIVE * 70)), (1599, int(75 - LIVE * 70)), (0, 0, 255), 1)
    cv2.putText(line, f'{vid}  live {np.mean(s > LIVE):.0%} of {t[-1]:.0f}s', (5, 15), 0, .5, (0, 0, 0), 1)
    cv2.imwrite(str(DATA / 'board' / f'{vid}_check.jpg'), np.vstack([line] + rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
