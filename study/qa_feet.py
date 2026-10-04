"""Pre-registered check by eye: are the foot points real? For each fight, 12 random standing frames, cropped to
McGregor's feet, with ankle (white), heel (red) and big toe (cyan) marked and the heel-lift value printed.

    python qa_feet.py <fight> ...     -> results/qa/<fight>_feet.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F       # noqa: E402
import metrics as M      # noqa: E402

for fid in sys.argv[1:]:
    w = F.windows()[fid]; vid = w.get('video', fid)
    d = F.load(fid)
    q = M.per_frame(d['mcg'], d['opp'], d['shot'], d['t'])
    ok = np.flatnonzero(np.isfinite(q['any_heel_up']))
    pick = np.sort(np.random.default_rng(1).choice(ok, min(12, len(ok)), replace=False))
    cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4'))
    tiles = []
    for i in pick:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(d['t'][i]) * 1000); _, im = cap.read()
        k = d['mcg'][i]
        pts = k[[15, 16, 17, 19, 20, 22], :2]
        c = np.nanmean(pts, 0); r = max(60, int(np.nanmax(np.abs(pts - c))) + 40)
        x0, y0 = int(max(0, c[0] - r)), int(max(0, c[1] - r))
        for j, col in [(15, (255, 255, 255)), (16, (255, 255, 255)), (19, (0, 0, 255)), (22, (0, 0, 255)), (17, (255, 255, 0)), (20, (255, 255, 0))]:
            if k[j, 2] >= M.CONF:
                cv2.circle(im, (int(k[j, 0]), int(k[j, 1])), 4, col, -1)
        crop = im[y0:int(c[1] + r), x0:int(c[0] + r)]
        crop = cv2.resize(crop, (220, 220)) if crop.size else np.zeros((220, 220, 3), np.uint8)
        cv2.putText(crop, f"{d['t'][i]:.0f}s lift {q['any_heel_up'][i]:+.2f}", (4, 16), 0, .45, (0, 255, 255), 1)
        tiles.append(crop)
    tiles += [np.zeros((220, 220, 3), np.uint8)] * (12 - len(tiles))
    grid = np.vstack([np.hstack(tiles[r:r + 6]) for r in (0, 6)])
    (HERE / 'results' / 'qa').mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(HERE / 'results' / 'qa' / f'{fid}_feet.jpg'), grid)
    print(fid, len(ok), 'frames with a heel-lift value')
