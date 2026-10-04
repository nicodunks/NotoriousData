"""Check by eye: are counted kicks kicks? Up to 16 detected kick events (first frame of each) across fights,
McGregor's kicks, cropped around him, with the fight and time printed.

    python qa_kicks.py <fight> ...     -> results/qa/kicks.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F       # noqa: E402
import metrics as M      # noqa: E402

tiles = []
for fid in sys.argv[1:]:
    w = F.windows()[fid]; vid = w.get('video', fid)
    d = F.load(fid)
    q = M.per_frame(d['mcg'], d['opp'], d['shot'], d['t'])
    f = np.nan_to_num(q['kick_pose']) > .5
    starts = np.flatnonzero(f & ~np.r_[False, f[:-1]])
    cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4'))
    for i in starts[:: max(1, len(starts) // 3)][:3]:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(d['t'][i]) * 1000); _, im = cap.read()
        k = d['mcg'][i]; ok = k[:, 2] > .4
        x0, y0 = k[ok, :2].min(0).astype(int) - 40; x1, y1 = k[ok, :2].max(0).astype(int) + 40
        crop = im[max(0, y0):y1, max(0, x0):x1]
        crop = cv2.resize(crop, (200, 260)) if crop.size else np.zeros((260, 200, 3), np.uint8)
        cv2.putText(crop, f'{fid[:4]} {fid[5:12]} {d["t"][i]:.0f}s', (4, 16), 0, .45, (0, 255, 255), 1)
        tiles.append(crop)
tiles += [np.zeros((260, 200, 3), np.uint8)] * ((-len(tiles)) % 8)
grid = np.vstack([np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)])
(HERE / 'results' / 'qa').mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(HERE / 'results' / 'qa' / 'kicks.jpg'), grid)
print(len(tiles), 'tiles')
