"""Who is who: cluster the shorts colours of every fighter-sized detection in a fight's live frames, and show a
row of sample crops per cluster, so the McGregor and opponent clusters can be named by eye (once per fight).
The named cluster centres go into windows.json as the fight's two prototypes.

    python identity.py <fight>      -> $DATA/identity/<fight>.jpg, cluster centres printed
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from frames import DATA, windows   # noqa: E402
from live import live              # noqa: E402

fid = sys.argv[1]
w = windows()[fid]
vid = w.get('video', fid)
d = np.load(DATA / 'pose' / f'{vid}.npz')
t, boxes, shorts, H = d['t'], d['boxes'], d['shorts'], float(d['height'])
bt, bon, _ = live(vid)
on = np.interp(t, bt, bon.astype(float)) > .5
if 'span' in w:
    on &= (t >= w['span'][0]) & (t <= w['span'][1])
hgt = (boxes[:, :, 3] - boxes[:, :, 1]) / H
ok = on[:, None] & (hgt > .25) & (boxes[:, :, 1] > .01 * H) & np.isfinite(shorts).all(-1)
fi, pj = np.nonzero(ok)
X = shorts[fi, pj]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
km = KMeans(k, n_init=10, random_state=0).fit(X)
cap = cv2.VideoCapture(str(DATA / 'raw' / f'{vid}.mp4'))
rng = np.random.default_rng(0)
rows = []
for c in range(k):
    members = np.flatnonzero(km.labels_ == c)
    pick = rng.choice(members, min(8, len(members)), replace=False)
    tiles = []
    for m in pick:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t[fi[m]]) * 1000); okk, im = cap.read()
        x0, y0, x1, y1 = boxes[fi[m], pj[m], :4].astype(int)
        crop = im[max(0, y0):y1, max(0, x0):x1]
        tiles.append(cv2.resize(crop, (90, 180)) if crop.size else np.zeros((180, 90, 3), np.uint8))
    tiles += [np.zeros((180, 90, 3), np.uint8)] * (8 - len(tiles))
    lab = np.uint8([[km.cluster_centers_[c]]]); swatch = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    sw = np.full((180, 60, 3), swatch[0, 0], np.uint8)
    cv2.putText(sw, str(c), (15, 100), 0, 1.5, (0, 255, 255), 3)
    rows.append(np.hstack([sw] + tiles))
    print(fid, 'cluster', c, 'n', len(members), 'Lab', km.cluster_centers_[c].round(1).tolist())
(DATA / 'identity').mkdir(exist_ok=True)
cv2.imwrite(str(DATA / 'identity' / f'{fid}.jpg'), np.vstack(rows))
