"""The page's opening: a real exchange, both men's tracked joints, smoothed and gap-filled for drawing only.
The Alvarez knockdown (UFC 205, round 1): 6 s ending with Alvarez on the canvas.

    python hero.py -> results/site_extra/hero.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402

OUT = HERE / 'results' / 'site_extra'; OUT.mkdir(parents=True, exist_ok=True)
F_, T0, T1 = "2016_alvarez", 174.6, 180.0
d = F.load(F_, keep='any', suffix='_30')
sel = np.flatnonzero((d['t'] >= T0) & (d['t'] <= T1))
print('frames', len(sel), 'of', round((T1 - T0) * 30), 'shots', np.unique(d['shot'][sel]))
grid = np.arange(T0, T1, 1 / 30)
seq = {}
for who in ('mcg', 'opp'):
    k = d[who][sel][:, :23].astype(float)
    k[k[..., 2] < .35, :2] = np.nan
    out = np.full((len(grid), 23, 2), np.nan)
    for j in range(23):
        for c in (0, 1):
            v = k[:, j, c]; ok = np.isfinite(v)
            if ok.sum() >= 4: out[:, j, c] = np.interp(grid, d['t'][sel][ok], v[ok])
    out = median_filter(out, size=(5, 1, 1), mode='nearest')
    seq[who] = out
allp = np.concatenate([seq[w].reshape(-1, 2) for w in seq]); allp = allp[np.isfinite(allp).all(1)]
x0, y0 = np.percentile(allp, 1, 0); x1, y1 = np.percentile(allp, 99.5, 0)
h = y1 - y0
norm = lambda a: np.round((a - [(x0 + x1) / 2, y1]) / h, 3)               # x centred, feet at 0, height 1
res = {'fight': F_, 'note': 'UFC 205, McGregor vs Alvarez, round 1: the first knockdown', 'fps': 30,
       'mcg': [[None if not np.isfinite(p).all() else p.tolist() for p in norm(f)] for f in seq['mcg']],
       'opp': [[None if not np.isfinite(p).all() else p.tolist() for p in norm(f)] for f in seq['opp']]}
(OUT / 'hero.json').write_text(json.dumps(res, separators=(',', ':')))
print('wrote', len(grid), 'frames')
