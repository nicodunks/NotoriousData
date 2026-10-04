"""Calibration set for the punch detector: every local peak of McGregor's left-arm extension under a very loose
net, with the features the strict rule would use, plus a strip per candidate for labelling by eye.

    python calib.py <fight> ...   -> results/left/calib.json, results/qa/calib_<n>.jpg
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F, metrics as M, strikes as S

rows, strips = [], []
for fid in sys.argv[1:]:
    w = F.windows()[fid]; vid = w.get('video', fid)
    d = F.load(fid, keep='upright', suffix='_30'); t, shot = d['t'], d['shot']; me, other = d['mcg'], d['opp']; rate = d['rate']
    ext, Sh, E, Wr = S.arm_series(me, 'left'); s = M.shot_scale(me, shot)
    fwd = np.sign(M.midp(other, 11, 12)[:, 0] - M.midp(me, 11, 12)[:, 0])
    sp = np.linalg.norm(np.gradient(Wr, axis=0), axis=-1) * rate / s
    win = int(.3 * rate); last = -9
    for i in range(win, len(t) - 1):
        if not (ext[i] >= .65 and ext[i] >= ext[i - 1] and ext[i] >= ext[i + 1]): continue
        j0 = i - win
        if shot[j0] != shot[i] or t[i] - t[j0] > .4: continue
        seg = ext[j0:i]
        if not np.isfinite(seg).any(): continue
        k0 = j0 + int(np.nanargmin(seg)); rise = ext[i] - ext[k0]
        if rise < .12 or t[i] - last < .35: continue
        last = t[i]
        rows.append({'fight': fid, 't': float(t[i]), 't0': float(t[k0]), 'ext': float(ext[i]), 'rise': float(rise),
                     'vmax': float(np.nanmax(sp[k0:i + 1])), 'travel': float((Wr[i, 0] - Wr[k0, 0]) * fwd[i] / s[i]),
                     'elbow': float(S.angle(Sh[i], E[i], Wr[i])), 'height': float((M.midp(me, 5, 6)[i, 1] - Wr[i, 1]) / s[i])})
print(len(rows), 'candidates')
rng = np.random.default_rng(3); pick = sorted(rng.choice(len(rows), min(40, len(rows)), replace=False))
for n, i in enumerate(pick):
    r = rows[i]; vid = F.windows()[r['fight']].get('video', r['fight'])
    cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4')); tiles = []
    for tt in (r['t0'] - .15, r['t0'], (r['t0'] + r['t']) / 2, r['t'], r['t'] + .1):
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000); ok, im = cap.read(); h, w = im.shape[:2]
        tiles.append(cv2.resize(im[int(.05 * h):, int(.08 * w):int(.92 * w)], (230, 150)))
    st = np.hstack(tiles); cv2.rectangle(st, (0, 0), (st.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(st, f"#{n} {r['fight'][2:]} {r['t']:.1f}s ext{r['ext']:.2f} rise{r['rise']:.2f} v{r['vmax']:.0f} trav{r['travel']:.2f} elb{r['elbow']:.0f}", (3, 12), 0, .42, (0, 255, 255), 1)
    strips.append(st); r['sample'] = n
(HERE / 'results' / 'left').mkdir(parents=True, exist_ok=True)
(HERE / 'results' / 'left' / 'calib.json').write_text(json.dumps(rows, indent=1))
for p in range(0, len(strips), 8):
    cv2.imwrite(str(HERE / 'results' / 'qa' / f'calib_{p // 8}.jpg'), np.vstack(strips[p:p + 8]))
