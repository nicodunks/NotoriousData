"""Data for keypoint-MoSeq: McGregor's every-frame keypoints (30 /s) in continuous stretches where both fighters are
upright, in his own body frame (opponent always to the right), one recording per stretch of >= 2 s.

    python moseq_prep.py   -> $DATA/moseq/data.npz (coords, confs per recording) and index.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402

KP = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22]
NAMES = ['nose', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee',
         'r_knee', 'l_ankle', 'r_ankle', 'l_toe', 'l_heel', 'r_toe', 'r_heel']
OUT = F.DATA / 'moseq'
OUT.mkdir(exist_ok=True)
W = F.windows()
coords, confs, index = {}, {}, []
for f in sorted(W, key=lambda f: W[f]['date']):
    if 'exclude' in W[f] or not (F.DATA / 'pose' / f"{W[f].get('video', f)}_30.npz").exists() or not W[f]['shorts']:
        continue
    d = F.load(f, keep='upright', suffix='_30')
    if not d['n']: continue
    t, shot, me, op = d['t'], d['shot'], d['mcg'], d['opp']
    fwd = np.sign(M.midp(op, 11, 12)[:, 0] - M.midp(me, 11, 12)[:, 0])
    # side-on shots only (both fighters about the same size on screen, well apart): the view in which a movement
    # looks the same from fight to fight, so syllables describe movement rather than camera angle
    q = M.per_frame(me, op, shot, t); qo = M.per_frame(op, me, shot, t)
    ratio = q['scale'] / qo['scale']
    sep = np.abs(q['hip_x'] * q['scale'] - qo['hip_x'] * qo['scale']) / ((q['scale'] + qo['scale']) / 2)
    side = (ratio > .8) & (ratio < 1.25) & (sep > 1.2)
    shot = np.where(side, shot, -1 - np.arange(len(shot)))          # break runs wherever the view isn't side-on
    r = M.runs(t, shot, max_gap=1.5 / d['rate'])
    for g in np.unique(r):
        idx = np.flatnonzero(r == g)
        if len(idx) < 2 * d['rate']: continue
        xy = me[idx][:, KP, :2].astype(float).copy(); c = me[idx][:, KP, 2].astype(float)
        xy[..., 0] *= fwd[idx][:, None]                      # mirror so the opponent is always to the right
        name = f'{f}__{len(index):04d}'
        coords[name] = xy; confs[name] = c
        index.append({'name': name, 'fight': f, 'date': W[f]['date'], 't0': float(t[idx[0]]), 't1': float(t[idx[-1]]),
                      'n': int(len(idx)), 'rate': float(d['rate'])})
    print(f, sum(i['n'] for i in index if i['fight'] == f), 'frames', flush=True)
np.savez_compressed(OUT / 'data.npz', **{f'c__{k}': v for k, v in coords.items()}, **{f'w__{k}': v for k, v in confs.items()})
(OUT / 'index.json').write_text(json.dumps({'bodyparts': NAMES, 'recordings': index}, indent=1))
print(len(index), 'recordings', sum(i['n'] for i in index), 'frames')
