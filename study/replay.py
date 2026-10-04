"""Skeleton replay data for the left-hand page: every McGregor straight left (and the opponent in the same frames),
time-normalised: 24 steps before onset (0.8 s), 8 steps from onset to full extension, 9 after (0.3 s). Coordinates
are in his torso lengths, hips at the origin, opponent to the right, as he stood (southpaw, not mirrored).
Also the median sequence for 2012-2015 and for 2016-2021.

    python replay.py   -> results/left/replay.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402

PRE, MID, POST = 24, 8, 9


def canon(me, ref_me, ref_other, scale):
    """Points of `me` relative to McGregor's hips, divided by his torso, opponent to the right."""
    hip = M.midp(ref_me, 11, 12)
    fwd = np.sign(M.midp(ref_other, 11, 12)[:, 0] - hip[:, 0])
    p = np.stack([M.P(me, i) for i in range(23)], axis=1)
    p = (p - hip[:, None, :]) / scale[:, None, None]
    p[..., 0] *= fwd[:, None]
    return p


def sample(t, arr, times):
    out = np.full((len(times),) + arr.shape[1:], np.nan)
    for j, tt in enumerate(times):
        i = int(np.argmin(np.abs(t - tt)))
        if abs(t[i] - tt) < .025: out[j] = arr[i]
    return out


PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)]
allev = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['kind'] == 'straight']
cache, seqs, oseqs = {}, [], []
for e in allev:
    if e['fight'] not in cache:
        d = F.load(e['fight'], keep='upright', suffix='_30')
        sm, so = M.shot_scale(d['mcg'], d['shot']), M.shot_scale(d['opp'], d['shot'])
        cache[e['fight']] = (d['t'], canon(d['mcg'], d['mcg'], d['opp'], sm), canon(d['opp'], d['mcg'], d['opp'], sm),
                             canon(d['opp'], d['opp'], d['mcg'], so), canon(d['mcg'], d['opp'], d['mcg'], so))
    t, me, op, ome, oop = cache[e['fight']]
    on, pk = e['t_onset'], e['t_peak']
    times = np.r_[on + np.arange(-PRE, 0) / 30, np.linspace(on, pk, MID, endpoint=False), pk + np.arange(POST) / 30]
    if e['who'] == 'mcg':
        a, b = sample(t, me, times), sample(t, op, times)
    else:
        a, b = sample(t, ome, times), sample(t, oop, times)
        if e['hand'] == 'right':          # relabel so the punching (rear) arm is always index 9, like McGregor's left
            for i, j in PAIRS:
                a[:, [i, j]] = a[:, [j, i]]
    if np.isnan(a[:, 9, 0]).mean() > .3:
        continue
    rec = {'fight': e['fight'], 'date': e['date'], 'opponent': e['opponent'], 'setup': e['setup'], 'hand': e['hand'],
           'speed': round(e['peak_speed'], 1), 'me': np.round(a, 2).tolist(), 'opp': np.round(b, 2).tolist()}
    (seqs if e['who'] == 'mcg' else oseqs).append(rec)
avg = {}
for era, test in (('early', lambda d: d < date(2016, 1, 1)), ('late', lambda d: d >= date(2016, 1, 1))):
    stack = np.array([s['me'] for s in seqs if test(date.fromisoformat(s['date']))], float)
    m = np.nanmedian(stack, axis=0)
    m[np.isfinite(stack[..., 0]).sum(0) < 5] = np.nan
    avg[era] = {'n': len(stack), 'me': np.round(m, 2).tolist()}
ost = np.array([s['me'] for s in oseqs], float); m = np.nanmedian(ost, axis=0); m[np.isfinite(ost[..., 0]).sum(0) < 5] = np.nan
mst = np.array([s['me'] for s in seqs], float); mm = np.nanmedian(mst, axis=0); mm[np.isfinite(mst[..., 0]).sum(0) < 5] = np.nan
avg['opp'] = {'n': len(oseqs), 'me': np.round(m, 2).tolist()}
avg['mcg'] = {'n': len(seqs), 'me': np.round(mm, 2).tolist()}
out = {'pre': PRE, 'mid': MID, 'post': POST, 'seqs': seqs, 'oseqs': oseqs, 'avg': avg}
(HERE / 'results' / 'left' / 'replay.json').write_text(json.dumps(out, separators=(',', ':')).replace('NaN', 'null'))
print(len(seqs), len(oseqs), 'sequences', {k: v['n'] for k, v in avg.items()}, round((HERE / 'results' / 'left' / 'replay.json').stat().st_size / 1e6, 2), 'MB')
