"""The defence before the left: what McGregor's head and torso did in the 0.8 s before each straight left, relative
to his own hips (so his footwork and the camera's pan drop out), and the same for opponents' rear straights.

Per punch (torso lengths, degrees, ms):
  head_back    furthest the head moved away from the opponent (relative to his hips) before onset
  head_down    furthest it dropped
  head_move    largest head displacement in any direction (the size of the evasive move)
  lean_swing   range of torso lean (degrees) in the window: a bigger swing = the upper body moved, not just the head
  lead_ms      time from the head's furthest point to the punch's onset (how fast he fired off the move)
  slipped      head_move >= 0.25 torso lengths, or lean_swing >= 10 degrees

What a broadcast can't show: a slip toward or away from the camera (depth). Pulls (back) and dips (down) are visible;
lateral slips mostly are not, so "slipped" is a lower bound.

    python slips.py  -> results/left/slips.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402
import strikes as S     # noqa: E402

RNG = np.random.default_rng(9)
WIN = .8

evs = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['kind'] == 'straight']
W = json.loads((HERE / 'windows.json').read_text())
cache = {}
out = []
for e in evs:
    f = e['fight']
    if f not in cache:
        d = F.load(f, keep='upright', suffix='_30')
        cache[f] = d
    d = cache[f]
    t, shot = d['t'], d['shot']
    me, other = (d['mcg'], d['opp']) if e['who'] == 'mcg' else (d['opp'], d['mcg'])
    on = int(np.argmin(np.abs(t - e['t_onset'])))
    st = int(np.argmin(np.abs(t - (e['t_onset'] - WIN))))
    if shot[st] != shot[on] or t[on] - t[st] < WIN * .8:
        continue
    sl = slice(st, on + 1)
    s = M.shot_scale(me, shot)[on]
    hip = (S.track(me, 11) + S.track(me, 12)) / 2
    face = np.nanmean(np.stack([S.track(me, i) for i in range(5)]), 0)
    sh = (S.track(me, 5) + S.track(me, 6)) / 2
    ohip = (S.track(other, 11) + S.track(other, 12)) / 2
    fwd = np.sign(ohip[on, 0] - hip[on, 0])
    rel = (face[sl] - hip[sl]) / s                        # head relative to hips
    rel[:, 0] *= fwd                                      # + = toward the opponent
    if np.isnan(rel).mean() > .3:
        continue
    base = np.nanmedian(rel[:5], 0)                       # where the head started
    dv = rel - base
    back = np.nanmax(-dv[:, 0]); down = np.nanmax(dv[:, 1])
    mag = np.linalg.norm(dv, axis=1)
    k = int(np.nanargmax(mag))
    lean = np.degrees(np.arctan2((sh[sl, 0] - hip[sl, 0]) * fwd, hip[sl, 1] - sh[sl, 1]))
    swing = float(np.nanmax(lean) - np.nanmin(lean))
    rec = {'fight': f, 'date': e['date'], 'who': e['who'], 't_onset': e['t_onset'], 'setup': e['setup'],
           'head_back': float(back), 'head_down': float(down), 'head_move': float(np.nanmax(mag)), 'lean_swing': swing,
           'lead_ms': float((t[on] - t[st + k]) * 1000),
           'path': np.round(dv, 3).tolist()}
    rec['slipped'] = bool(rec['head_move'] >= .25 or swing >= 10)
    out.append(rec)

days = lambda f: (date.fromisoformat(W[f]['date']) - date(2011, 1, 1)).days
summary = {'per_fight': [], 'compare': {}, 'trend': {}}
for f in sorted({r['fight'] for r in out}, key=lambda f: W[f]['date']):
    row = {'fight': f, 'date': W[f]['date'], 'opponent': W[f]['opponent']}
    for who in ('mcg', 'opp'):
        rs = [r for r in out if r['fight'] == f and r['who'] == who]
        row[who] = {'n': len(rs)}
        if rs:
            row[who].update({k: float(np.median([r[k] for r in rs])) for k in ('head_move', 'head_back', 'head_down', 'lean_swing')})
            row[who]['slip_share'] = float(np.mean([r['slipped'] for r in rs]))
    summary['per_fight'].append(row)
# McGregor vs opponents, pooled punches, fight bootstrap
for k in ('head_move', 'head_back', 'head_down', 'lean_swing', 'slipped', 'lead_ms'):
    vals = {}
    for who in ('mcg', 'opp'):
        rs = [r for r in out if r['who'] == who]
        fl = sorted({r['fight'] for r in rs})
        stat = np.mean if k == 'slipped' else np.median
        b = []
        for _ in range(3000):
            pool = [float(r[k]) for g in RNG.choice(fl, len(fl)) for r in rs if r['fight'] == g]
            b.append(stat(pool))
        vals[who] = {'value': float(stat([float(r[k]) for r in rs])), 'ci': np.percentile(b, [2.5, 97.5]).tolist(), 'n': len(rs)}
    summary['compare'][k] = vals
# over time (McGregor, fights with >= 3 lefts)
for k in ('head_move', 'lean_swing', 'slip_share'):
    pts = [(days(r['fight']), r['mcg'][k]) for r in summary['per_fight'] if r['mcg']['n'] >= 3 and k in r['mcg']]
    x, y = np.array(pts).T
    rho = spearmanr(x, y)[0]
    null = [spearmanr(x, RNG.permutation(y))[0] for _ in range(5000)]
    summary['trend'][k] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'n': len(pts)}
# does the slip pay? share of slipped lefts by set-up, and era split
mc = [r for r in out if r['who'] == 'mcg']
for era, test in (('2012-2015', lambda d: d < '2016'), ('2016-2021', lambda d: d >= '2016')):
    rs = [r for r in mc if test(r['date'])]
    summary.setdefault('era', {})[era] = {'n': len(rs), 'slip_share': float(np.mean([r['slipped'] for r in rs])),
                                          'head_move': float(np.median([r['head_move'] for r in rs])),
                                          'lean_swing': float(np.median([r['lean_swing'] for r in rs]))}
# baseline: random 0.8 s windows of McGregor standing, at least 1.5 s from any detected attack of either fighter
allev = json.loads((HERE / 'results' / 'left' / 'events.json').read_text())
base = []
for f, d in cache.items():
    t, shot = d['t'], d['shot']; me, other = d['mcg'], d['opp']
    hip = (S.track(me, 11) + S.track(me, 12)) / 2; face = np.nanmean(np.stack([S.track(me, i) for i in range(5)]), 0)
    sh = (S.track(me, 5) + S.track(me, 6)) / 2; ohip = (S.track(other, 11) + S.track(other, 12)) / 2; sc = M.shot_scale(me, shot)
    ats = [e['t_onset'] for e in allev if e['fight'] == f]
    for _ in range(60):
        on = int(RNG.integers(30, len(t))); st = int(np.argmin(np.abs(t - (t[on] - WIN))))
        if shot[st] != shot[on] or t[on] - t[st] < WIN * .8 or any(abs(t[on] - a) < 1.5 for a in ats): continue
        fwd = np.sign(ohip[on, 0] - hip[on, 0]); sl = slice(st, on + 1)
        rel = (face[sl] - hip[sl]) / sc[on]; rel[:, 0] *= fwd
        if np.isnan(rel).mean() > .3: continue
        dv = rel - np.nanmedian(rel[:5], 0)
        lean = np.degrees(np.arctan2((sh[sl, 0] - hip[sl, 0]) * fwd, hip[sl, 1] - sh[sl, 1]))
        base.append({'fight': f, 'head_move': float(np.nanmax(np.linalg.norm(dv, axis=1))), 'head_back': float(np.nanmax(-dv[:, 0])),
                     'head_down': float(np.nanmax(dv[:, 1])), 'lean_swing': float(np.nanmax(lean) - np.nanmin(lean))})
summary['baseline'] = {'n': len(base), **{k: float(np.median([b[k] for b in base])) for k in ('head_move', 'head_back', 'head_down', 'lean_swing')},
                       'big_move_share': float(np.mean([b['head_move'] >= .4 for b in base]))}
summary['big_move_share'] = {w: float(np.mean([r['head_move'] >= .4 for r in out if r['who'] == w])) for w in ('mcg', 'opp')}
summary['events'] = out
(HERE / 'results' / 'left' / 'slips.json').write_text(json.dumps(summary, separators=(',', ':')))
print(json.dumps({k: v for k, v in summary.items() if k not in ('events', 'per_fight')}, indent=1))
for r in summary['per_fight']:
    print(r['fight'], r['mcg'].get('n'), round(r['mcg'].get('slip_share', float('nan')), 2), round(r['mcg'].get('head_move', float('nan')), 2))
