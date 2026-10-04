"""Feet: flat-footed or bouncing, read from the feet themselves, in every standing frame where heel, toe and ankle
are seen (any camera angle). McGregor and the opponent in the same frames.

Per fight:
  heel_lift      median heel lift, both feet (toe height minus heel height, per foot length)
  heel_up_share  share of foot-frames with the heel clearly up (lift >= 0.35 of a foot length)
  ankle_bounce   RMS of each ankle's vertical motion at 1.5-4 Hz, per continuous stretch, torso lengths
  planted_share  share of frames with both ankles nearly still (< 0.6 torso lengths / s, above pose jitter) and both heels down (< 0.25)
  springs        up-and-down cycles of the ankles per second (dominant frequency, 1-4.5 Hz)

    python feet.py -> results/stance/feet.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import metrics as M     # noqa: E402

RNG = np.random.default_rng(13)
R = json.loads((HERE / 'results' / 'results.json').read_text())
FEET = [(15, 17, 19), (16, 20, 22)]             # ankle, big toe, heel


def foot_measures(k, other, t, shot):
    s = M.shot_scale(k, shot)
    out = {'lift': [], 'up': [], 'bounce': [], 'freq': []}
    ank_v, heel_down, seen_both = [], [], np.ones(len(t), bool)
    r = M.runs(t, shot)
    for a, toe, heel in FEET:
        A, T, H = M.P(k, a), M.P(k, toe), M.P(k, heel)
        foot = np.linalg.norm(H - T, axis=-1)
        ok = np.isfinite(foot) & np.isfinite(A[:, 0]) & (foot > .12 * s)
        seen_both &= ok
        lift = np.where(ok, (T[:, 1] - H[:, 1]) / foot, np.nan)
        out['lift'].append(lift); out['up'].append(np.where(ok, lift >= .35, np.nan))
        y = A[:, 1] / s
        v = np.full(len(t), np.nan)
        for g in np.unique(r):
            m = np.flatnonzero(r == g)
            if len(m) < 20 or np.isnan(y[m]).mean() > .15: continue
            yy = np.interp(np.arange(len(m)), np.flatnonzero(~np.isnan(y[m])), y[m][~np.isnan(y[m])])
            yy = yy - np.polyval(np.polyfit(np.arange(len(m)), yy, 2), np.arange(len(m)))
            f, pxx = welch(yy, fs=10, nperseg=min(len(m), 20))
            band = (f >= 1.5) & (f <= 4)
            out['bounce'].append((np.sqrt(np.trapezoid(pxx[band], f[band])), len(m)))
            b2 = (f >= 1) & (f <= 4.5)
            out['freq'].append((f[b2][np.argmax(pxx[b2])], len(m)))
            xy = np.stack([np.interp(np.arange(len(m)), np.flatnonzero(~np.isnan(A[m, d])), A[m, d][~np.isnan(A[m, d])]) for d in (0, 1)], 1) / s[m, None]
            v[m[1:-1]] = np.linalg.norm(xy[2:] - xy[:-2], axis=1)[:] * 10 / 2
        ank_v.append(v); heel_down.append(lift < .25)
    planted = (ank_v[0] < .6) & (ank_v[1] < .6) & heel_down[0] & heel_down[1]
    valid = seen_both & np.isfinite(ank_v[0]) & np.isfinite(ank_v[1])
    w = lambda lst: float(np.average([x for x, _ in lst], weights=[n for _, n in lst])) if lst else np.nan
    return {'heel_lift': float(np.nanmedian(np.concatenate(out['lift']))),
            'heel_up_share': float(np.nanmean(np.concatenate(out['up']))),
            'ankle_bounce': w(out['bounce']), 'springs': w(out['freq']),
            'planted_share': float(planted[valid].mean()) if valid.sum() >= 50 else np.nan,
            'feet_seen_s': float(valid.sum() / 10)}


rows = []
for f in R['fights']:
    p = HERE / 'results' / 'perframe' / f"{f['fight']}.npz"
    if not p.exists() or f['standing_seconds'] < 30: continue
    d = np.load(p); t, shot = d['t'], d['shot']
    mk, ok = d['mcg_kps'].astype(np.float32), d['opp_kps'].astype(np.float32)
    rows.append({'fight': f['fight'], 'date': f['date'], 'opponent': f['opponent'],
                 'mcg': foot_measures(mk, ok, t, shot), 'opp': foot_measures(ok, mk, t, shot)})
days = lambda d: (date.fromisoformat(d) - date(2011, 1, 1)).days
tests = {}
for k in ('heel_lift', 'heel_up_share', 'ankle_bounce', 'springs', 'planted_share'):
    tests[k] = {}
    for who in ('mcg', 'opp', 'diff'):
        pts = [(days(r['date']), r['mcg'][k] - r['opp'][k] if who == 'diff' else r[who][k]) for r in rows
               if np.isfinite(r['mcg'][k]) and np.isfinite(r['opp'][k])]
        x, y = np.array(pts).T
        rho = spearmanr(x, y)[0]; null = np.array([spearmanr(x, RNG.permutation(y))[0] for _ in range(4000)])
        e, l = y[x < days('2016-01-01')], y[x >= days('2016-01-01')]
        tests[k][who] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'n': len(y),
                         'early': float(np.median(e)), 'late': float(np.median(l))}
    vs = [r['mcg'][k] - r['opp'][k] for r in rows if np.isfinite(r['mcg'][k]) and np.isfinite(r['opp'][k])]
    tests[k]['mcg_more_in'] = [int(np.sum(np.array(vs) > 0)), len(vs)]
(HERE / 'results' / 'stance').mkdir(parents=True, exist_ok=True)
(HERE / 'results' / 'stance' / 'feet.json').write_text(json.dumps({'fights': rows, 'tests': tests}, indent=1, default=float))
for r in rows:
    print(r['fight'][:14].ljust(15), ' '.join(f"{k[:9]} {r['mcg'][k]:.3f}/{r['opp'][k]:.3f}" for k in ('heel_lift', 'ankle_bounce', 'springs', 'planted_share')), f"feet {r['mcg']['feet_seen_s']:.0f}s")
for k, v in tests.items():
    print(f"{k:14s} him {v['mcg']['early']:.3f}->{v['mcg']['late']:.3f} rho {v['mcg']['rho']:+.2f} p {v['mcg']['p']:.3f} | opp rho {v['opp']['rho']:+.2f} p {v['opp']['p']:.3f} | diff rho {v['diff']['rho']:+.2f} p {v['diff']['p']:.3f} | him>opp {v['mcg_more_in']}")
