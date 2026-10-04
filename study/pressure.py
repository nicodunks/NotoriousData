"""Counter-striker or forward pressure? From every upright second (both fighters on their feet, 30 frames/s):

  advancing / retreating   share of time his hips moved toward / away from the opponent (> 0.3 torso lengths / s)
  pressure                 advancing minus retreating, his, and the same for the opponent
  counter_share            share of his hand strikes thrown within 0.7 s after an opponent attack
  first_share              share thrown with no opponent attack in the 1.5 s before (he started it)
  moving_in_share          share of his strikes thrown while his hips moved toward the opponent (in the 0.5 s before)
  backing_share            share thrown while moving away (a strike on the retreat)
  head_share               share of his hand strikes at head height

    python pressure.py -> results/left/pressure.json
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

RNG = np.random.default_rng(23)
R = {f['fight']: f for f in json.loads((HERE / 'results' / 'results.json').read_text())['fights']}
W = json.loads((HERE / 'windows.json').read_text())


def attacks(me, other, d):
    out = []
    for hand in ('left', 'right'):
        ext, Sh, El, Wr = S.arm_series(me, hand)
        for on, pk, kind in S.extensions(me, other, d['t'], d['shot'], hand, d['rate'], strict=False):
            out.append((on, pk, hand, Wr[pk, 1]))
    return sorted(out)


rows = []
for f in sorted(R, key=lambda f: R[f]['date']):
    if 'exclude' in W[f] or not (F.DATA / 'pose' / f"{W[f].get('video', f)}_30.npz").exists(): continue
    d = F.load(f, keep='upright', suffix='_30')
    minutes = d['n'] / d['rate'] / 60
    if minutes < 1: continue
    t, shot = d['t'], d['shot']
    row = {'fight': f, 'date': R[f]['date'], 'opponent': R[f]['opponent'], 'minutes': minutes}
    sides = {}
    for who in ('mcg', 'opp'):
        me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
        q = M.per_frame(me, other, shot, t)
        fwd = np.sign(M.midp(other, 11, 12)[:, 0] - M.midp(me, 11, 12)[:, 0])
        # footwork at 30 fps, smoothed over ~0.2 s
        v = M.footwork(t, q['hip_x'], fwd, shot, rate=d['rate'])
        v = np.convolve(np.nan_to_num(v), np.ones(5) / 5, mode='same') * np.where(np.isfinite(v), 1, np.nan)
        sides[who] = {'v': v, 'att': attacks(me, other, d), 'oshy': M.midp(other, 5, 6)[:, 1], 'ohipy': M.midp(other, 11, 12)[:, 1]}
    for who, other_who in (('mcg', 'opp'), ('opp', 'mcg')):
        v = sides[who]['v']; ok = np.isfinite(v)
        adv, ret = float(np.mean(v[ok] > .3)), float(np.mean(v[ok] < -.3))
        oth_t = np.array([t[pk] for _, pk, _, _ in sides[other_who]['att']])
        n = len(sides[who]['att']); c = fs = mi = bk = hd = 0
        for on, pk, hand, wy in sides[who]['att']:
            dt = t[on] - oth_t
            if np.any((dt >= -.1) & (dt <= .7)): c += 1
            elif not np.any((dt > 0) & (dt <= 1.5)): fs += 1
            pre = (t >= t[on] - .5) & (t <= t[on]) & (shot == shot[on])
            mv = np.nanmedian(v[pre]) if np.isfinite(v[pre]).any() else np.nan
            mi += mv > .3; bk += mv < -.3
            hd += wy < sides[who]['oshy'][pk] + .15 * (sides[who]['ohipy'][pk] - sides[who]['oshy'][pk])
        row[who] = {'advancing': adv, 'retreating': ret, 'pressure': adv - ret, 'strikes_per_min': n / minutes,
                    'counter_share': c / n if n else np.nan, 'first_share': fs / n if n else np.nan,
                    'moving_in_share': mi / n if n else np.nan, 'backing_share': bk / n if n else np.nan,
                    'head_share': hd / n if n else np.nan, 'n': n}
    rows.append(row)
days = lambda dd: (date.fromisoformat(dd) - date(2011, 1, 1)).days
KEYS = ['advancing', 'retreating', 'pressure', 'counter_share', 'first_share', 'moving_in_share', 'backing_share', 'head_share', 'strikes_per_min']
tests = {}
for k in KEYS:
    tests[k] = {}
    for who in ('mcg', 'opp', 'diff'):
        pts = [(days(r['date']), r['mcg'][k] - r['opp'][k] if who == 'diff' else r[who][k]) for r in rows
               if np.isfinite(r['mcg'][k]) and np.isfinite(r['opp'][k])]
        x, y = np.array(pts).T
        rho = spearmanr(x, y)[0]; null = np.array([spearmanr(x, RNG.permutation(y))[0] for _ in range(4000)])
        e, l = y[x < days('2016-01-01')], y[x >= days('2016-01-01')]
        tests[k][who] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'early': float(np.median(e)), 'late': float(np.median(l)), 'n': len(y)}
    mo = [r['mcg'][k] - r['opp'][k] for r in rows]
    tests[k]['mcg_more_in'] = [int(np.sum(np.array(mo) > 0)), len(mo)]
(HERE / 'results' / 'left' / 'pressure.json').write_text(json.dumps({'fights': rows, 'tests': tests}, indent=1, default=float))
for r in rows:
    m, o = r['mcg'], r['opp']
    print(r['fight'][:14].ljust(15), f"adv {m['advancing']:.2f}/{o['advancing']:.2f} ret {m['retreating']:.2f}/{o['retreating']:.2f} ctr {m['counter_share']:.2f}/{o['counter_share']:.2f} first {m['first_share']:.2f}/{o['first_share']:.2f} in {m['moving_in_share']:.2f} back {m['backing_share']:.2f} head {m['head_share']:.2f}")
for k, v in tests.items():
    print(f"{k:16s} him {v['mcg']['early']:.2f}->{v['mcg']['late']:.2f} rho {v['mcg']['rho']:+.2f} p {v['mcg']['p']:.3f} | opp {v['opp']['early']:.2f}->{v['opp']['late']:.2f} rho {v['opp']['rho']:+.2f} p {v['opp']['p']:.3f} | diff rho {v['diff']['rho']:+.2f} p {v['diff']['p']:.3f} | him>opp {v['mcg_more_in']}")
