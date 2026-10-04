"""Strike selection over the career: every fast extension of either arm (the loose punch rule, so jabs and hooks
count), by hand and shape, plus kicks, per upright minute, McGregor and opponents in the same footage.

  rear straight / rear hook / lead straight (jab) / lead hook, by wrist path (straight if displacement >= 75 % of
  distance travelled); target height from the fist at full extension against the opponent's shoulder line
  (above = head, below = body); kicks from the standing study's kick counter.

    python selection.py -> results/left/selection.json
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

RNG = np.random.default_rng(17)
R = {f['fight']: f for f in json.loads((HERE / 'results' / 'results.json').read_text())['fights']}
W = json.loads((HERE / 'windows.json').read_text())
rows = []
for f in sorted(R, key=lambda f: R[f]['date']):
    if not (F.DATA / 'pose' / f"{W[f].get('video', f)}_30.npz").exists() or 'exclude' in W[f]: continue
    d = F.load(f, keep='upright', suffix='_30')
    minutes = d['n'] / d['rate'] / 60
    if minutes < 1: continue
    row = {'fight': f, 'date': R[f]['date'], 'opponent': R[f]['opponent'], 'minutes': minutes}
    for who in ('mcg', 'opp'):
        me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
        q = M.per_frame(me, other, d['shot'], d['t'])
        south = np.nanmean(q['orthodox'] == -1) > .5 if np.isfinite(q['orthodox']).any() else who == 'mcg'
        rear, lead = ('left', 'right') if south else ('right', 'left')
        osh = M.midp(other, 5, 6)[:, 1]; ohip = M.midp(other, 11, 12)[:, 1]
        c = {}
        for hand, role in ((rear, 'rear'), (lead, 'lead')):
            ext, Sh, El, Wr = S.arm_series(me, hand)
            for on, pk, kind in S.extensions(me, other, d['t'], d['shot'], hand, d['rate'], strict=False):
                c[f'{role} {"straight" if kind == "straight" else "hook"}'] = c.get(f'{role} {"straight" if kind == "straight" else "hook"}', 0) + 1
                tgt = 'head' if Wr[pk, 1] < osh[pk] + .15 * (ohip[pk] - osh[pk]) else 'body'
                c[f'to {tgt}'] = c.get(f'to {tgt}', 0) + 1
        c['kicks'] = R[f][who]['kicks_per_min'] * R[f]['standing_seconds'] / 60 if np.isfinite(R[f][who].get('kicks_per_min', np.nan)) else 0
        row[who] = {k: v / minutes for k, v in c.items()}
        hands = sum(v for k, v in c.items() if k.startswith(('rear', 'lead')))
        row[who]['rear_share'] = (c.get('rear straight', 0) + c.get('rear hook', 0)) / hands if hands else np.nan
        row[who]['body_share'] = c.get('to body', 0) / hands if hands else np.nan
        row[who]['total'] = hands / minutes
    rows.append(row)
days = lambda d: (date.fromisoformat(d) - date(2011, 1, 1)).days
KEYS = ['total', 'rear straight', 'rear hook', 'lead straight', 'lead hook', 'kicks', 'rear_share', 'body_share']
tests = {}
for k in KEYS:
    tests[k] = {}
    for who in ('mcg', 'opp'):
        pts = [(days(r['date']), r[who].get(k, 0.0)) for r in rows if np.isfinite(r[who].get(k, 0.0))]
        x, y = np.array(pts).T
        rho = spearmanr(x, y)[0]; null = np.array([spearmanr(x, RNG.permutation(y))[0] for _ in range(4000)])
        e, l = y[x < days('2016-01-01')], y[x >= days('2016-01-01')]
        tests[k][who] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'early': float(np.median(e)), 'late': float(np.median(l)), 'n': len(y)}
    mo = [r['mcg'].get(k, 0) - r['opp'].get(k, 0) for r in rows]
    tests[k]['mcg_more_in'] = [int(np.sum(np.array(mo) > 0)), len(mo)]
(HERE / 'results' / 'left' / 'selection.json').write_text(json.dumps({'fights': rows, 'tests': tests}, indent=1, default=float))
for r in rows:
    m = r['mcg']; print(r['fight'][:14].ljust(15), f"{r['minutes']:.1f}m", ' '.join(f"{k[:9]} {m.get(k, 0):.1f}" for k in ('rear straight', 'rear hook', 'lead straight', 'lead hook', 'kicks')), f"rear {m['rear_share']:.2f} body {m['body_share']:.2f}")
for k, v in tests.items():
    print(f"{k:14s} him {v['mcg']['early']:.2f}->{v['mcg']['late']:.2f} rho {v['mcg']['rho']:+.2f} p {v['mcg']['p']:.3f} | opp {v['opp']['early']:.2f}->{v['opp']['late']:.2f} rho {v['opp']['rho']:+.2f} p {v['opp']['p']:.3f} | him>opp {v['mcg_more_in']}")
