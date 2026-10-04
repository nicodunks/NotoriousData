"""When does he fade? A flat-then-declining model, fitted to every 30-second chunk of standing footage across his
fights, with a separate level per fight: y = a_fight + b * max(0, t - tau). tau is chosen by least squares over
minutes 2 to 15 (half-minute steps); its uncertainty by resampling whole fights (2,000 times). Run for McGregor's
steps a minute and hands height, and for the same measures relative to his opponent in the same chunk.

    python breakpoint.py -> results/fatigue/breakpoint.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FA = json.loads((HERE / 'results' / 'fatigue' / 'fatigue.json').read_text())
MEAS = ['steps_min', 'step_cm', 'heel_lift', 'bounce_cm', 'strikes_min', 'guard_cm', 'width_cm', 'hip_cm']
GRID = np.arange(2, 15.01, .5)
RNG = np.random.default_rng(5)


def data(key, rel):
    j = MEAS.index(key) + 1; out = []
    for r in FA['fights']:
        m = {round(c[0]): c[j] for c in r['chunks'].get('mcg', []) if c[j] is not None and np.isfinite(c[j])}
        o = {round(c[0]): c[j] for c in r['chunks'].get('opp', []) if c[j] is not None and np.isfinite(c[j])}
        for t, v in m.items():
            if rel and t not in o: continue
            out.append((r['fight'], t / 60, v - (o[t] if rel else 0)))
    return out


def fit(rows, tau):
    fights = sorted({f for f, _, _ in rows}); fi = {f: i for i, f in enumerate(fights)}
    X = np.zeros((len(rows), len(fights) + 1)); y = np.array([v for _, _, v in rows])
    for i, (f, t, _) in enumerate(rows): X[i, fi[f]] = 1; X[i, -1] = max(0, t - tau)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum()), float(beta[-1])


def best(rows):
    sse = [fit(rows, tau) for tau in GRID]
    k = int(np.argmin([s for s, _ in sse]))
    return float(GRID[k]), sse[k][1], [s for s, _ in sse]


res = {}
for key in ('steps_min', 'guard_cm'):
    for rel in (False, True):
        rows = data(key, rel)
        # only fights that reached past minute 6, so a late slope is identifiable
        longf = {f for f, t, _ in rows if t > 6}; rows = [r for r in rows if r[0] in longf]
        tau, slope, curve = best(rows)
        fights = sorted(longf); boots = []
        for _ in range(2000):
            pick = RNG.choice(fights, len(fights)); bs = [r for f in pick for r in rows if r[0] == f]
            if len({f for f, _, _ in bs}) < 3: continue
            boots.append(best(bs)[:2])
        bt = np.array(boots)
        flat = fit(rows, 99)[0]                                   # no decline at all within the fight
        res[f"{key}{'_vs_opp' if rel else ''}"] = {'tau_min': tau, 'slope_per_min': slope, 'tau_ci': np.percentile(bt[:, 0], [10, 90]).tolist(),
                                                   'slope_ci': np.percentile(bt[:, 1], [2.5, 97.5]).tolist(), 'share_slope_neg': float((bt[:, 1] < 0).mean()),
                                                   'n_chunks': len(rows), 'n_fights': len(fights), 'sse_curve': curve, 'sse_flat': flat,
                                                   'points': [(f, round(t, 2), round(v, 2)) for f, t, v in rows]}
        v = res[f"{key}{'_vs_opp' if rel else ''}"]
        print(f"{key:10s} {'vs opp' if rel else 'own   '} break at {tau:4.1f} min (80% {v['tau_ci'][0]:.1f}-{v['tau_ci'][1]:.1f}) slope {slope:+.2f}/min after "
              f"(95% {v['slope_ci'][0]:+.2f} to {v['slope_ci'][1]:+.2f}, negative in {v['share_slope_neg']:.0%}) chunks {len(rows)} fights {len(fights)}")
(HERE / 'results' / 'fatigue' / 'breakpoint.json').write_text(json.dumps(res, default=float))
