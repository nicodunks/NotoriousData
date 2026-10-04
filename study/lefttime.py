"""The left over time: per-fight medians of every punch measurement, McGregor and opponents, with intervals
(bootstrap over punches within a fight) and the career trend (Spearman vs date, permutation p).
Also tests early (to 2015) vs late (2016 on) on the pooled punches, fights resampled.

    python lefttime.py  -> results/left/time.json
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
RNG = np.random.default_rng(5)
KEYS = ['peak_speed', 'duration_ms', 'reach', 'hip_drive', 'lead_step', 'lead_knee_onset', 'lead_knee_peak',
        'rear_knee_onset', 'rear_knee_peak', 'lean_onset', 'lean_peak', 'lean_change', 'shoulder_turn', 'level_change',
        'stance_width_onset', 'head_fwd_at_peak', 'range_at_onset', 'lead_high_at_peak']
SPLIT = date(2016, 1, 1)

evs = json.loads((HERE / 'results' / 'left' / 'events.json').read_text())
W = json.loads((HERE / 'windows.json').read_text())
out = {'fights': [], 'trend': {}, 'era': {}}
fights = sorted({e['fight'] for e in evs}, key=lambda f: W[f]['date'])
for f in fights:
    row = {'fight': f, 'date': W[f]['date'], 'opponent': W[f]['opponent'], 'mcg': {}, 'opp': {}, 'mcg_ci': {}, 'opp_ci': {}, 'n': {}}
    for who in ('mcg', 'opp'):
        ev = [e for e in evs if e['fight'] == f and e['who'] == who and e['kind'] == 'straight']
        row['n'][who] = len(ev)
        for k in KEYS:
            v = np.array([e[k] for e in ev if e.get(k) is not None and np.isfinite(e[k])])
            if len(v) >= 2:
                row[who][k] = float(np.median(v))
                b = [np.median(RNG.choice(v, len(v))) for _ in range(1000)]
                row[f'{who}_ci'][k] = np.percentile(b, [2.5, 97.5]).tolist()
    out['fights'].append(row)
days = lambda f: (date.fromisoformat(W[f]['date']) - date(2011, 1, 1)).days
for k in KEYS:
    pts = [(days(r['fight']), r['mcg'][k]) for r in out['fights'] if k in r['mcg'] and r['n']['mcg'] >= 3]
    if len(pts) >= 6:
        x, y = np.array(pts).T
        rho = spearmanr(x, y)[0]
        null = [spearmanr(x, RNG.permutation(y))[0] for _ in range(5000)]
        out['trend'][k] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'n': len(pts)}
    # early vs late: pooled punches, bootstrap resampling fights
    mc = [e for e in evs if e['who'] == 'mcg' and e['kind'] == 'straight' and e.get(k) is not None and np.isfinite(e[k])]
    early = {f for f in fights if date.fromisoformat(W[f]['date']) < SPLIT}
    E = [e[k] for e in mc if e['fight'] in early]; L = [e[k] for e in mc if e['fight'] not in early]
    if len(E) >= 8 and len(L) >= 8:
        fe = sorted({e['fight'] for e in mc if e['fight'] in early}); fl = sorted({e['fight'] for e in mc if e['fight'] not in early})
        b = []
        for _ in range(3000):
            pe = [e[k] for g in RNG.choice(fe, len(fe)) for e in mc if e['fight'] == g]
            pl = [e[k] for g in RNG.choice(fl, len(fl)) for e in mc if e['fight'] == g]
            if pe and pl: b.append(np.median(pl) - np.median(pe))
        out['era'][k] = {'early': float(np.median(E)), 'late': float(np.median(L)), 'n_early': len(E), 'n_late': len(L),
                         'diff': float(np.median(L) - np.median(E)), 'ci': np.percentile(b, [2.5, 97.5]).tolist()}
(HERE / 'results' / 'left' / 'time.json').write_text(json.dumps(out, indent=1))
for k in KEYS:
    t = out['trend'].get(k, {}); e = out['era'].get(k)
    print(f"{k:20s} trend rho {t.get('rho', float('nan')):+.2f} p {t.get('p', float('nan')):.3f}  |  early {e['early']:7.2f} late {e['late']:7.2f} diff {e['diff']:+.2f} ci {np.round(e['ci'], 2)}" if e else k)
