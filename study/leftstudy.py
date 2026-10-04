"""The left hand, analysed: from results/left/events.json to results/left/summary.json.

Unit of inference: the fight. McGregor's lefts are compared with his opponents' rear straights thrown in the same
fights (same cameras, same frames), with intervals from a cluster bootstrap that resamples fights.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RNG = np.random.default_rng(11)
FEATURES = [  # key, label, unit, higher means
    ('duration_ms', 'Onset to full extension', 'ms', 'slower'),
    ('peak_speed', 'Peak fist speed', 'torso lengths/s', 'faster'),
    ('reach', 'Fist ahead of his own hips at full extension', 'torso lengths', 'longer'),
    ('hip_drive', 'Hips moved forward during the punch', 'torso lengths', 'more weight forward'),
    ('lead_step', 'Lead foot stepped in', 'torso lengths', 'bigger step'),
    ('head_fwd_at_peak', 'Head ahead of hips at full extension', 'torso lengths', 'head further over the front'),
    ('range_at_onset', 'Distance between hips when the punch starts', 'torso lengths', 'farther'),
    ('lead_high_at_peak', 'Other hand height at full extension', 'torso lengths', 'other hand higher'),
    ('opp_head_snap', 'Target head moved away in the next 250 ms', 'torso lengths', 'bigger visible effect'),
]
SETUPS = ['pull counter', 'slip counter', 'counter, head still', 'caught coming in', 'behind the paw', 'straight lead']


def by_fight(evs, key):
    out = {}
    for e in evs:
        v = e.get(key)
        if v is not None and np.isfinite(v):
            out.setdefault(e['fight'], []).append(v)
    return {f: float(np.median(v)) for f, v in out.items()}


def cluster_diff(a: dict, b: dict, reps=5000):
    """Median over fights of (McGregor - opponent) per-fight medians, with a fight bootstrap interval."""
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    d = np.array([a[f] - b[f] for f in common])
    boots = [np.median(RNG.choice(d, len(d))) for _ in range(reps)]
    return {'median_diff': float(np.median(d)), 'ci': np.percentile(boots, [2.5, 97.5]).tolist(),
            'mcg_higher_in': int((d > 0).sum()), 'n_fights': len(common),
            'mcg': float(np.median([a[f] for f in common])), 'opp': float(np.median([b[f] for f in common]))}


def main():
    evs = json.loads((HERE / 'results' / 'left' / 'events.json').read_text())
    res = json.loads((HERE / 'results' / 'results.json').read_text())
    W = json.loads((HERE / 'windows.json').read_text())
    minutes = json.loads((HERE / 'results' / 'left' / 'minutes.json').read_text())
    mcg = [e for e in evs if e['who'] == 'mcg' and e['kind'] == 'straight']
    mhook = [e for e in evs if e['who'] == 'mcg' and e['kind'] == 'hook']
    opp = [e for e in evs if e['who'] == 'opp' and e['kind'] == 'straight']
    fights = sorted({e['fight'] for e in evs} | set(minutes), key=lambda f: W[f]['date'])
    out = {'n_lefts': len(mcg), 'n_hooks': len(mhook), 'n_opp_rear': len(opp), 'fights': [], 'features': {}, 'setups': {},
           'trends': {}}
    # rate per upright minute, per fight
    for f in fights:
        m = minutes.get(f, 0)
        nm = sum(e['fight'] == f for e in mcg); no = sum(e['fight'] == f for e in opp)
        sm = Counter(e['setup'] for e in mcg if e['fight'] == f)
        out['fights'].append({'fight': f, 'date': W[f]['date'], 'opponent': W[f]['opponent'], 'upright_min': m,
                              'lefts': nm, 'opp_rear': no, 'left_rate': nm / m if m else None,
                              'opp_rate': no / m if m else None,
                              'counter_share': (sm['pull counter'] + sm['slip counter'] + sm['counter, head still']) / nm if nm else None})
    rated = [f for f in out['fights'] if f['upright_min'] and f['upright_min'] >= 1]
    out['rate'] = cluster_diff({f['fight']: f['left_rate'] for f in rated}, {f['fight']: f['opp_rate'] for f in rated})
    # mechanics, McGregor's left vs the opponents' rear straight
    for key, *_ in FEATURES:
        out['features'][key] = cluster_diff(by_fight(mcg, key), by_fight(opp, key))
    # set-ups: shares, pooled and with a fight bootstrap
    for who, ev in (('mcg', mcg), ('opp', opp)):
        c = Counter(e['setup'] for e in ev); n = sum(c.values())
        fl = sorted({e['fight'] for e in ev})
        boots = {s: [] for s in SETUPS}
        for _ in range(3000):
            pick = RNG.choice(fl, len(fl))
            pool = [e for f in pick for e in ev if e['fight'] == f]
            cc = Counter(e['setup'] for e in pool); nn = max(1, len(pool))
            for s in SETUPS: boots[s].append(cc[s] / nn)
        out['setups'][who] = {s: {'n': c[s], 'share': c[s] / n if n else 0, 'ci': np.percentile(boots[s], [2.5, 97.5]).tolist()} for s in SETUPS}
    # does the set-up change the visible effect? (opponent head snap by set-up)
    eff = {}
    for s in SETUPS:
        v = [e['opp_head_snap'] for e in mcg if e['setup'] == s and e.get('opp_head_snap') is not None]
        if len(v) >= 5:
            b = [np.median(RNG.choice(v, len(v))) for _ in range(3000)]
            eff[s] = {'n': len(v), 'median': float(np.median(v)), 'ci': np.percentile(b, [2.5, 97.5]).tolist()}
    out['effect_by_setup'] = eff
    # over the career: per-fight rate, counter share, speed, duration vs date
    days = lambda f: (date.fromisoformat(W[f]['date']) - date(2011, 1, 1)).days
    for key, series in (('left_rate', {f['fight']: f['left_rate'] for f in rated}),
                        ('counter_share', {f['fight']: f['counter_share'] for f in rated if f['counter_share'] is not None and f['lefts'] >= 4}),
                        ('peak_speed', by_fight(mcg, 'peak_speed')), ('duration_ms', by_fight(mcg, 'duration_ms')),
                        ('reach', by_fight(mcg, 'reach'))):
        ks = [k for k, v in series.items() if v is not None]
        if len(ks) >= 5:
            x = np.array([days(k) for k in ks]); y = np.array([series[k] for k in ks])
            rho = spearmanr(x, y)[0]
            null = [spearmanr(x, RNG.permutation(y))[0] for _ in range(5000)]
            out['trends'][key] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'n': len(ks),
                                  'points': [{'fight': k, 'date': W[k]['date'], 'value': float(series[k])} for k in sorted(ks, key=days)]}
    (HERE / 'results' / 'left' / 'summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('n_lefts', 'n_hooks', 'n_opp_rear', 'rate')}, indent=1))
    for k, v in out['features'].items():
        if v: print(f"{k:20s} mcg {v['mcg']:7.3f} opp {v['opp']:7.3f} diff {v['median_diff']:+.3f} ci {np.round(v['ci'], 3)} higher in {v['mcg_higher_in']}/{v['n_fights']}")
    for w in ('mcg', 'opp'):
        print(w, {s: f"{v['share']:.0%}" for s, v in out['setups'][w].items()})
    print('effect', {s: round(v['median'], 3) for s, v in eff.items()})
    print('trends', {k: (round(v['rho'], 2), round(v['p'], 3), v['n']) for k, v in out['trends'].items()})


if __name__ == '__main__':
    main()
