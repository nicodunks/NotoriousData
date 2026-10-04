"""The study: every fight through frames.py and metrics.py, per-fight summaries with block-bootstrap intervals,
then the pre-registered tests across fights (PREREGISTRATION.md).

    python study.py            -> $OUT/results.json, $OUT/perframe/<fight>.npz
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F            # noqa: E402
import metrics as M           # noqa: E402

OUT = HERE / 'results'
SPLIT = date(2016, 3, 5)       # first Diaz fight: the split the story names (pre-registered)
MIN_STANDING = 60.0            # seconds of standing frames a fight needs to enter the tests
BLOCK = 10.0                   # bootstrap block, seconds
REPS = int(__import__('os').environ.get('REPS', 1000))
RNG = np.random.default_rng(7)

# per-fight summary of each quantity: (name, how)
HYP = {  # pre-registered: name -> (quantity, predicted sign of late - early)
    'H1 stance width': ('stance_width', -1),
    'H2 on the toes (rear heel up)': ('rear_heel_up', -1),
    'H3 bounce': ('bounce', -1),
    'H4 guard height': ('guard', +1),
    'H5 kicks per minute': ('kicks_per_min', -1),
    'H6 stance switches per minute': ('switches_per_min', -1),
    'H7 range': ('range', -1),
}
EXTRA = ['stance_length', 'any_heel_up', 'guard_lead', 'guard_rear', 'lead_reach', 'hip_height', 'lean',
         'head_forward', 'orthodox_share']


def summarize(q: dict, t, shot, minutes) -> dict:
    """One number per quantity for a set of frames."""
    s = {}
    for k in ['stance_width', 'stance_length', 'guard', 'guard_lead', 'guard_rear', 'lead_reach', 'hip_height',
              'lean', 'head_forward', 'range']:
        s[k] = float(np.nanmedian(q[k])) if np.isfinite(q[k]).any() else np.nan
    for k in ['rear_heel_up', 'any_heel_up']:
        s[k] = float(np.nanmedian(q[k])) if np.isfinite(q[k]).any() else np.nan
    o = q['orthodox']
    s['orthodox_share'] = float(np.mean(o[np.isfinite(o)] == 1)) if np.isfinite(o).any() else np.nan   # over frames where the stance was read
    b = M.bounce(t, q['hip_y'], shot)
    s['bounce'] = float(np.average([r for r, _ in b], weights=[d for _, d in b])) if b else np.nan
    s['kicks_per_min'] = M.events(q['kick_pose'], t, shot) / minutes if minutes else np.nan
    # exploratory (not pre-registered): footwork and rhythm
    s['speed'] = s['retreat_share'] = np.nan
    v = q.get('advance')
    if v is not None and np.isfinite(v).any():
        s['speed'] = float(np.nanmedian(np.abs(v)))
        moving = np.abs(v) > .3
        s['retreat_share'] = float(np.nanmean(v[moving] < 0)) if moving.any() else np.nan
    hz = M.bounce_hz(t, q['hip_y'], shot)
    s['bounce_hz'] = float(np.average([h for h, _ in hz], weights=[d for _, d in hz])) if hz else np.nan
    s['switches_per_min'] = M.switches(q['orthodox'], t, shot) / minutes if minutes else np.nan
    return s


def fight_summary(fid: str) -> dict | None:
    d = F.load(fid)
    if not d['n']:
        return {'fight': fid, 'standing_seconds': 0.0}
    t, shot = d['t'], d['shot']
    qm = M.per_frame(d['mcg'], d['opp'], shot, t)
    qo = M.per_frame(d['opp'], d['mcg'], shot, t)
    for q, other in ((qm, qo), (qo, qm)):
        fwd = np.sign(other['hip_x'] * other['scale'] - q['hip_x'] * q['scale'])
        q['advance'] = M.footwork(t, q['hip_x'], fwd, shot)
    minutes = len(t) / 10 / 60
    res = {'fight': fid, 'standing_seconds': len(t) / 10, 'live_seconds': d['live_seconds'],
           'mcg': summarize(qm, t, shot, minutes), 'opp': summarize(qo, t, shot, minutes)}
    # block bootstrap: resample 10 s blocks of standing frames, recompute both fighters' summaries together
    blocks = np.floor(t / BLOCK).astype(int)
    ub = np.unique(blocks)
    boot_m, boot_o = [], []
    for _ in range(REPS):
        pick = RNG.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        # keep time order inside each block copy, separate copies by a gap so runs/events don't join
        tt = np.concatenate([t[blocks == b] - t[blocks == b][0] + 1000 * j for j, b in enumerate(pick)])
        ss = np.concatenate([shot[blocks == b] + 100000 * j for j, b in enumerate(pick)])
        sub = lambda q: {k: v[idx] for k, v in q.items()}
        boot_m.append(summarize(sub(qm), tt, ss, len(idx) / 600))
        boot_o.append(summarize(sub(qo), tt, ss, len(idx) / 600))
    res['mcg_ci'] = {k: np.nanpercentile([b[k] for b in boot_m], [2.5, 97.5]).tolist() for k in res['mcg']}
    res['opp_ci'] = {k: np.nanpercentile([b[k] for b in boot_o], [2.5, 97.5]).tolist() for k in res['opp']}
    res['diff_ci'] = {k: np.nanpercentile([bm[k] - bo[k] for bm, bo in zip(boot_m, boot_o)], [2.5, 97.5]).tolist()
                      for k in res['mcg']}
    (OUT / 'perframe').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'perframe' / f'{fid}.npz', t=t, shot=shot, mcg_kps=d['mcg'].astype(np.float16),
                        opp_kps=d['opp'].astype(np.float16),
                        **{f'mcg_{k}': v for k, v in qm.items()}, **{f'opp_{k}': v for k, v in qo.items()})
    return res


def perm_spearman(x, y, reps=10000):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 4:
        return np.nan, np.nan, len(x)
    rho = spearmanr(x, y)[0]
    null = np.array([spearmanr(x, RNG.permutation(y))[0] for _ in range(reps)])
    return float(rho), float((np.abs(null) >= abs(rho)).mean()), len(x)


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def main():
    OUT.mkdir(exist_ok=True)
    W = F.windows()
    ready = lambda f: (F.DATA / 'pose' / f"{W[f].get('video', f)}.npz").exists() and W[f]['shorts']
    for f in W:
        if 'exclude' not in W[f] and not ready(f):
            print('not ready (pose or shorts missing):', f)
    order = sorted((f for f in W if 'exclude' not in W[f] and ready(f)), key=lambda f: W[f]['date'])
    fights = []
    for fid in order:
        print('fight', fid, flush=True)
        r = fight_summary(fid)
        r['date'] = W[fid]['date']; r['opponent'] = W[fid]['opponent']; r['official'] = W[fid].get('official', True)
        fights.append(r)
        print(' ', fid, f"{r['standing_seconds']:.0f}s standing", flush=True)
    tests = {'primary': run_tests([f for f in fights if f['standing_seconds'] >= MIN_STANDING]),
             'sensitivity_30s': run_tests([f for f in fights if f['standing_seconds'] >= 30]),
             'official_only': run_tests([f for f in fights if f['standing_seconds'] >= MIN_STANDING and f['official']])}
    (OUT / 'results.json').write_text(json.dumps({'fights': fights, 'tests': tests, 'min_standing': MIN_STANDING,
                                                  'split': SPLIT.isoformat()}, indent=1, default=float))
    for name, r in tests['primary'].items():
        print(name, {w: (round(v['spearman_rho'], 2), round(v['perm_p'], 3)) for w, v in r.items()})


def run_tests(used):
    tests = {}
    if len(used) < 4:
        return tests
    days = np.array([(date.fromisoformat(f['date']) - date(2011, 1, 1)).days for f in used], float)
    late = np.array([date.fromisoformat(f['date']) >= SPLIT for f in used], bool)
    for name, (k, sign) in HYP.items():
        for who in ['diff', 'mcg', 'opp']:
            # range is one distance between the two fighters: McGregor minus opponent is zero by construction,
            # so its primary ("diff") test is McGregor's own value
            v = np.array([(f['mcg'][k] - f['opp'][k]) if who == 'diff' and k != 'range' else f['mcg' if who == 'diff' else who][k] for f in used])
            rho, p, n = perm_spearman(days, v)
            e, l = v[~late & np.isfinite(v)], v[late & np.isfinite(v)]
            boots = [RNG.choice(l, len(l)).mean() - RNG.choice(e, len(e)).mean() for _ in range(10000)] if len(e) and len(l) else [np.nan]
            tests.setdefault(name, {})[who] = {
                'quantity': k, 'predicted_sign': sign, 'spearman_rho': rho, 'perm_p': p, 'n_fights': n,
                'early_mean': float(np.mean(e)) if len(e) else np.nan, 'late_mean': float(np.mean(l)) if len(l) else np.nan,
                'late_minus_early': float(np.mean(l) - np.mean(e)) if len(e) and len(l) else np.nan,
                'late_minus_early_ci': np.nanpercentile(boots, [2.5, 97.5]).tolist() if np.isfinite(boots).any() else [np.nan, np.nan],
                'direction_as_predicted': bool(np.sign(rho) == sign) if np.isfinite(rho) else None}
    for who in ['diff', 'mcg', 'opp']:
        adj = holm([tests[h][who]['perm_p'] for h in HYP])
        for h, a in zip(HYP, adj):
            tests[h][who]['holm_p'] = a
    return tests


if __name__ == '__main__':
    main()
