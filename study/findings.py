"""The stance, tested with per-fight tests and a strict reporting rule, plus our opponent control.

Units. Lengths in cm: McGregor is 175 cm; thigh + shank is 0.49 of standing height (Winter's anthropometric
table), so his leg is ~86 cm. His torso (shoulder centre to hip centre) is measured against his leg in frames where
the leg is near straight and side-on (90th percentile of leg/torso), which gives cm per torso length. Opponents are
converted with the same factor (same weight classes; stated as approximate).

Confounds regressed out, per frame, pooled over fights: zoom (torso height / frame height), view (how side-on: the two
fighters' apparent size ratio and their horizontal separation), and range. Each fight gives one number (median of the
adjusted frames). Early (2012-2015) against late (2016-2021) by Welch's t-test; Holm across every measure tested; a
change is reported only if corrected p < 0.05 AND the shift is at least 0.4 of the frame-to-frame spread.

    python findings.py -> results/stance/findings.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import metrics as M        # noqa: E402
import stance_deep as SD   # noqa: E402  (reuses its measure definitions; importing runs nothing heavy? see guard)

R = json.loads((HERE / 'results' / 'results.json').read_text())
HEIGHT_CM, LEG_FRAC = 175.0, .49
LABELS = {  # key: (plain name, unit kind, decimals)
    'width': ('Stance width (ankle to ankle)', 'cm', 0), 'length': ('Stance length (lead foot ahead of rear)', 'cm', 0),
    'crouch': ('Hip height above the ankles', 'cm', 0), 'lead_knee': ('Lead knee angle', '°', 0), 'rear_knee': ('Rear knee angle', '°', 0),
    'lean': ('Torso lean toward the opponent', '°', 1), 'head_fwd': ('Head ahead of the hips', 'cm', 0),
    'hunch': ('Neck height (nose above shoulder line)', 'cm', 0), 'shoulder': ('Lead shoulder above the rear shoulder', 'cm', 0), 'blade': ('Shoulders turned side-on (bladed)', '°', 0), 'weight': ('Hips between the feet (0 rear, 1 lead)', '', 2),
    'guard': ('Wrists above the shoulder line', 'cm', 0), 'reach': ('Lead hand ahead of the shoulders', 'cm', 0),
    'heel': ('Heel lift', 'foot', 2), 'step_speed': ('Footwork speed', 'cm/s', 0), 'range': ('Distance between hips', 'cm', 0),
}
CM = {'width', 'length', 'crouch', 'head_fwd', 'hunch', 'shoulder', 'guard', 'reach', 'step_speed', 'range'}


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * ps[i])); out[i] = run
    return out


frames = {'mcg': [], 'opp': []}
leg_ratio = []
for f in R['fights']:
    p = HERE / 'results' / 'perframe' / f"{f['fight']}.npz"
    if not p.exists() or f['standing_seconds'] < 30: continue
    d = np.load(p); t, shot = d['t'], d['shot']
    mk, ok = d['mcg_kps'].astype(np.float32), d['opp_kps'].astype(np.float32)
    for who, a, b in (('mcg', mk, ok), ('opp', ok, mk)):
        m, q = SD.measures(a, b, t, shot)
        qo = M.per_frame(b, a, shot, t)
        zoom = q['scale'] / 720.0
        size_ratio = q['scale'] / qo['scale']
        sep = np.abs(q['hip_x'] * q['scale'] - qo['hip_x'] * qo['scale']) / ((q['scale'] + qo['scale']) / 2)
        late = f['date'] >= '2016-01-01'
        frames[who].append({'fight': f['fight'], 'date': f['date'], 'late': late, 'm': m, 'cov': np.c_[zoom, size_ratio, sep]})
        if who == 'mcg':
            leg = np.nanmean(np.c_[np.linalg.norm(M.P(a, 11) - M.P(a, 13), axis=-1) + np.linalg.norm(M.P(a, 13) - M.P(a, 15), axis=-1),
                                   np.linalg.norm(M.P(a, 12) - M.P(a, 14), axis=-1) + np.linalg.norm(M.P(a, 14) - M.P(a, 16), axis=-1)], axis=1)
            side = (size_ratio > .85) & (size_ratio < 1.18) & (sep > 1.4)
            leg_ratio.append(leg[side] / q['scale'][side])
ratio = float(np.nanpercentile(np.concatenate(leg_ratio), 90))
torso_cm = HEIGHT_CM * LEG_FRAC / ratio
print(f'leg/torso {ratio:.2f} -> torso {torso_cm:.1f} cm', flush=True)

out = {'torso_cm': torso_cm, 'measures': {}}
pvals, keys = [], []
for k in LABELS:
    res = {}
    for who in ('mcg', 'opp'):
        Y, X, G = [], [], []
        for i, fr in enumerate(frames[who]):
            y = fr['m'][k]; ok = np.isfinite(y) & np.isfinite(fr['cov']).all(1)
            Y.append(y[ok]); X.append(fr['cov'][ok]); G.append(np.full(ok.sum(), i))
        Y, X, G = np.concatenate(Y), np.concatenate(X), np.concatenate(G)
        if len(Y) < 200: continue
        # fight fixed effects + covariates: the covariate slopes come from within-fight variation only
        D = np.c_[np.eye(G.max() + 1)[G], X - X.mean(0)]
        beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
        adj = Y - (X - X.mean(0)) @ beta[-X.shape[1]:]
        scale = torso_cm if k in CM else 1.0
        per = []
        for i, fr in enumerate(frames[who]):
            v = adj[G == i]
            if len(v) >= 50: per.append((fr['fight'], fr['date'], fr['late'], float(np.median(v)) * scale))
        e = [v for _, _, l, v in per if not l]; l = [v for _, _, l, v in per if l]
        sd = float(np.std(adj)) * scale
        tt = ttest_ind(l, e, equal_var=False) if len(e) >= 3 and len(l) >= 3 else None
        res[who] = {'per_fight': per, 'early': float(np.mean(e)), 'late': float(np.mean(l)),
                    'early_ci': (1.96 * np.std(e, ddof=1) / np.sqrt(len(e))), 'late_ci': (1.96 * np.std(l, ddof=1) / np.sqrt(len(l))),
                    'p': float(tt.pvalue) if tt else np.nan, 'effect_sd': float((np.mean(l) - np.mean(e)) / sd), 'frame_sd': sd}
    out['measures'][k] = res
    pvals.append(res['mcg']['p']); keys.append(k)
for k, ph in zip(keys, holm(pvals)):
    r = out['measures'][k]['mcg']
    r['holm_p'] = float(ph)
    r['reported'] = bool(ph < .05 and abs(r['effect_sd']) >= .4)
(HERE / 'results' / 'stance' / 'findings.json').write_text(json.dumps(out, indent=1, default=float))
for k in keys:
    m, o = out['measures'][k]['mcg'], out['measures'][k].get('opp', {})
    print(f"{k:11s} him {m['early']:7.1f} -> {m['late']:7.1f}  p {m['p']:.4f} holm {m['holm_p']:.3f} d {m['effect_sd']:+.2f} {'REPORTED' if m['reported'] else ''} | opp {o.get('early', np.nan):7.1f} -> {o.get('late', np.nan):7.1f} p {o.get('p', np.nan):.3f}")

# His signature: McGregor against his opponent in the same fight (paired by fight), same adjustment, same rule:
# Wilcoxon signed-rank over fights, Holm across measures, |median difference| >= 0.4 of the frame-to-frame spread.
from scipy.stats import wilcoxon  # noqa: E402
sig_p, sig_k = [], []
for k in keys:
    m, o = out['measures'][k].get('mcg'), out['measures'][k].get('opp')
    if not m or not o: continue
    om = {f: v for f, _, _, v in o['per_fight']}
    diffs = [v - om[f] for f, _, _, v in m['per_fight'] if f in om]
    if len(diffs) < 6: continue
    p = float(wilcoxon(diffs).pvalue)
    out['measures'][k]['signature'] = {'median_diff': float(np.median(diffs)), 'n': len(diffs), 'his_higher_in': int(np.sum(np.array(diffs) > 0)),
                                      'p': p, 'effect_sd': float(np.median(diffs) / ((m['frame_sd'] + o['frame_sd']) / 2)),
                                      'mcg': float(np.median([v for _, _, _, v in m['per_fight']])), 'opp': float(np.median(list(om.values())))}
    sig_p.append(p); sig_k.append(k)
for k, ph in zip(sig_k, holm(sig_p)):
    s = out['measures'][k]['signature']; s['holm_p'] = float(ph); s['reported'] = bool(ph < .05 and abs(s['effect_sd']) >= .4)
(HERE / 'results' / 'stance' / 'findings.json').write_text(json.dumps(out, indent=1, default=float))
for k in sig_k:
    s = out['measures'][k]['signature']
    print(f"SIG {k:11s} him {s['mcg']:7.1f} opp {s['opp']:7.1f} diff {s['median_diff']:+.1f} higher {s['his_higher_in']}/{s['n']} p {s['p']:.4f} holm {s['holm_p']:.3f} d {s['effect_sd']:+.2f} {'REPORTED' if s['reported'] else ''}")
