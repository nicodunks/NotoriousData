"""The stance, in depth: every standing frame (10 per second, both fighters upright and apart), McGregor and the
opponent in the same frames.

Measures (lengths in the fighter's own torso lengths, angles in degrees, all from the 2-D broadcast view):
  width        ankle to ankle                       length    lead ankle ahead of rear, toward the opponent
  crouch       hip height above the ankles (lower = deeper)
  lead_knee    lead knee angle (180 = straight)      rear_knee rear knee angle
  lean         torso tilt toward the opponent       head_fwd  head ahead of the hips
  hunch        nose height above the shoulder line (lower = shoulders up, chin down)
  weight       hips between the feet: 0 over the rear foot, 1 over the lead foot
  guard        wrists above the shoulder line       reach     lead wrist ahead of the shoulders
  heel         rear heel lift                       bounce    hip motion at 1.5-4 Hz (RMS)
  step_speed   median hip speed (torso lengths / s) range     hip-to-hip distance

    python stance_deep.py   -> results/stance/stance.json, results/stance/clouds.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, theilslopes

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import metrics as M     # noqa: E402

OUT = HERE / 'results' / 'stance'
RNG = np.random.default_rng(21)
W = json.loads((HERE / 'windows.json').read_text())
R = json.loads((HERE / 'results' / 'results.json').read_text())
SPLIT = '2016-01-01'


def ang(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), -1, 1)))


def measures(me, other, t, shot):
    q = M.per_frame(me, other, shot, t)
    s = q['scale']
    hip, ohip = M.midp(me, 11, 12), M.midp(other, 11, 12)
    fwd = np.sign(ohip[:, 0] - hip[:, 0])
    la, ra = M.P(me, 15), M.P(me, 16)
    south = q['orthodox'] == -1
    lead_a = np.where(south[:, None], ra, la); rear_a = np.where(south[:, None], la, ra)
    lead_k = np.where(south[:, None], M.P(me, 14), M.P(me, 13)); rear_k = np.where(south[:, None], M.P(me, 13), M.P(me, 14))
    lead_h = np.where(south[:, None], M.P(me, 12), M.P(me, 11)); rear_h = np.where(south[:, None], M.P(me, 11), M.P(me, 12))
    side_ok = np.isfinite(q['orthodox'])
    sh = M.midp(me, 5, 6)
    nose = M.P(me, 0)
    span = (lead_a[:, 0] - rear_a[:, 0]) * fwd
    m = {
        'width': q['stance_width'], 'length': q['stance_length'], 'crouch': q['hip_height'], 'lean': q['lean'],
        'head_fwd': q['head_forward'], 'guard': q['guard'], 'reach': q['lead_reach'], 'heel': q['any_heel_up'],
        'range': q['range'],
        'lead_knee': np.where(side_ok, ang(lead_h, lead_k, lead_a), np.nan),
        'rear_knee': np.where(side_ok, ang(rear_h, rear_k, rear_a), np.nan),
        'hunch': (sh[:, 1] - nose[:, 1]) / s,
        # lead shoulder above the rear one (+ = lead shoulder up, the boxer's shoulder over the chin)
        'shoulder': np.where(side_ok, (np.where(south, M.P(me, 5)[:, 1], M.P(me, 6)[:, 1]) - np.where(south, M.P(me, 6)[:, 1], M.P(me, 5)[:, 1])) / s, np.nan),
        'weight': np.where(side_ok & (span > .3 * s), ((hip[:, 0] - rear_a[:, 0]) * fwd) / np.where(span > 0, span, np.nan), np.nan),
    }
    v = M.footwork(t, q['hip_x'], fwd, shot)
    m['step_speed'] = np.abs(v)
    # blade: how far the shoulder line is turned from square toward the opponent, in degrees (90 = fully side-on,
    # karate; 0 = shoulders squared, boxing). From a side-on camera the shoulders' horizontal separation is
    # W cos(theta), theta the angle between the shoulder line and the line to the opponent, W the fighter's full
    # shoulder width (95th percentile of his 2-D shoulder distance in this fight). Side-on shots only.
    qo = M.per_frame(other, me, shot, t)
    ratio = s / qo['scale']; sep = np.abs(q['hip_x'] * s - qo['hip_x'] * qo['scale']) / ((s + qo['scale']) / 2)
    # strictly side-on for this one: same size on screen, well apart, and level (the line between them runs across
    # the picture, not into it), otherwise a squared stance seen from behind reads as bladed
    dy = np.abs(M.midp(me, 11, 12)[:, 1] - M.midp(other, 11, 12)[:, 1]) / s
    side_on = (ratio > .9) & (ratio < 1.11) & (sep > 1.8) & (dy < .25)
    ls_, rs_ = M.P(me, 5), M.P(me, 6)
    full = np.nanpercentile(np.linalg.norm(ls_ - rs_, axis=-1) / s, 95) if np.isfinite(ls_[:, 0]).sum() > 50 else np.nan
    m['blade'] = np.where(side_on, np.degrees(np.arcsin(np.clip(np.abs(ls_[:, 0] - rs_[:, 0]) / s / full, 0, 1))), np.nan)
    return m, q


def summarise(m, q, t, shot):
    s = {k: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan for k, v in m.items()}
    b = M.bounce(t, q['hip_y'], shot)
    s['bounce'] = float(np.average([r for r, _ in b], weights=[d for _, d in b])) if b else np.nan
    return s


def runs(t, shot, k, other, length=30):
    """Continuous 3-second standing stretches, centred on the stretch's median hip, scaled to its median torso,
    opponent to the right. Returns a list of (length, 23, 2)."""
    r = M.runs(t, shot)
    out = []
    for g in np.unique(r):
        idx = np.flatnonzero(r == g)
        for a in range(0, len(idx) - length + 1, length):
            seg = idx[a:a + length]
            kk = k[seg]; oo = other[seg]
            hip = M.midp(kk, 11, 12); s = np.nanmedian(np.linalg.norm(M.midp(kk, 5, 6) - hip, axis=-1))
            if not np.isfinite(s) or np.isnan(hip).mean() > .1: continue
            c = np.nanmedian(hip, 0)
            fwd = np.sign(np.nanmedian(M.midp(oo, 11, 12)[:, 0]) - c[0])
            p = (np.stack([M.P(kk, i) for i in range(23)], 1) - c) / s
            p[..., 0] *= fwd
            out.append(p)
    return out


def main():
    global fights, clouds, tests
    fights, clouds = [], {'early': {'runs': [], 'frames': []}, 'late': {'runs': [], 'frames': []}}
    for f in R['fights']:
        p = HERE / 'results' / 'perframe' / f"{f['fight']}.npz"
        if not p.exists() or f['standing_seconds'] < 30: continue
        d = np.load(p); t, shot = d['t'], d['shot']
        mk, ok = d['mcg_kps'].astype(np.float32), d['opp_kps'].astype(np.float32)
        mm, mq = measures(mk, ok, t, shot); om, oq = measures(ok, mk, t, shot)
        row = {'fight': f['fight'], 'date': f['date'], 'opponent': f['opponent'], 'seconds': f['standing_seconds'],
               'mcg': summarise(mm, mq, t, shot), 'opp': summarise(om, oq, t, shot), 'mcg_ci': {}, 'opp_ci': {}}
        # side-on view only: both fighters about the same size on screen (same distance from the camera) and well apart
        dx = np.abs(mq['hip_x'] * mq['scale'] - oq['hip_x'] * oq['scale']); ratio = mq['scale'] / oq['scale']
        side = (dx > 1.4 * (mq['scale'] + oq['scale']) / 2) & (ratio > .85) & (ratio < 1.18)
        row['side_share'] = float(side.mean())
        row['mcg_side'] = {k: float(np.nanmedian(v[side])) if np.isfinite(v[side]).sum() >= 20 else np.nan for k, v in mm.items()}
        row['opp_side'] = {k: float(np.nanmedian(v[side])) if np.isfinite(v[side]).sum() >= 20 else np.nan for k, v in om.items()}
        blocks = np.floor(t / 10).astype(int); ub = np.unique(blocks)
        for who, mmm in (('mcg', mm), ('opp', om)):
            for k, v in mmm.items():
                bs = []
                for _ in range(300):
                    idx = np.concatenate([np.flatnonzero(blocks == b) for b in RNG.choice(ub, len(ub))])
                    x = v[idx]; bs.append(np.nanmedian(x) if np.isfinite(x).any() else np.nan)
                row[f'{who}_ci'][k] = np.nanpercentile(bs, [2.5, 97.5]).tolist()
            row[f'{who}_ci']['bounce'] = f.get(f'{who}_ci', {}).get('bounce', [np.nan, np.nan])
        fights.append(row)
        era = 'early' if f['date'] < SPLIT else 'late'
        south = mq['orthodox'] == -1                          # his own southpaw frames only, so the figures line up
        rs = runs(t[south], shot[south], mk[south], ok[south]) if south.sum() > 60 else []
        clouds[era]['runs'] += [(f['fight'], r) for r in rs]
        hip = M.midp(mk, 11, 12); sc = mq['scale']; fwd = np.sign(M.midp(ok, 11, 12)[:, 0] - hip[:, 0])
        pts = (np.stack([M.P(mk, i) for i in range(23)], 1) - hip[:, None]) / sc[:, None, None]; pts[..., 0] *= fwd[:, None]
        sel = np.flatnonzero(south & np.isfinite(pts[:, 15, 0]) & np.isfinite(pts[:, 16, 0]))
        clouds[era]['frames'] += [pts[i] for i in RNG.choice(sel, min(40, len(sel)), replace=False)] if len(sel) else []

    KEYS = ['width', 'length', 'crouch', 'lead_knee', 'rear_knee', 'lean', 'head_fwd', 'hunch', 'weight', 'guard', 'reach',
            'heel', 'bounce', 'step_speed', 'range']
    days = lambda d: (date.fromisoformat(d) - date(2011, 1, 1)).days
    tests = {}
    for k in KEYS:
        tests[k] = {}
        for who in ('mcg', 'opp', 'diff'):
            pts = [(days(r['date']), r['mcg'][k] - r['opp'][k] if who == 'diff' else r[who][k]) for r in fights
                   if np.isfinite(r['mcg'][k]) and np.isfinite(r['opp'][k])]
            x, y = np.array(pts).T
            rho = spearmanr(x, y)[0]
            null = np.array([spearmanr(x, RNG.permutation(y))[0] for _ in range(4000)])
            sl, ic, lo, hi = theilslopes(y, x)
            e = y[x < days(SPLIT)]; l = y[x >= days(SPLIT)]
            tests[k][who] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'n': len(y),
                             'slope_per_year': float(sl * 365.25), 'intercept': float(ic),
                             'early': float(np.median(e)), 'late': float(np.median(l))}
    for k in KEYS:
        if k == 'bounce': continue
        for who in ('mcg', 'diff'):
            pts = [(days(r['date']), r['mcg_side'][k] - r['opp_side'][k] if who == 'diff' else r['mcg_side'][k]) for r in fights
                   if np.isfinite(r['mcg_side'].get(k, np.nan)) and np.isfinite(r['opp_side'].get(k, np.nan))]
            if len(pts) < 6: continue
            x, y = np.array(pts).T
            rho = spearmanr(x, y)[0]; null = np.array([spearmanr(x, RNG.permutation(y))[0] for _ in range(4000)])
            tests[k][f'{who}_side'] = {'rho': float(rho), 'p': float(np.mean(np.abs(null) >= abs(rho))), 'n': len(y)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stance.json').write_text(json.dumps({'fights': fights, 'tests': tests, 'split': SPLIT}, indent=1, default=float))

    # animation data: up to 36 runs per era (fights spread evenly), every frame of each; plus median figure per era
    def pick_runs(rs, n=36):
        by = {}
        for f, r in rs: by.setdefault(f, []).append(r)
        order = []
        while len(order) < n and any(by.values()):
            for f in list(by):
                if by[f] and len(order) < n: order.append(by[f].pop(RNG.integers(len(by[f]))))
        return order
    cl = {}
    for era in ('early', 'late'):
        rr = pick_runs(clouds[era]['runs'])
        fr = np.array(clouds[era]['frames'])
        med = np.nanmedian(fr, 0)
        cl[era] = {'runs': [np.round(r, 2).tolist() for r in rr], 'median': np.round(med, 2).tolist(),
                   'n_runs_total': len(clouds[era]['runs']), 'n_frames': len(fr),
                   'frames': np.round(fr[RNG.choice(len(fr), min(260, len(fr)), replace=False)], 2).tolist()}
    (OUT / 'clouds.json').write_text(json.dumps(cl, separators=(',', ':')).replace('NaN', 'null'))
    for k in KEYS:
        t_ = tests[k]
        sd = t_.get('mcg_side', {}); dd = t_.get('diff_side', {})
        print(f"{k:11s} mcg {t_['mcg']['early']:7.2f}->{t_['mcg']['late']:7.2f} rho {t_['mcg']['rho']:+.2f} p {t_['mcg']['p']:.3f} | opp {t_['opp']['rho']:+.2f} p {t_['opp']['p']:.3f} | diff {t_['diff']['rho']:+.2f} p {t_['diff']['p']:.3f} || side-on: mcg {sd.get('rho', float('nan')):+.2f} p {sd.get('p', float('nan')):.3f} diff {dd.get('rho', float('nan')):+.2f} p {dd.get('p', float('nan')):.3f}")
    print({e: (len(cl[e]['runs']), cl[e]['n_runs_total'], cl[e]['n_frames']) for e in cl}, round((OUT / 'clouds.json').stat().st_size / 1e6, 2), 'MB')


if __name__ == '__main__':
    main()
