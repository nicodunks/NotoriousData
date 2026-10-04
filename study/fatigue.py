"""Does he slow down after eight minutes? McGregor against himself (round 1 vs minute 8 on) and against the man in
front of him over the same minutes. Rules fixed before the results were looked at:

Fight time   cumulative live play (the fight clock on screen, live.py) inside the fight's span, rescaled so each
             fight's total equals its official length. Breaks between rounds don't count.
Fights       those that went past 9 official minutes (so there is at least a minute after minute 8): Hill,
             Holloway, Mendes, Diaz 1, Diaz 2, Nurmagomedov; a fight counts for a window only with >= 10 s of standing
             footage in it (Hill went to the ground in round 2 and drops out).
Contrast     early = minutes 0-5 (round 1); late = minute 8 to the end. One number per fight per window.
Frames       standing, both upright and apart (frames.py 'standing'), every frame (30 /s pose files).
Measures (per fighter, per window)
  steps_min    steps per minute standing. A step is a burst of the vector between the two ankles changing faster than
               1.2 torso lengths/s for >= 0.1 s, with a net change >= 8 cm. Relative to the other foot, so camera
               pans can't make a step; a two-footed glide or a hop in place is not a step.
  step_cm      median size of those steps (cm)
  heel_lift    median heel lift, both feet (toe height minus heel height over foot length): lower = flatter
  bounce_cm    hip up-down motion at 1.5-4 Hz (RMS, cm), per continuous stretch of >= 2 s: the bounce
  strikes_min  hand strikes per standing minute (fast arm extensions, either hand, the study's loose rule)
  guard_cm     wrists above the shoulder line (cm): hands dropping
  width_cm     ankle to ankle (cm)
  hip_cm       hip height above the ankles (cm): standing taller or sinking
Tests        family 'self': McGregor late - early, one-sample t over fights; family 'vs': (his change) - (his
             opponent's change), one-sample t over fights. Holm within each family; shown as a finding only if
             corrected p < 0.05 and the mean change is >= 0.4 of the frame-to-frame spread.
Also         a pooled minute-by-minute curve (3-minute moving window) for both fighters, every fight that reached
             that minute, for the picture; and early/late windows for the clips.

    python fatigue.py -> results/fatigue/fatigue.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402
import strikes as S     # noqa: E402
from live import live as clock_live  # noqa: E402

OUT = HERE / 'results' / 'fatigue'; OUT.mkdir(parents=True, exist_ok=True)
TC = json.loads((HERE / 'results' / 'stance' / 'findings.json').read_text())['torso_cm']
OFFICIAL = {'2012_hill': 527, '2013_holloway': 900, '2015_mendes': 597, '2016_diaz1': 552, '2016_diaz2': 1500, '2018_khabib': 1083,
            '2016_alvarez': 484, '2021_poirier2': 452, '2015_siver': 414, '2021_poirier3': 300}
LONG = ['2012_hill', '2013_holloway', '2015_mendes', '2016_diaz1', '2016_diaz2', '2018_khabib']
EARLY, LATE = (0, 300), (480, 1e9)
MEASURES = ['steps_min', 'step_cm', 'heel_lift', 'bounce_cm', 'strikes_min', 'guard_cm', 'width_cm', 'hip_cm']


def fight_clock(f, t):
    """Fight seconds at each video time t."""
    w = F.windows()[f]; vid = w.get('video', f)
    bt, on, _ = clock_live(vid)
    if 'span' in w: on = on & (bt >= w['span'][0]) & (bt <= w['span'][1])
    cum = np.cumsum(on) / 10.0
    if f in OFFICIAL: cum = cum * OFFICIAL[f] / cum[-1]
    return np.interp(t, bt, cum)


def steps(k, t, shot, rate):
    """(time, size in torso lengths) of each step."""
    s = M.shot_scale(k, shot)
    v = (M.P(k, 16) - M.P(k, 15)) / s[:, None]
    out = []
    r = M.runs(t, shot, max_gap=1.5 / rate)
    for g in np.unique(r):
        m = np.flatnonzero(r == g)
        if len(m) < rate: continue
        vv = v[m]
        if np.isnan(vv).mean() > .2: continue
        ok = ~np.isnan(vv[:, 0])
        vv = np.stack([np.interp(np.arange(len(m)), np.flatnonzero(ok), vv[ok, d]) for d in (0, 1)], 1)
        vv = np.stack([np.convolve(vv[:, d], np.ones(3) / 3, mode='same') for d in (0, 1)], 1)
        sp = np.linalg.norm(np.gradient(vv, axis=0), axis=1) * rate
        on = sp > 1.2
        on[:2] = on[-2:] = False
        d = np.diff(np.r_[0, on.astype(int), 0]); st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        # merge bursts closer than 0.1 s
        merged = []
        for a, b in zip(st, en):
            if merged and a - merged[-1][1] < .1 * rate: merged[-1][1] = b
            else: merged.append([a, b])
        for a, b in merged:
            if b - a < .1 * rate: continue
            a0, b0 = max(0, a - 1), min(len(m) - 1, b)
            size = float(np.linalg.norm(vv[b0] - vv[a0]))
            if size * TC >= 8: out.append((float(t[m[a]]), size))
    return out


def bounce(t, hip_y, shot, rate):
    """Hip vertical RMS in 1.5-4 Hz per continuous run of >= 2 s (2 s Welch segments), as (rms, seconds)."""
    from scipy.signal import welch
    out = []
    r = M.runs(t, shot, max_gap=1.5 / rate)
    for g in np.unique(r):
        m = np.flatnonzero(r == g); y = hip_y[m]
        if len(m) < 2 * rate or np.isnan(y).mean() > .1: continue
        y = np.interp(np.arange(len(y)), np.flatnonzero(~np.isnan(y)), y[~np.isnan(y)])
        y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 2), np.arange(len(y)))
        f, pxx = welch(y, fs=rate, nperseg=int(min(len(y), 2 * rate)))
        sel = (f >= 1.5) & (f <= 4)
        out.append((float(np.sqrt(np.trapezoid(pxx[sel], f[sel]))), len(m) / rate))
    return out


def per_window(d, who, ft, win, min_s=10):
    me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
    sel = (ft >= win[0]) & (ft < win[1])
    if sel.sum() < min_s * d['rate']: return None                  # at least 10 s of standing footage
    t, shot = d['t'][sel], d['shot'][sel]; k, o = me[sel], other[sel]
    rate = d['rate']
    q = M.per_frame(k, o, shot, t)
    secs = M.runs(t, shot, max_gap=1.5 / rate)
    standing_s = sum((np.sum(secs == g) / rate) for g in np.unique(secs) if np.sum(secs == g) >= rate)
    st = steps(k, t, shot, rate)
    lift = []
    for a, toe, heel in [(15, 17, 19), (16, 20, 22)]:
        T, H = M.P(k, toe), M.P(k, heel); foot = np.linalg.norm(H - T, axis=-1)
        ok = np.isfinite(foot) & (foot > .12 * q['scale'])
        lift.append(np.where(ok, (T[:, 1] - H[:, 1]) / foot, np.nan))
    b = bounce(t, q['hip_y'], shot, rate)
    orear = None
    n_strikes = 0
    for hand in ('left', 'right'):
        n_strikes += len(S.extensions(k, o, t, shot, hand, rate, strict=False))
    return {'standing_s': standing_s,
            'steps_min': len(st) / standing_s * 60 if standing_s else np.nan,
            'step_cm': float(np.median([z for _, z in st]) * TC) if len(st) >= 5 else np.nan,
            'heel_lift': float(np.nanmedian(np.concatenate(lift))),
            'bounce_cm': float(np.average([x for x, _ in b], weights=[n for _, n in b]) * TC) if b else np.nan,
            'strikes_min': n_strikes / standing_s * 60 if standing_s else np.nan,
            'guard_cm': float(np.nanmedian(q['guard']) * TC), 'width_cm': float(np.nanmedian(q['stance_width']) * TC),
            'hip_cm': float(np.nanmedian(q['hip_height']) * TC), 'step_times': [round(x, 2) for x, _ in st]}


def main():
    W = F.windows()
    res = {'fights': [], 'curve': {}}
    curve = {'mcg': {m: [] for m in MEASURES}, 'opp': {m: [] for m in MEASURES}}
    for f in sorted(OFFICIAL, key=lambda f: W[f]['date']):
        d = F.load(f, keep='standing', suffix='_30')
        if not d['n']: continue
        ft = fight_clock(f, d['t'])
        row = {'fight': f, 'date': W[f]['date'], 'opponent': W[f]['opponent'], 'official_s': OFFICIAL[f], 'long': f in LONG}
        if f in LONG:
            for who in ('mcg', 'opp'):
                row[who] = {'early': per_window(d, who, ft, EARLY), 'late': per_window(d, who, ft, LATE)}
        # minute-by-minute (each fight, every minute it reached), for the curve
        row['minutes'] = {}
        for mn in range(int(OFFICIAL[f] // 60) + 1):
            for who in ('mcg', 'opp'):
                r = per_window(d, who, ft, (mn * 60 - 60, mn * 60 + 120))          # 3-minute window centred on the minute
                if r: row['minutes'].setdefault(who, {})[mn] = {m: r[m] for m in MEASURES}
        # 30-second chunks of fight time (>= 8 s standing each), for the exploratory slope over the whole fight
        row['chunks'] = {}
        for who in ('mcg', 'opp'):
            ch = []
            for c0 in range(0, int(OFFICIAL[f]), 30):
                r = per_window(d, who, ft, (c0, c0 + 30), min_s=8)
                if r: ch.append([c0 + 15] + [r[m] for m in MEASURES])
            row['chunks'][who] = ch
        res['fights'].append(row)
        print(f, 'standing min', round(d['n'] / d['rate'] / 60, 1), flush=True)
    # tests
    fams = {'self': [], 'vs': []}
    tests = {}
    for m in MEASURES:
        rows = [r for r in res['fights'] if r['long'] and r['mcg']['early'] and r['mcg']['late'] and r['opp']['early'] and r['opp']['late']]
        dm = np.array([r['mcg']['late'][m] - r['mcg']['early'][m] for r in rows])
        do = np.array([r['opp']['late'][m] - r['opp']['early'][m] for r in rows])
        ok = np.isfinite(dm) & np.isfinite(do)
        dm, do = dm[ok], do[ok]
        spread = np.nanstd([r[w][e][m] for r in rows for w in ('mcg', 'opp') for e in ('early', 'late')], ddof=1)
        for fam, x in (('self', dm), ('vs', dm - do)):
            tt = ttest_1samp(x, 0)
            tests[f'{fam}_{m}'] = {'family': fam, 'measure': m, 'n': int(len(x)), 'mean_change': float(np.mean(x)), 'p': float(tt.pvalue),
                                   'down_in': int((x < 0).sum()), 'effect_sd': float(np.mean(x) / spread) if spread else 0.0,
                                   'mcg_early': float(np.mean([r['mcg']['early'][m] for r in rows])), 'mcg_late': float(np.mean([r['mcg']['late'][m] for r in rows])),
                                   'opp_early': float(np.mean([r['opp']['early'][m] for r in rows])), 'opp_late': float(np.mean([r['opp']['late'][m] for r in rows])),
                                   'per_fight': [(r['fight'], r['date'], r['mcg']['early'][m], r['mcg']['late'][m], r['opp']['early'][m], r['opp']['late'][m]) for r in rows]}
            fams[fam].append(f'{fam}_{m}')
    for fam, ks in fams.items():
        ps = np.array([tests[k]['p'] for k in ks]); order = np.argsort(ps); run = 0; adj = np.empty(len(ps))
        for r_, i in enumerate(order): run = max(run, min(1, (len(ps) - r_) * ps[i])); adj[i] = run
        for k, a in zip(ks, adj):
            tests[k]['holm_p'] = float(a); tests[k]['reported'] = bool(a < .05 and abs(tests[k]['effect_sd']) >= .4)
    res['tests'] = tests
    # exploratory (added after the window test came back null): per fight, the least-squares slope of each measure
    # over fight time (per 5 minutes), using every 30-s chunk; fights spanning >= 5 minutes of chunks; his slope,
    # and his minus his opponent's; one-sample t over fights, Holm within each family.
    expl = {}
    for fam in ('self', 'vs'):
        ks = []
        for j, m in enumerate(MEASURES):
            xs = []
            for r in res['fights']:
                sl = {}
                for who in ('mcg', 'opp'):
                    a = np.array(r['chunks'][who], float) if r['chunks'][who] else np.zeros((0, len(MEASURES) + 1))
                    a = a[np.isfinite(a[:, j + 1])] if len(a) else a
                    if len(a) >= 4 and a[:, 0].max() - a[:, 0].min() >= 300:
                        sl[who] = np.polyfit(a[:, 0] / 300, a[:, j + 1], 1)[0]
                if 'mcg' in sl and (fam == 'self' or 'opp' in sl):
                    xs.append((r['fight'], sl['mcg'] if fam == 'self' else sl['mcg'] - sl['opp']))
            x = np.array([v for _, v in xs])
            spread = np.nanstd([c[j + 1] for r in res['fights'] for w in ('mcg', 'opp') for c in r['chunks'][w]], ddof=1)
            tt = ttest_1samp(x, 0) if len(x) >= 3 else None
            expl[f'{fam}_{m}'] = {'family': fam, 'measure': m, 'n': len(x), 'slope_per_5min': float(np.mean(x)) if len(x) else np.nan,
                                  'p': float(tt.pvalue) if tt else np.nan, 'down_in': int((x < 0).sum()),
                                  'effect_sd': float(np.mean(x) / spread) if spread else 0.0, 'per_fight': xs}
            ks.append(f'{fam}_{m}')
        ps = np.array([expl[k]['p'] for k in ks]); order = np.argsort(ps); run = 0; adj = np.empty(len(ps))
        for r_, i in enumerate(order): run = max(run, min(1, (len(ps) - r_) * ps[i])); adj[i] = run
        for k, a in zip(ks, adj):
            expl[k]['holm_p'] = float(a); expl[k]['passes'] = bool(a < .05 and abs(expl[k]['effect_sd']) >= .4)
    res['slopes'] = expl
    for k, v in expl.items():
        print(f"SLOPE {k:16s} {v['slope_per_5min']:+7.2f} per 5 min  down {v['down_in']}/{v['n']} p {v['p']:.3f} holm {v['holm_p']:.3f} d {v['effect_sd']:+.2f} {'PASSES' if v['passes'] else ''}")
    (OUT / 'fatigue.json').write_text(json.dumps(res, indent=1, default=float).replace('NaN', 'null'))
    for k, v in tests.items():
        print(f"{k:18s} him {v['mcg_early']:7.2f} -> {v['mcg_late']:7.2f} | opp {v['opp_early']:7.2f} -> {v['opp_late']:7.2f} | change {v['mean_change']:+.2f} down {v['down_in']}/{v['n']} p {v['p']:.3f} holm {v['holm_p']:.3f} d {v['effect_sd']:+.2f} {'REPORTED' if v['reported'] else ''}")


if __name__ == '__main__':
    main()
