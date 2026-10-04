"""How close does the punch come? Distance management on the pull and the slip, McGregor against his opponents.

Rules fixed before running:
  punches     every committed arm extension (strikes.py, strict rule, either hand) thrown by one man at the other,
              in shots where the two are about the same size on screen (size ratio 0.8-1.25: side-on, so the
              punch travels across the picture, not toward the camera)
  the fist    the wrist plus 9 cm along the forearm (the front of the glove), in cm from the target's torso
  the head    the centre of the target's five face points; the gap is fist-to-head-centre minus a 9 cm head
              radius, so 0 is touching. 2-D: a gap is a lower bound on the real one, an overlap isn't proof of
              contact
  closest     the smallest gap from the punch's start to 0.15 s after full extension
  the move    the target's head against his own hips, from 0.3 s before the punch starts to the moment of
              closest approach: 'pull' if it went back (away from the puncher) >= 5 cm and back more than down,
              'slip' if down >= 5 cm and down more than back, else 'still'
Tests       per fight, McGregor as the target against his opponent as the target: median closest gap on pulls
            that missed (gap > 0), share of punches he pulled on, head travel on pulls. Wilcoxon over fights with
            >= 3 such punches each side.

    python evasion.py -> results/left/evasion.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402
import strikes as S     # noqa: E402

TC = json.loads((HERE / 'results' / 'stance' / 'findings.json').read_text())['torso_cm']
FIST, HEAD = 9.0, 9.0


def main():
    W = F.windows()
    res = {'punches': [], 'fights': []}
    for f in sorted(W, key=lambda f: W[f]['date']):
        if 'exclude' in W[f] or not (F.DATA / 'pose' / f"{W[f].get('video', f)}_30.npz").exists() or not W[f]['shorts']: continue
        d = F.load(f, keep='upright', suffix='_30')
        if not d['n']: continue
        t, shot, rate = d['t'], d['shot'], d['rate']
        for att, tgt in (('opp', 'mcg'), ('mcg', 'opp')):
            A, B = d[att], d[tgt]
            sa, sb = M.shot_scale(A, shot), M.shot_scale(B, shot)
            hipB = M.midp(B, 11, 12); hipA = M.midp(A, 11, 12)
            face = np.nanmean(np.stack([S.track(B, i) for i in range(5)]), 0)
            # the target's own punches (either hand, committed), for "did the pull turn into a counter?"
            b_on = np.array(sorted(t[o_] for h_ in ('left', 'right') for o_, _, _ in S.extensions(B, A, t, shot, h_, rate, strict=True)))
            for hand in ('left', 'right'):
                ext, Sh, El, Wr = S.arm_series(A, hand)
                for on, pk, kind in S.extensions(A, B, t, shot, hand, rate, strict=True):
                    r = sa[on] / sb[on]
                    if not (.8 < r < 1.25): continue
                    end = min(len(t) - 1, pk + int(.15 * rate)); st = max(0, on - int(.3 * rate))
                    if shot[end] != shot[on] or shot[st] != shot[on]: continue
                    idx = np.arange(on, end + 1)
                    fa = Wr[idx] - El[idx]; nrm = np.linalg.norm(fa, axis=1, keepdims=True)
                    tip = Wr[idx] + fa / np.where(nrm > 0, nrm, np.nan) * (FIST / TC) * sb[idx, None]
                    gap = np.linalg.norm(tip - face[idx], axis=1) / sb[idx] * TC - HEAD
                    if not np.isfinite(gap).any(): continue
                    # the counterfactual: the same fist path against the head held where it was as the punch began
                    h0 = np.nanmedian(face[st:on + 1], 0)
                    gap_still = np.linalg.norm(tip - h0, axis=1) / sb[idx] * TC - HEAD
                    gs = float(np.nanmin(gap_still)) if np.isfinite(gap_still).any() else np.nan
                    k = int(np.nanargmin(gap)); ic = idx[k]
                    away = np.sign(hipB[on, 0] - hipA[on, 0])                       # + = away from the puncher
                    rel = (face - hipB) / sb[:, None] * TC
                    base = np.nanmedian(rel[st:on + 1], 0)
                    dv = rel[ic] - base
                    back, down = float(dv[0] * away), float(dv[1])
                    move = 'pull' if back >= 5 and back > down else 'slip' if down >= 5 and down > back else 'still'
                    countered = bool(len(b_on) and np.any((b_on > t[ic]) & (b_on <= t[ic] + .6)))
                    res['punches'].append({'fight': f, 'date': W[f]['date'], 'target': tgt, 'kind': kind, 't': float(t[pk]),
                                           'gap_cm': round(float(gap[k]), 1), 'gap_still_cm': round(gs, 1) if np.isfinite(gs) else None, 'back_cm': round(back, 1), 'down_cm': round(down, 1), 'move': move, 'countered': countered})
        print(f, sum(p['fight'] == f for p in res['punches']), 'punches', flush=True)
    P = res['punches']
    tests = {}
    def per_fight(fn, min_n=3):
        rows = []
        for f in sorted({p['fight'] for p in P}):
            vals = {}
            for who in ('mcg', 'opp'):
                v = fn([p for p in P if p['fight'] == f and p['target'] == who])
                vals[who] = v
            if all(v is not None for v in vals.values()): rows.append((f, vals['mcg'], vals['opp']))
        return rows
    def med_gap_pull(ps):
        g = [p['gap_cm'] for p in ps if p['move'] == 'pull' and p['gap_cm'] > 0]
        return float(np.median(g)) if len(g) >= 3 else None
    def pull_share(ps):
        return float(np.mean([p['move'] == 'pull' for p in ps])) if len(ps) >= 5 else None
    def pull_travel(ps):
        g = [p['back_cm'] for p in ps if p['move'] == 'pull']
        return float(np.median(g)) if len(g) >= 3 else None
    def evade_share(ps):
        g = [p['gap_cm'] > 0 for p in ps if p['move'] in ('pull', 'slip')]
        return float(np.mean(g)) if len(g) >= 3 else None
    for name, fn in (('miss_gap_on_pull', med_gap_pull), ('pull_share', pull_share), ('pull_travel', pull_travel), ('evaded_when_moved', evade_share)):
        rows = per_fight(fn)
        dd = np.array([m - o for _, m, o in rows])
        p = float(wilcoxon(dd).pvalue) if len(dd) >= 5 and np.any(dd != 0) else np.nan
        tests[name] = {'n': len(rows), 'mcg': float(np.median([m for _, m, _ in rows])) if rows else np.nan,
                       'opp': float(np.median([o for _, _, o in rows])) if rows else np.nan, 'his_lower_in': int((dd < 0).sum()), 'p': p, 'per_fight': rows}
    pooled = {}
    for who in ('mcg', 'opp'):
        ps = [p for p in P if p['target'] == who]
        pulls = [p for p in ps if p['move'] == 'pull']
        miss = [p['gap_cm'] for p in pulls if p['gap_cm'] > 0]
        pooled[who] = {'punches': len(ps), 'pulls': len(pulls), 'slips': sum(p['move'] == 'slip' for p in ps),
                       'pull_miss_gap_median': float(np.median(miss)) if miss else np.nan,
                       'pull_miss_within_5cm': float(np.mean(np.array(miss) <= 5)) if miss else np.nan,
                       'pull_evaded': float(np.mean([p['gap_cm'] > 0 for p in pulls])) if pulls else np.nan,
                       'pull_back_median': float(np.median([p['back_cm'] for p in pulls])) if pulls else np.nan,
                       'gaps_on_pull': [p['gap_cm'] for p in pulls]}
    # distance management: punches that would have landed on a still head (gap_still <= 0) that he moved away from
    def saved_share(ps):
        w = [p for p in ps if p['gap_still_cm'] is not None and p['gap_still_cm'] <= 0 and p['move'] in ('pull', 'slip')]
        return float(np.mean([p['gap_cm'] > 0 for p in w])) if len(w) >= 3 else None
    def saved_margin(ps):
        w = [p['gap_cm'] for p in ps if p['gap_still_cm'] is not None and p['gap_still_cm'] <= 0 and p['move'] == 'pull' and p['gap_cm'] > 0]
        return float(np.median(w)) if len(w) >= 2 else None
    def excess(ps):
        w = [p['back_cm'] - (-p['gap_still_cm']) for p in ps if p['gap_still_cm'] is not None and p['gap_still_cm'] <= 0 and p['move'] == 'pull' and p['gap_cm'] > 0]
        return float(np.median(w)) if len(w) >= 2 else None
    def counter_rate(ps):
        w = [p['countered'] for p in ps if p['move'] in ('pull', 'slip') and p['gap_cm'] > 0]
        return float(np.mean(w)) if len(w) >= 3 else None
    for name, fn in (('counter_after_evade', counter_rate), ('saved_share', saved_share), ('saved_margin', saved_margin), ('pull_excess', excess)):
        rows = per_fight(fn)
        dd = np.array([m - o for _, m, o in rows])
        p = float(wilcoxon(dd).pvalue) if len(dd) >= 5 and np.any(dd != 0) else np.nan
        tests[name] = {'n': len(rows), 'mcg': float(np.median([m for _, m, _ in rows])) if rows else np.nan,
                       'opp': float(np.median([o for _, _, o in rows])) if rows else np.nan, 'his_lower_in': int((dd < 0).sum()), 'p': p, 'per_fight': rows}
    for who in ('mcg', 'opp'):
        ps = [p for p in P if p['target'] == who and p['gap_still_cm'] is not None and p['gap_still_cm'] <= 0]
        mv = [p for p in ps if p['move'] in ('pull', 'slip')]; pl = [p for p in ps if p['move'] == 'pull']
        sv = [p for p in pl if p['gap_cm'] > 0]
        pooled[who].update({'would_land': len(ps), 'would_land_moved': len(mv), 'saved_by_move': float(np.mean([p['gap_cm'] > 0 for p in mv])) if mv else np.nan,
                            'saved_by_pull': float(np.mean([p['gap_cm'] > 0 for p in pl])) if pl else np.nan, 'n_pull_would_land': len(pl),
                            'saved_margin_median': float(np.median([p['gap_cm'] for p in sv])) if sv else np.nan,
                            'saved_needed_median': float(np.median([-p['gap_still_cm'] for p in sv])) if sv else np.nan,
                            'saved_travel_median': float(np.median([p['back_cm'] for p in sv])) if sv else np.nan,
                            'saved_within_5cm': float(np.mean([p['gap_cm'] <= 5 for p in sv])) if sv else np.nan,
                            'counter_after_pull': float(np.mean([p['countered'] for p in P if p['target'] == who and p['move'] == 'pull' and p['gap_cm'] > 0])),
                            'counter_after_slip': float(np.mean([p['countered'] for p in P if p['target'] == who and p['move'] == 'slip' and p['gap_cm'] > 0])),
                            'counter_after_still': float(np.mean([p['countered'] for p in P if p['target'] == who and p['move'] == 'still'])),
                            'still_landed_share': float(np.mean([p['gap_cm'] <= 0 for p in ps if p['move'] == 'still'])) if ps else np.nan})
    res['tests'], res['pooled'] = tests, pooled
    (HERE / 'results' / 'left' / 'evasion.json').write_text(json.dumps(res, indent=1, default=float).replace('NaN', 'null'))
    for k, v in pooled.items(): print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a != 'gaps_on_pull'})
    for k, v in tests.items(): print(k, 'him', round(v['mcg'], 2), 'opp', round(v['opp'], 2), 'lower in', v['his_lower_in'], '/', v['n'], 'p', v['p'])


if __name__ == '__main__':
    main()
