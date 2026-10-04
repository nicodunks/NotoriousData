"""What the lead hand did before the rear straight, and the counters in detail.

Lead hand, per punch (the 1 s before onset), McGregor's left vs opponents' rear straights:
  lead_ext_onset  lead-arm extension at the moment the rear hand starts (0 folded, 1 straight)
  lead_motion     lead-wrist path length relative to the lead shoulder in the second before, torso lengths
  lead_jabbed     a fast lead-hand extension (>= 0.2 of its length, >= 5 torso lengths/s) in that second
  posted          lead arm extended (>= 0.6) and still (motion < 0.6) when the rear hand fires: a hand held out
                  as a range-finder, not thrown
Also jabs per upright minute (fast lead-hand extensions, loose rule), McGregor vs opponents.

Counters, each set-up apart: head and torso movement (from slips.json), fist speed, range, how fast he fired.

    python leadhand.py  -> results/left/leadhand.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402
import strikes as S     # noqa: E402

RNG = np.random.default_rng(4)
evs = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['kind'] == 'straight']
slips = {(r['fight'], r['who'], round(r['t_onset'], 3)): r for r in json.loads((HERE / 'results' / 'left' / 'slips.json').read_text())['events']}
mins = json.loads((HERE / 'results' / 'left' / 'minutes.json').read_text())
cache, recs, jabs = {}, [], {}
for f in sorted({e['fight'] for e in evs}):
    d = F.load(f, keep='upright', suffix='_30'); cache[f] = d
    rate = d['rate']
    for who in ('mcg', 'opp'):
        me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
        q = M.per_frame(me, other, d['shot'], d['t'])
        south = np.nanmean(q['orthodox'] == -1) > .5 if np.isfinite(q['orthodox']).any() else who == 'mcg'
        lead = 'right' if south else 'left'
        jabs[(f, who)] = len(S.extensions(me, other, d['t'], d['shot'], lead, rate, strict=False))
for e in evs:
    d = cache[e['fight']]; t, shot, rate = d['t'], d['shot'], d['rate']
    me, other = (d['mcg'], d['opp']) if e['who'] == 'mcg' else (d['opp'], d['mcg'])
    lead = 'right' if e['hand'] == 'left' else 'left'
    ext, Sh, El, Wr = S.arm_series(me, lead)
    sc = M.shot_scale(me, shot)
    on = int(np.argmin(np.abs(t - e['t_onset']))); st = int(np.argmin(np.abs(t - (e['t_onset'] - 1))))
    if shot[st] != shot[on]: st = int(np.flatnonzero(shot == shot[on])[0])
    seg = (Wr[st:on + 1] - Sh[st:on + 1]) / sc[on]                     # the hand relative to its own shoulder: body and camera drop out
    motion = float(np.nansum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    jl = [x for x in S.extensions(me[st:on + 1], other[st:on + 1], t[st:on + 1], shot[st:on + 1], lead, rate, strict=False)]
    r = {'fight': e['fight'], 'date': e['date'], 'who': e['who'], 'setup': e['setup'], 'speed': e['peak_speed'],
         'range': e['range_at_onset'], 'snap': e.get('opp_head_snap'),
         'lead_ext_onset': float(ext[on]) if np.isfinite(ext[on]) else None, 'lead_motion': motion, 'lead_jabbed': bool(jl)}
    r['posted'] = bool(r['lead_ext_onset'] is not None and r['lead_ext_onset'] >= .6 and motion < .6)
    sl = slips.get((e['fight'], e['who'], round(e['t_onset'], 3)))
    if sl: r.update({k: sl[k] for k in ('head_move', 'head_back', 'head_down', 'lean_swing', 'lead_ms')})
    recs.append(r)


def boot(vals, stat=np.median, fights=None, reps=3000):
    vals = [v for v in vals if v[1] is not None and np.isfinite(v[1])]
    if len(vals) < 3: return None
    fl = sorted({f for f, _ in vals})
    b = [stat([v for g in RNG.choice(fl, len(fl)) for f, v in vals if f == g]) for _ in range(reps)]
    return {'value': float(stat([v for _, v in vals])), 'ci': np.nanpercentile(b, [2.5, 97.5]).tolist(), 'n': len(vals)}


out = {'lead': {}, 'jab_rate': {}, 'setups': {}, 'records': recs}
for k, stat in (('lead_ext_onset', np.median), ('lead_motion', np.median), ('lead_jabbed', np.mean), ('posted', np.mean)):
    out['lead'][k] = {w: boot([(r['fight'], float(r[k]) if r[k] is not None else None) for r in recs if r['who'] == w], stat) for w in ('mcg', 'opp')}
for w in ('mcg', 'opp'):
    rates = [(f, jabs[(f, w)] / mins[f]) for f in mins if (f, w) in jabs and mins[f] >= 1]
    out['jab_rate'][w] = {'value': float(np.median([v for _, v in rates])), 'per_fight': rates}
mc = [r for r in recs if r['who'] == 'mcg']
for s in ['pull counter', 'slip counter', 'counter, head still', 'caught coming in', 'behind the paw', 'straight lead']:
    rs = [r for r in mc if r['setup'] == s]
    row = {'n': len(rs), 'fights': len({r['fight'] for r in rs})}
    for k in ('head_move', 'head_back', 'head_down', 'lean_swing', 'lead_ms', 'speed', 'range', 'snap', 'lead_ext_onset', 'lead_motion'):
        v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
        row[k] = float(np.median(v)) if v else None
    row['posted'] = float(np.mean([r['posted'] for r in rs])) if rs else None
    row['examples'] = [(r['fight'], r['date']) for r in rs]
    out['setups'][s] = row
(HERE / 'results' / 'left' / 'leadhand.json').write_text(json.dumps(out, indent=1, default=float))
print('LEAD', json.dumps({k: {w: (round(v['value'], 2), [round(c, 2) for c in v['ci']]) if v else None for w, v in x.items()} for k, x in out['lead'].items()}))
print('JAB RATE', {w: round(v['value'], 2) for w, v in out['jab_rate'].items()})
for s, r in out['setups'].items():
    print(f"{s:20s} n{r['n']:3d} f{r['fights']:2d} head {r['head_move']} back {r['head_back']} down {r['head_down']} swing {r['lean_swing']} fire {r['lead_ms']} spd {r['speed']} rng {r['range']} posted {r['posted']}")
