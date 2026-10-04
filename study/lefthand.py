"""The left hand, stage 1: candidate straight lefts from the 10-per-second standing frames.

A deliberately loose net (stage 2 decides): over every frame where both fighters are upright (close exchanges
included), McGregor's left wrist opens to >= 0.75 of his arm length, rising >= 0.15 within 0.4 s, travelling
toward the opponent by >= 0.25 torso lengths, at chest-to-head height. Stage 2 (lefthand_native.py) re-measures
each candidate at the broadcast's own frame rate, where the punch lives, with the strict definition.

    python lefthand.py   -> results/left/candidates.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import metrics as M     # noqa: E402
import frames as F      # noqa: E402

L_SH, L_EL, L_WR, R_SH, R_EL, R_WR = 5, 7, 9, 6, 8, 10


def arm_extension(k, sh, el, wr):
    """Shoulder-to-wrist distance over the arm's own length (upper arm + forearm), per frame."""
    S, E, Wr = M.P(k, sh), M.P(k, el), M.P(k, wr)
    length = np.linalg.norm(S - E, axis=-1) + np.linalg.norm(E - Wr, axis=-1)
    reach = np.linalg.norm(S - Wr, axis=-1)
    # the arm's length is steadier per fight than per frame (a bent arm foreshortens): use its 90th percentile
    L = np.nanpercentile(length, 90) if np.isfinite(length).any() else np.nan
    return reach / L, Wr


def candidates(fid: str, who='mcg', hand='left'):
    d = F.load(fid, keep='upright')
    t, shot = d['t'], d['shot']
    me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
    s = M.shot_scale(me, shot)
    sh, el, wr = (L_SH, L_EL, L_WR) if hand == 'left' else (R_SH, R_EL, R_WR)
    ext, W = arm_extension(me, sh, el, wr)
    fwd = np.sign(M.midp(other, 11, 12)[:, 0] - M.midp(me, 11, 12)[:, 0])
    shoulder_y = M.midp(me, 5, 6)[:, 1]; nose = M.P(me, 0)
    out = []
    for i in range(3, len(t)):
        if not (ext[i] >= .75): continue
        j0 = i - 3
        if shot[j0] != shot[i] or t[i] - t[j0] > .45: continue
        lo = np.nanmin(ext[j0:i])
        if not np.isfinite(lo) or ext[i] - lo < .15: continue
        k = j0 + int(np.nanargmin(ext[j0:i]))
        travel = (W[i, 0] - W[k, 0]) * fwd[i] / s[i]
        if not travel >= .25: continue
        height = (shoulder_y[i] - W[i, 1]) / s[i]                     # + above the shoulder line
        if not (-.45 <= height <= .9): continue
        if ext[i + 1] >= ext[i] if i + 1 < len(t) and shot[i + 1] == shot[i] else False:
            continue                                                  # keep the peak frame only
        if out and out[-1]['fight'] == fid and t[i] - out[-1]['t_peak'] < .6:
            continue                                                  # one punch, one event
        out.append({'fight': fid, 'who': who, 'hand': hand, 't_peak': float(t[i]), 't_start': float(t[k]),
                    'extension': float(ext[i]), 'travel': float(travel), 'height': float(height)})
    return out


if __name__ == '__main__':
    W = json.loads((HERE / 'windows.json').read_text())
    res = json.loads((HERE / 'results' / 'results.json').read_text())
    allc = []
    for f in res['fights']:
        fid = f['fight']
        if not (HERE / 'results' / 'perframe' / f'{fid}.npz').exists(): continue
        c = candidates(fid)
        # the opponents' rear straight as the baseline: orthodox rear = right hand, southpaw rear = left
        rear = 'right' if f['opp'].get('orthodox_share', 1) >= .5 else 'left'
        o = candidates(fid, who='opp', hand=rear)
        allc += c + o
        print(f"{fid:16s} McGregor left candidates {len(c):4d}   opponent rear-hand candidates {len(o):4d}")
    (HERE / 'results' / 'left').mkdir(parents=True, exist_ok=True)
    (HERE / 'results' / 'left' / 'candidates.json').write_text(json.dumps(allc, indent=1))
    print(len(allc), 'candidates')
