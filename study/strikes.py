"""The left hand, at the broadcast's own frame rate: every McGregor straight left, measured and classified, and
every opponent rear straight measured the same way as the baseline.

A committed punch (calibrated by eye on 24 labelled Khabib candidates, then checked on other fights): the arm
opens by >= 0.2 of its length within 400 ms to >= 0.65, the wrist reaches >= 8 torso lengths per second and
travels >= 0.3 torso lengths toward the opponent. Elbow angle proved unreliable (foreshortening), so the kind
comes from the wrist's path: straight if its displacement is >= 75 % of the distance it travelled, else hook.

    python strikes.py            -> results/left/events.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402

SH = {'left': 5, 'right': 6}; EL = {'left': 7, 'right': 8}; WR = {'left': 9, 'right': 10}
ANK = {'left': 15, 'right': 16}


def smooth(x, n=3):
    """Centered moving average that ignores NaN (keeps NaN where the window has nothing)."""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        w = x[max(0, i - n // 2): i + n // 2 + 1]
        w = w[np.isfinite(w)] if w.ndim == 1 else w[np.isfinite(w).all(-1)]
        if len(w): out[i] = w.mean(0) if w.ndim == 1 else np.nan
    return out


def track(k, i):
    p = M.P(k, i).copy()
    for d in (0, 1):
        p[:, d] = smooth(p[:, d])
    return p


def angle(a, b, c):
    """Angle at b, degrees."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def arm_series(me, hand):
    S, E, W = track(me, SH[hand]), track(me, EL[hand]), track(me, WR[hand])
    length = np.linalg.norm(S - E, axis=-1) + np.linalg.norm(E - W, axis=-1)
    L = np.nanpercentile(length, 90) if np.isfinite(length).any() else np.nan
    return np.linalg.norm(S - W, axis=-1) / L, S, E, W


def extensions(me, other, t, shot, hand, rate, strict=True):
    """Fast extensions of one arm: (onset index, peak index, kind) with kind 'straight' or 'hook'."""
    ext, S, E, W = arm_series(me, hand)
    s = M.shot_scale(me, shot)
    fwd = np.sign(M.midp(other, 11, 12)[:, 0] - M.midp(me, 11, 12)[:, 0])
    speed = np.linalg.norm(np.gradient(W, axis=0), axis=-1) * rate / s
    win = int(round(.4 * rate))
    out = []
    for i in range(win, len(t) - 1):
        if not (ext[i] >= (.65 if strict else .6)) or not (ext[i] >= ext[i - 1]) or not (ext[i] >= ext[i + 1]):
            continue                                               # a local peak of extension
        j0 = i - win
        if shot[j0] != shot[i] or t[i] - t[j0] > .4 + 1.5 / rate:
            continue
        seg = ext[j0:i]
        if not np.isfinite(seg).any() or ext[i] - np.nanmin(seg) < (.2 if strict else .15):
            continue
        k0 = j0 + int(np.nanargmin(seg))
        vmax = np.nanmax(speed[k0:i + 1]) if np.isfinite(speed[k0:i + 1]).any() else np.nan
        travel = (W[i, 0] - W[k0, 0]) * fwd[i] / s[i]
        if not (vmax >= (8 if strict else 5)) or not (travel >= (.3 if strict else .2)):
            continue
        path = np.nansum(np.linalg.norm(np.diff(W[k0:i + 1], axis=0), axis=-1))
        straightness = np.linalg.norm(W[i] - W[k0]) / path if path > 0 else np.nan
        kind = 'straight' if straightness >= .75 else 'hook'
        # onset: first frame of the rise where the wrist is past 30 % of its peak speed
        on = k0
        for j in range(k0, i + 1):
            if speed[j] >= .3 * vmax: on = j; break
        if out and t[i] - t[out[-1][1]] < .35:
            continue
        out.append((on, i, kind))
    return out


def describe(d, who, hand, on, pk, rate, opp_attacks):
    """Everything we measure about one punch (lengths in the puncher's torso lengths, times in ms)."""
    t, shot = d['t'], d['shot']
    me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
    s = M.shot_scale(me, shot)[pk]
    hip, ohip = track(me, 11) / 2 + track(me, 12) / 2, track(other, 11) / 2 + track(other, 12) / 2
    fwd = np.sign(ohip[pk, 0] - hip[pk, 0])
    face = np.nanmean(np.stack([track(me, i) for i in range(5)]), 0)
    oface = np.nanmean(np.stack([track(other, i) for i in range(5)]), 0)
    ext, S, E, W = arm_series(me, hand)
    lead = 'right' if hand == 'left' else 'left'
    lext, LS, LE, LW = arm_series(me, lead)
    speed = np.linalg.norm(np.gradient(W, axis=0), axis=-1) * rate / M.shot_scale(me, shot)
    n = lambda sec: int(round(sec * rate))
    pre = max(0, on - n(.6)); pre_shot_ok = shot[pre] == shot[on]
    f = lambda v: float(v) if v is not None and np.isfinite(v) else None
    ev = {
        'fight': d['fight'], 'who': who, 'hand': hand, 't_onset': float(t[on]), 't_peak': float(t[pk]),
        'duration_ms': f((t[pk] - t[on]) * 1000),
        'peak_speed': f(np.nanmax(speed[on:pk + 1])),
        'extension': f(ext[pk]),
        'reach': f((W[pk, 0] - hip[pk, 0]) * fwd / s),                       # wrist in front of his hips
        'elbow_deg': f(angle(S[pk], E[pk], W[pk])),
        'shoulder_drive': f((S[pk, 0] - S[on, 0]) * fwd / s),                  # punching shoulder forward
        'hip_drive': f((hip[pk, 0] - hip[on, 0]) * fwd / s),                   # weight forward
        'lead_step': f((track(me, ANK[lead])[pk, 0] - track(me, ANK[lead])[on, 0]) * fwd / s),
        'rear_drag': f((track(me, ANK[hand])[pk, 0] - track(me, ANK[hand])[on, 0]) * fwd / s),
        'head_fwd_at_peak': f((face[pk, 0] - hip[pk, 0]) * fwd / s),          # head over/behind his hips
        'head_drop_at_peak': f((hip[pk, 1] - face[pk, 1]) / s),               # head height above hips
        'head_back_before': f(-(face[on, 0] - face[pre, 0]) * fwd / s) if pre_shot_ok else None,   # + = pulled away
        'head_down_before': f((face[on, 1] - face[pre, 1]) / s) if pre_shot_ok else None,          # + = dropped
        'range_at_onset': f(np.linalg.norm(hip[on] - ohip[on]) / s),
        'opp_advance_before': f(((ohip[on, 0] - ohip[pre, 0]) * -fwd) / s / .6) if pre_shot_ok else None,  # torso/s toward him
        'lead_reach_before': f(np.nanmax(lext[max(0, on - n(1.0)):on + 1])) if on > 0 else None,
        'lead_high_at_peak': f((LS[pk, 1] - LW[pk, 1]) / s),                  # lead hand height vs its shoulder
        'opp_head_snap': None,
    }
    # body angles at onset and at full extension (2-D, from the broadcast view)
    HIP_ = {'left': 11, 'right': 12}; KNEE_ = {'left': 13, 'right': 14}
    def knee(side, i):
        return angle(track(me, HIP_[side])[i], track(me, KNEE_[side])[i], track(me, ANK[side])[i])
    sh_mid = (track(me, 5) + track(me, 6)) / 2
    def lean(i):   # torso angle from vertical, toward the opponent (+)
        v = sh_mid[i] - hip[i]
        return np.degrees(np.arctan2(v[0] * fwd, -v[1]))
    shw = np.linalg.norm(track(me, 5) - track(me, 6), axis=-1) / M.shot_scale(me, shot)
    hipw = np.linalg.norm(track(me, 11) - track(me, 12), axis=-1) / M.shot_scale(me, shot)
    ev.update({
        'lead_knee_onset': f(knee(lead, on)), 'lead_knee_peak': f(knee(lead, pk)),
        'rear_knee_onset': f(knee(hand, on)), 'rear_knee_peak': f(knee(hand, pk)),
        'lean_onset': f(lean(on)), 'lean_peak': f(lean(pk)), 'lean_change': f(lean(pk) - lean(on)),
        'shoulder_turn': f(shw[pk] - shw[on]),          # apparent shoulder width change: rotation proxy
        'hip_turn': f(hipw[pk] - hipw[on]),
        'level_change': f((hip[pk, 1] - hip[on, 1]) / s),   # + = hips dropped during the punch
        'stance_width_onset': f(np.linalg.norm(track(me, 15)[on] - track(me, 16)[on]) / s),
    })
    # opponent's face moving away from the punch in the 250 ms after the peak, relative to his own hips (a
    # visible effect, not proof of contact)
    after = min(len(t) - 1, pk + n(.25))
    if shot[after] == shot[pk]:
        ev['opp_head_snap'] = f(((oface[after, 0] - ohip[after, 0]) - (oface[pk, 0] - ohip[pk, 0])) * fwd / s)
    # what the opponent was doing: an attack of his that peaked from 700 ms before onset to 100 ms after
    ev['opp_attack_before'] = any(t[on] - .7 <= ta <= t[on] + .1 for ta in opp_attacks)
    return ev


def setup(ev):
    """The set-up, by transparent rules (in this order)."""
    back, down, adv = ev['head_back_before'], ev['head_down_before'], ev['opp_advance_before']
    if ev['opp_attack_before']:
        if back is not None and back >= .15: return 'pull counter'
        if down is not None and down >= .12: return 'slip counter'
        return 'counter, head still'
    if adv is not None and adv >= .35: return 'caught coming in'
    if ev['lead_reach_before'] is not None and ev['lead_reach_before'] >= .8: return 'behind the paw'
    return 'straight lead'


def main():
    W = json.loads((HERE / 'windows.json').read_text())
    res = json.loads((HERE / 'results' / 'results.json').read_text())
    events = []; minutes_by_fight = {}
    for f in res['fights']:
        fid = f['fight']; vid = W[fid].get('video', fid)
        if not (F.DATA / 'pose' / f'{vid}_30.npz').exists():
            print('no every-frame pose yet:', fid); continue
        d = F.load(fid, keep='upright', suffix='_30')
        if not d['n']:
            continue
        rate = d['rate']
        orear = 'right' if f['opp'].get('orthodox_share', 1) >= .5 else 'left'
        # every attack of each fighter (either hand, loose definition) for the "what was he doing" context
        att = {}
        for who in ('mcg', 'opp'):
            me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
            att[who] = [d['t'][pk] for hand in ('left', 'right') for _, pk, _ in extensions(me, other, d['t'], d['shot'], hand, rate, strict=False)]
        n_m = n_o = 0
        for who, hand in (('mcg', 'left'), ('opp', orear)):
            me, other = (d['mcg'], d['opp']) if who == 'mcg' else (d['opp'], d['mcg'])
            for on, pk, kind in extensions(me, other, d['t'], d['shot'], hand, rate):
                ev = describe(d, who, hand, on, pk, rate, att['opp' if who == 'mcg' else 'mcg'])
                ev['kind'] = kind; ev['date'] = f['date']; ev['opponent'] = f['opponent']
                ev['setup'] = setup(ev) if kind == 'straight' else None
                events.append(ev)
                n_m += who == 'mcg'; n_o += who == 'opp'
        minutes = d['n'] / rate / 60
        minutes_by_fight[fid] = minutes
        print(f"{fid:16s} upright {minutes:5.1f} min  McGregor lefts {n_m:4d}  opponent rear {n_o:4d}", flush=True)
    (HERE / 'results' / 'left').mkdir(parents=True, exist_ok=True)
    (HERE / 'results' / 'left' / 'events.json').write_text(json.dumps(events, indent=1))
    (HERE / 'results' / 'left' / 'minutes.json').write_text(json.dumps(minutes_by_fight, indent=1))
    print(len(events), 'events')


if __name__ == '__main__':
    main()
