"""The pre-registered quantities, per frame, for any fighter given the other (the opponent sets "forward").

Every length is divided by the fighter's torso length in that shot (median over the shot), so zoom cancels.
A quantity is NaN where the points it needs aren't seen (score < CONF); nothing is filled in.

Index map (COCO-WholeBody): 0 nose, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees,
15/16 ankles, 17/18 left big/small toe, 19 left heel, 20/21 right big/small toe, 22 right heel.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

CONF = .4


def P(k, i):
    """Point i as (x, y), NaN where unseen."""
    p = k[..., i, :2].astype(np.float64).copy()
    p[k[..., i, 2] < CONF] = np.nan
    return p


def midp(k, a, b):
    return (P(k, a) + P(k, b)) / 2


def shot_scale(k, shot):
    """Torso length per frame, replaced by its median over the shot (steadier than one frame)."""
    tor = np.linalg.norm(midp(k, 5, 6) - midp(k, 11, 12), axis=-1)
    out = np.full(len(k), np.nan)
    for s in np.unique(shot):
        m = shot == s
        if np.isfinite(tor[m]).sum() >= 3:
            out[m] = np.nanmedian(tor[m])
    return out


def per_frame(me, other, shot, t):
    """Dict of per-frame quantities for fighter `me` facing `other`."""
    s = shot_scale(me, shot)
    hip, ohip = midp(me, 11, 12), midp(other, 11, 12)
    fwd = np.sign(ohip[:, 0] - hip[:, 0])                         # +1: opponent to the right in the image
    la, ra = P(me, 15), P(me, 16)
    q = {}
    # H1 stance width: ankle to ankle
    q['stance_width'] = np.linalg.norm(la - ra, axis=-1) / s
    # stance length: how far the lead ankle is in front of the rear one, toward the opponent
    q['stance_length'] = np.abs(la[:, 0] - ra[:, 0]) / s
    # H6 stance side: lead foot is the ankle nearer the opponent; left lead = orthodox
    lead_left = (la[:, 0] - ra[:, 0]) * fwd > 0
    side = np.where(np.isfinite(la[:, 0]) & np.isfinite(ra[:, 0]) & (np.abs(la[:, 0] - ra[:, 0]) > .15 * s),
                    np.where(lead_left, 1.0, -1.0), np.nan)       # +1 orthodox, -1 southpaw
    q['orthodox'] = side
    # H2 on the toes: heel lift, how far the heel sits above the big toe in the image, per foot length. A camera
    # above the cage already lifts every heel a little in the image, so this is read against the opponent in
    # the same frames (the pre-registered McGregor - opponent difference), where the camera cancels.
    def heel_up(heel, toe):
        h, tt = P(me, heel), P(me, toe)
        foot = np.linalg.norm(h - tt, axis=-1)
        ok = np.isfinite(foot) & (foot > .12 * s)
        return np.where(ok, (tt[:, 1] - h[:, 1]) / foot, np.nan)
    lh, rh = heel_up(19, 17), heel_up(22, 20)
    rear_heel = np.where(side == 1, rh, np.where(side == -1, lh, np.nan))   # orthodox: right foot is rear
    q['rear_heel_up'] = rear_heel
    q['any_heel_up'] = np.nanmean(np.c_[lh, rh], axis=1)              # both feet, mean lift
    # H4 guard: wrist height above the shoulder line (positive = above), each hand and the mean
    sh_y = midp(me, 5, 6)[:, 1]
    lw, rw = P(me, 9), P(me, 10)
    q['guard_left'] = (sh_y - lw[:, 1]) / s
    q['guard_right'] = (sh_y - rw[:, 1]) / s
    q['guard'] = np.nanmean(np.c_[q['guard_left'], q['guard_right']], axis=1)
    lead_w = np.where((side == 1)[:, None], lw, rw); rear_w = np.where((side == 1)[:, None], rw, lw)
    q['guard_lead'] = (sh_y - lead_w[:, 1]) / s
    q['guard_rear'] = (sh_y - rear_w[:, 1]) / s
    # lead hand reach: how far the lead wrist sits out in front of the shoulders, toward the opponent
    q['lead_reach'] = (lead_w[:, 0] - midp(me, 5, 6)[:, 0]) * fwd / s
    # posture: hip height over the ankles (crouch), and forward lean of the torso toward the opponent
    q['hip_height'] = (np.nanmean(np.c_[la[:, 1], ra[:, 1]], axis=1) - hip[:, 1]) / s
    sh = midp(me, 5, 6)
    q['lean'] = np.degrees(np.arctan2((sh[:, 0] - hip[:, 0]) * fwd, hip[:, 1] - sh[:, 1]))
    q['head_forward'] = (P(me, 0)[:, 0] - hip[:, 0]) * fwd / s
    # H7 range: hip to hip, in mean torso lengths of the two
    os_ = shot_scale(other, shot)
    q['range'] = np.linalg.norm(hip - ohip, axis=-1) / ((s + os_) / 2)
    # kick: the kicking ankle above the standing leg's knee AND the leg extended away from the hips
    # (>= 0.75 of its own length), held >= 2 frames; one-frame blips, steps and knee checks don't count
    lk, rk = P(me, 13), P(me, 14)
    lh_, rh_ = P(me, 11), P(me, 12)
    def leg_len(h, k, a):
        return np.linalg.norm(h - k, axis=-1) + np.linalg.norm(k - a, axis=-1)
    def up(ank, hip_, knee_, other_knee):
        L = np.nanmedian(np.r_[leg_len(lh_, lk, la), leg_len(rh_, rk, ra)]) if np.isfinite(la).any() else np.nan
        return (ank[:, 1] < other_knee[:, 1]) & (np.linalg.norm(ank - hip_, axis=-1) > .75 * L)
    raw = (up(la, lh_, lk, rk) | up(ra, rh_, rk, lk)).astype(float)
    held = raw * np.r_[raw[1:], 0] + raw * np.r_[0, raw[:-1]]       # on in this frame and a neighbour
    q['kick_pose'] = (held > 0).astype(float)
    q['hip_y'] = hip[:, 1] / s                                   # for bounce, per continuous run
    q['hip_x'] = hip[:, 0] / s
    q['scale'] = s
    return q


def runs(t, shot, max_gap=.15):
    """Continuous stretches: same shot, no missing kept frame between."""
    br = np.r_[True, (np.diff(t) > max_gap) | (np.diff(shot) != 0)]
    return np.cumsum(br)


def bounce(t, hip_y, shot, rate=10, band=(1.5, 4.0), min_len=2.0):
    """Hip vertical motion in the bounce band, as RMS (torso lengths), per continuous run of >= min_len s.
    Returns (rms values, seconds) per run."""
    out = []
    r = runs(t, shot)
    for k in np.unique(r):
        m = r == k
        y = hip_y[m]
        if m.sum() < min_len * rate or np.isnan(y).mean() > .1:
            continue
        y = np.interp(np.arange(len(y)), np.flatnonzero(~np.isnan(y)), y[~np.isnan(y)])
        y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 2), np.arange(len(y)))   # camera drift out
        f, pxx = welch(y, fs=rate, nperseg=min(len(y), 20))
        sel = (f >= band[0]) & (f <= band[1])
        out.append((np.sqrt(np.trapezoid(pxx[sel], f[sel])), m.sum() / rate))
    return out


def events(flag, t, shot, min_gap=.6):
    """Count separate events in a 0/1 per-frame flag (one event per contiguous run, runs closer than min_gap
    in the same shot merged)."""
    f = np.nan_to_num(flag) > .5
    n, last_t, last_s, active = 0, -1e9, None, False
    for on, tt, s in zip(f, t, shot):
        if on and not active:
            if not (s == last_s and tt - last_t < min_gap):
                n += 1
        if on:
            last_t, last_s = tt, s
        active = on
    return n


def switches(side, t, shot, hold=1.0):
    """Stance switches: changes of the lead foot that then hold for >= `hold` seconds (filters flicker)."""
    n = 0; cur = None; cand = None; since = None; last_shot = None
    for v, tt, s in zip(side, t, shot):
        if s != last_shot:
            cur, cand, since, last_shot = None, None, None, s
        if np.isnan(v):
            continue
        if cur is None:
            cur = v; continue
        if v != cur:
            if cand != v:
                cand, since = v, tt
            elif tt - since >= hold:
                n += 1; cur = v; cand = None
        else:
            cand = None
    return n


def footwork(t, hip_x, fwd_sign, shot, rate=10):
    """Per frame, horizontal hip velocity toward the opponent (torso lengths per second), inside continuous runs;
    NaN at run edges. Positive = advancing, negative = retreating."""
    v = np.full(len(t), np.nan)
    r = runs(t, shot)
    for k in np.unique(r):
        m = np.flatnonzero(r == k)
        if len(m) < 5:
            continue
        x = hip_x[m]
        if np.isnan(x).mean() > .2:
            continue
        x = np.interp(np.arange(len(x)), np.flatnonzero(~np.isnan(x)), x[~np.isnan(x)])
        x = np.convolve(x, np.ones(3) / 3, mode='same')
        dv = np.gradient(x) * rate
        v[m[1:-1]] = (dv * fwd_sign[m])[1:-1]
    return v


def bounce_hz(t, hip_y, shot, rate=10, band=(1.0, 4.5), min_len=2.0):
    """Dominant frequency of hip vertical motion per run (Hz), with run lengths."""
    out = []
    r = runs(t, shot)
    for k in np.unique(r):
        m = r == k
        y = hip_y[m]
        if m.sum() < min_len * rate or np.isnan(y).mean() > .1:
            continue
        y = np.interp(np.arange(len(y)), np.flatnonzero(~np.isnan(y)), y[~np.isnan(y)])
        y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 2), np.arange(len(y)))
        f, pxx = welch(y, fs=rate, nperseg=min(len(y), 20))
        sel = (f >= band[0]) & (f <= band[1])
        if sel.any():
            out.append((f[sel][np.argmax(pxx[sel])], m.sum() / rate))
    return out
