"""From one fight's pose file to McGregor and his opponent, frame by frame, in the live fight only.

Steps, in order (each drops frames it can't vouch for; none guesses):
  1. live      the fight clock is on screen (board.py / live.py), inside the fight's span (windows.json:
               needed only where one video holds several fights)
  2. shots     split at camera cuts (colour histogram jump between kept frames)
  3. wide      two fighter-sized detections (box >= 25 % of the frame height, head not cut by the top edge).
               Points within 1.5 % of the frame border count as unseen, so a foot cut off by the bottom
               edge never enters a stance measurement, while the same fighter's guard and range still do
  4. identity  shorts colour against the fight's two prototypes (windows.json), then held continuous inside a
               shot: a body keeps the identity of the fighter it continues from (the game's colour-swap fix)
  5. standing  both hips clearly above both knees, and a gap between the fighters (no clinch, no ground)

The result is a table: one row per kept frame with both fighters' 133 points and a shot number.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get('MMA_WORK', 'work') + '/mcgregor')
HERE = Path(__file__).parent
CUT = .35            # histogram L1 distance between kept frames that marks a camera cut
CONF = .4            # keypoint score below which a point counts as unseen

L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16


def windows() -> dict:
    return json.loads((HERE / 'windows.json').read_text())


def mid(k, a, b):
    return (k[..., a, :2] + k[..., b, :2]) / 2


def seen(k, *idx):
    return np.all(k[..., list(idx), 2] >= CONF, axis=-1)


def torso(k):
    return np.linalg.norm(mid(k, L_SH, R_SH) - mid(k, L_HIP, R_HIP), axis=-1)


def load(fight: str, keep: str = 'standing', suffix: str = '') -> dict:
    """Kept frames of one fight (a fight is a pose file plus a live window list), as arrays."""
    import sys; sys.path.insert(0, str(HERE))
    from live import live as clock_live
    w = windows()[fight]
    vid = w.get('video', fight)
    d = np.load(DATA / 'pose' / f'{vid}{suffix}.npz')
    t, boxes, kps, shorts, hist = d['t'], d['boxes'], d['kps'].astype(np.float32), d['shorts'], d['hist']
    H = float(d['height'])
    if w.get('no_clock'):                    # a feed without a clock: the span set by eye is the live window
        live = np.ones(len(t), bool)
    else:
        bt, bon, _ = clock_live(vid)
        live = np.interp(t, bt, bon.astype(float)) > .5
    if 'span' in w:
        live &= (t >= w['span'][0]) & (t <= w['span'][1])
    # shots: a cut wherever the frame histogram jumps (or the clock jumps: a gap in kept frames)
    jump = np.r_[True, (np.abs(np.diff(hist, axis=0)).sum(1) > CUT) | (np.diff(t) > .25)]
    # the real sampling rate from the timestamps: a 25 fps upload sampled at 'every frame' is 25, not 30
    rate = float(min(float(d['rate']) if 'rate' in d.files else 10.0, 1 / np.median(np.diff(t)))) if len(t) > 1 else 10.0
    shot = np.cumsum(jump)
    # wide: the two most confident detections are both fighter-sized
    hgt = (boxes[:, :, 3] - boxes[:, :, 1]) / H
    fighter = (hgt > .25) & (boxes[:, :, 1] > .01 * H)
    Wd = float(d['width'])
    edge = (kps[..., 0] < .015 * Wd) | (kps[..., 0] > .985 * Wd) | (kps[..., 1] < .015 * H) | (kps[..., 1] > .985 * H)
    kps[..., 2] = np.where(edge, 0, kps[..., 2])
    # identity by shorts colour (Lab), against the fight's prototypes
    # each fighter may have several prototypes (the same shorts under different light); nearest one counts
    def near(protos):
        P_ = np.array(protos, np.float32).reshape(-1, 3)
        return np.linalg.norm(shorts[:, :, None, :] - P_[None, None], axis=-1).min(-1)
    dist = np.stack([near(w['shorts']['mcgregor']), near(w['shorts']['opponent'])], axis=-1)   # frame, person, which
    rows = []
    last = {}
    for i in np.flatnonzero(live):
        cand = [j for j in range(boxes.shape[1]) if fighter[i, j]]
        if len(cand) < 2:
            continue
        # the pair whose shorts best explain McGregor + opponent (drops a referee the detector let through)
        best = None
        for a in cand:
            for b in cand:
                if a == b or np.isnan(dist[i, a, 0]) or np.isnan(dist[i, b, 1]):
                    continue
                cost = dist[i, a, 0] + dist[i, b, 1]
                margin = (dist[i, a, 1] + dist[i, b, 0]) - cost        # how much better than the swapped labels
                if best is None or cost < best[0]:
                    best = (cost, margin, a, b)
        if best is None:
            continue
        _, margin, m, o = best
        km, ko = kps[i, m], kps[i, o]
        # continuity inside a shot: if the swapped labels continue the previous frame far better, swap
        prev = last.get(shot[i])
        if prev is not None:
            pm, po = prev
            keep = np.nanmean(np.linalg.norm(km[:17, :2] - pm[:17, :2], axis=-1)) + np.nanmean(np.linalg.norm(ko[:17, :2] - po[:17, :2], axis=-1))
            swap = np.nanmean(np.linalg.norm(km[:17, :2] - po[:17, :2], axis=-1)) + np.nanmean(np.linalg.norm(ko[:17, :2] - pm[:17, :2], axis=-1))
            if swap < .6 * keep and margin < w.get('margin', 25):
                km, ko = ko, km
        elif margin < w.get('margin', 25):
            continue                       # first frame of a shot must be clearly labelled by colour
        last[shot[i]] = (km, ko)
        rows.append((i, km, ko))
    if not rows:
        return {'fight': fight, 'n': 0}
    idx = np.array([r[0] for r in rows]); M = np.stack([r[1] for r in rows]); O = np.stack([r[2] for r in rows])
    # standing: hips above knees for both (image y grows downward), and not tangled
    def upright(k):
        ok = seen(k, L_HIP, R_HIP, L_KNEE, R_KNEE, L_SH, R_SH)
        hip_y = mid(k, L_HIP, R_HIP)[:, 1]; knee_y = mid(k, L_KNEE, R_KNEE)[:, 1]
        sh_y = mid(k, L_SH, R_SH)[:, 1]
        vertical = (hip_y - sh_y) > .7 * torso(k)          # torso within ~45 degrees of vertical (not grappling)
        return ok & (knee_y - hip_y > .45 * torso(k)) & vertical
    gap = np.linalg.norm(mid(M, L_HIP, R_HIP) - mid(O, L_HIP, R_HIP), axis=-1) / ((torso(M) + torso(O)) / 2)
    standing = upright(M) & upright(O) & (gap > 1.1)
    if keep == 'upright':                        # exchanges at close range too (for strikes): upright, any gap
        standing = upright(M) & upright(O)
    elif keep == 'any':                          # every identified pair, standing or not (falls, knockdowns)
        standing = np.ones(len(M), bool)
    return {'fight': fight, 'rate': rate, 'n': int(standing.sum()), 't': t[idx][standing], 'shot': shot[idx][standing],
            'mcg': M[standing], 'opp': O[standing], 'live_seconds': float(live.sum() / rate),
            'wide_pairs': len(rows)}
