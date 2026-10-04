"""The average stance per fight: every standing frame of a fighter in one frame of reference, then the median.

Frame of reference: hips at the origin, torso length 1, the opponent always to the right, and southpaw frames
mirrored to orthodox (so lead and rear limbs line up across frames). Only seen points enter the median.
"""
from __future__ import annotations

import numpy as np

import metrics as M

BODY = list(range(23))      # body + feet; hands and face are too small at broadcast distance to average well
# left/right pairs among the first 23 COCO-WholeBody points (mirroring swaps them)
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)]


def canonical(k: np.ndarray, other: np.ndarray, shot: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(frames, 23, 2) points in the shared frame of reference; NaN where unseen."""
    q = M.per_frame(k, other, shot, t)
    s = q['scale']
    hip = M.midp(k, 11, 12)
    fwd = np.sign(M.midp(other, 11, 12)[:, 0] - hip[:, 0])
    pts = np.stack([M.P(k, i) for i in BODY], axis=1)               # frames, 23, 2
    pts = (pts - hip[:, None, :]) / s[:, None, None]
    pts[..., 0] *= fwd[:, None]                                        # opponent to the right
    south = q['orthodox'] == -1
    m = pts[south].copy()
    m[..., 0] *= -1                                                    # mirror southpaw to orthodox ...
    for a, b in PAIRS:                                                 # ... and swap left/right labels
        m[:, [a, b]] = m[:, [b, a]]
    pts[south] = m
    keep = np.isfinite(q['orthodox'])
    return pts[keep]


def average(pts: np.ndarray, min_seen: int = 30) -> np.ndarray:
    """Median position of each point over frames where it was seen (NaN if seen fewer than min_seen times)."""
    out = np.nanmedian(pts, axis=0)
    out[np.isfinite(pts[..., 0]).sum(0) < min_seen] = np.nan
    return out


def canonical_all(k, other, shot, t):
    """One row per input frame: hips at the origin, torso 1, opponent to the right, no mirroring (so a
    southpaw's left hand stays his left hand)."""
    q = M.per_frame(k, other, shot, t)
    s = q['scale']; hip = M.midp(k, 11, 12)
    fwd = np.sign(M.midp(other, 11, 12)[:, 0] - hip[:, 0])
    pts = (np.stack([M.P(k, i) for i in BODY], axis=1) - hip[:, None, :]) / s[:, None, None]
    pts[..., 0] *= fwd[:, None]
    return pts
