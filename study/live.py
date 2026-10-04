"""Live play from the clock score (board.py): smoothed over a second, live where at least 55 % of the clock's
own full-strength score (its 95th percentile in this video) is present. Anchoring to the clock's own strength
works whether the video is mostly fight or mostly intro, where a two-class split (Otsu) does not. One rule, used by pose.py (what to compute) and frames.py
(what to keep)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

DATA = Path(os.environ.get('MMA_WORK', 'work') + '/mcgregor')
# Set by eye from board_check images where the automatic threshold cuts into live play:
# UFC 229's box lands on the scorebar's name panel, which fades with the background; its score splits at 0.15
# (non-live below 0.1, live 0.15-0.75), giving 1122 s live for a 1083 s fight.
OVERRIDE = {'2018_khabib': .15}


def otsu(x: np.ndarray) -> float:
    h, e = np.histogram(x, bins=50, range=(0, 1)); c = (e[:-1] + e[1:]) / 2
    best, thr = -1, .5
    for k in range(1, 50):
        w0, w1 = h[:k].sum(), h[k:].sum()
        if not w0 or not w1: continue
        m0, m1 = (h[:k] * c[:k]).sum() / w0, (h[k:] * c[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best: best, thr = v, e[k]
    return float(np.clip(thr, .3, .65))


def live(vid: str, margin: float = 0.0):
    """(times, live flags, threshold) on the 10-per-second grid shared with pose.py; `margin` seconds of
    padding either side of each live stretch."""
    b = np.load(DATA / 'board' / f'{vid}.npz')
    t, s = b['t'], median_filter(b['score'], size=11, mode='nearest')
    thr = OVERRIDE.get(vid, float(np.clip(.55 * np.percentile(s, 95), .25, .6)))
    on = s > thr
    if margin:
        pad = int(round(margin * 10))
        on = np.convolve(on.astype(int), np.ones(2 * pad + 1, int), mode='same') > 0
    return t, on, thr
