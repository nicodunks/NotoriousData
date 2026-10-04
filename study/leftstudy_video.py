"""The left, studied: real footage of his lefts (left panel, one after another) beside every straight left he threw,
as tracked vectors stacked in step (right panel), the one on screen drawn bold. All time-aligned the same way:
0.8 s before the punch starts, the punch (start to full extension, stretched to a fixed length), 0.3 s after.
Half speed.

    python leftstudy_video.py -> results/pairs/left_study.mp4
"""
import json
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402
import metrics as M     # noqa: E402
import pairs as P       # noqa: E402

PRE, MID, POST = 24, 8, 9
N = PRE + MID + POST
SLOW = 2
PW, PH = P.PW, P.PH
BONES = P.BONES
KO = {('2012_buchinger', 737.87), ('2016_alvarez', 177.84)}


def times(e):
    on, pk = e['t_onset'], e['t_peak']
    return np.r_[on + np.arange(-PRE, 0) / 30, np.linspace(on, pk, MID, endpoint=False), pk + np.arange(POST) / 30]


def main():
    W = F.windows()
    evs = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['who'] == 'mcg' and e['kind'] == 'straight']
    seqs, cand = [], []
    for e in evs:
        d = P.fight(e['fight']); t = d['t']; ts = times(e)
        idx = [int(np.argmin(np.abs(t - x))) for x in ts]
        ok = np.array([abs(t[i] - x) < .025 for i, x in zip(idx, ts)])
        if ok.mean() < .8: continue
        k = d['mcg'][idx]; o = d['opp'][idx]
        sc = M.shot_scale(d['mcg'], d['shot'])[idx]
        hip = M.midp(k, 11, 12); fwd = np.sign(np.nanmedian(M.midp(o, 11, 12)[:, 0] - hip[:, 0])) or 1
        hip0 = np.nanmedian(hip[:PRE], 0); s0 = np.nanmedian(sc)
        p = (k[:, :23, :2].astype(float) - hip0) / s0; p[..., 0] *= fwd
        p[k[:, :23, 2] < .4] = np.nan; p[~ok] = np.nan
        seqs.append(p)
        same_shot = len(np.unique(d['shot'][idx])) == 1
        seen = (k[:, [5, 6, 7, 9, 11, 12, 15, 16], 2] >= .4).mean()
        if same_shot and ok.all() and seen > .85:
            cand.append((e, idx, len(seqs) - 1))
    # footage: the two knockdown lefts, then the fastest clean left of each other fight, in date order, up to 9
    ko = [c for c in cand if any(c[0]['fight'] == f and abs(c[0]['t_peak'] - tp) < .2 for f, tp in KO)]
    rest = {}
    for c in cand:
        if c in ko: continue
        f = c[0]['fight']
        if f not in rest or c[0]['peak_speed'] > rest[f][0]['peak_speed']: rest[f] = c
    show = sorted(ko + list(rest.values()), key=lambda c: (c[0]['date'], c[0]['t_peak']))[:9]
    print(len(seqs), 'lefts stacked;', len(show), 'shown:', [(c[0]['fight'], round(c[0]['t_peak'], 1)) for c in show])
    A = np.array(seqs)                                             # (n, N, 23, 2)
    med = np.nanmedian(A, 0)
    TOP = 34
    out = HERE / 'results' / 'pairs' / 'left_study.mp4'
    wr = imageio_ffmpeg.write_frames(str(out), (PW * 2 + 8, PH + TOP + 40), fps=30, codec='libx264', macro_block_size=8,
                                     output_params=['-crf', '25', '-preset', 'slow', '-movflags', '+faststart'])
    wr.send(None)
    # right-panel mapping: hips at (0.5 W, 0.62 H), one torso length = PH / 4.6
    s = PH / 5.4; ox, oy = PW * .44, PH * .5
    paper, red, faint, ink = (243, 246, 246), (40, 45, 210), (180, 185, 236), (30, 28, 26)
    def bones(img, pose, col, w):
        for a, b in BONES:
            pa, pb = pose[a], pose[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(img, (int(ox + pa[0] * s), int(oy + pa[1] * s)), (int(ox + pb[0] * s), int(oy + pb[1] * s)), col, w, cv2.LINE_AA)
        f = pose[:5][np.isfinite(pose[:5]).all(1)]
        if len(f): c = f.mean(0); cv2.circle(img, (int(ox + c[0] * s), int(oy + c[1] * s)), int(.19 * s), col, w, cv2.LINE_AA)
    for e, idx, si in show:
        d = P.fight(e['fight']); vid = W[e['fight']].get('video', e['fight'])
        cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4')); Hh, Ww = int(cap.get(4)), int(cap.get(3))
        box = P.crop_box(list(d['mcg'][idx]) + list(d['opp'][idx]), Hh, Ww); x, y, w, h = box
        mph = e['peak_speed'] * P.TORSO_CM / 100 * 2.237
        ko_ = any(e['fight'] == f and abs(e['t_peak'] - tp) < .2 for f, tp in KO)
        trail = []
        for j, i in enumerate(idx):
            cap.set(cv2.CAP_PROP_POS_MSEC, d['t'][i] * 1000); ok, im = cap.read()
            if not ok: continue
            k = d['mcg'][i]
            P.skeleton(im, d['opp'][i], P.GREY, 1); P.skeleton(im, k, P.ACC, 3)
            fp = P.pt(k, 9)
            if PRE <= j < PRE + MID + 2 and fp: trail.append(fp)
            for a_, b_ in zip(trail[:-1], trail[1:]): cv2.line(im, a_, b_, P.GOLD, 4, cv2.LINE_AA)
            L = (cv2.resize(im[y:y + h, x:x + w], (PW, PH), interpolation=cv2.INTER_AREA) * .85).astype(np.uint8)
            if j >= PRE + MID - 1: P.mph_sign(L, mph)
            R = np.full((PH, PW, 3), paper, np.uint8)
            cv2.line(R, (40, int(oy + 2.05 * s)), (PW - 40, int(oy + 2.05 * s)), (210, 210, 210), 1, cv2.LINE_AA)
            G = R.copy()
            for q in A: bones(G, q[j], red, 1)
            R = cv2.addWeighted(G, .22, R, .78, 0)
            bones(R, med[j], red, 4)
            bones(R, A[si][j], ink, 2)
            phase = 'build-up' if j < PRE else 'the punch' if j < PRE + MID else 'follow-through'
            fr = np.full((PH + TOP + 40, PW * 2 + 8, 3), 20, np.uint8)
            fr[TOP:TOP + PH, :PW] = L; fr[TOP:TOP + PH, PW + 8:] = R
            P.label(fr, f"{e['date'][:4]} vs {e['opponent']}" + ('  ·  knockdown' if ko_ else ''), (10, 23), P.WHITE, .55, bg=False)
            P.label(fr, f'every straight left he threw ({len(A)}), in step', (PW + 18, 23), P.WHITE, .55, bg=False)
            P.label(fr, phase, (PW * 2 - 130, 23), (150, 150, 150), .5, bg=False)
            P.label(fr, 'red: his median left  ·  black: the left on the left  ·  gold: the fist  ·  half speed', (10, TOP + PH + 26), P.WHITE, .5, bg=False)
            frame = np.ascontiguousarray(fr[:, :, ::-1])
            for _ in range(SLOW): wr.send(frame)
        for _ in range(8): wr.send(frame)                           # a beat between lefts
    wr.close()
    print('wrote', out)


if __name__ == '__main__':
    main()
