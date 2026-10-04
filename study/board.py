"""Live play = the fight clock is on screen. Broadcasts drop the clock for replays, walkouts, interviews and
between rounds, so the clock is the cleanest marker of live action we have (Peter's step 0 keeps the main
camera for the same reason).

The clock graphic is found, not hand-placed: in the lower 40 % of the frame, the region whose edges stay put
across the whole video while everything behind it moves. Its edge pattern is the template; a frame is live when
most of the template's edges are present.

    python board.py <video id>     ->  $DATA/board/<id>.npz  (t, score, bbox) and a check image
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(2)          # several of these run side by side; OpenCV's default (all cores each) thrashes

DATA = Path(os.environ.get('MMA_WORK', 'work') + '/mcgregor')
WIDTH = 320
RATE = 10


def edges(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy) > 80


def frames(cap, fps, n, rate):
    """Yield (time, small gray frame) at `rate` per second, frames chosen exactly as pose.py chooses them."""
    step = fps / rate; nxt = 0.0; i = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while i < n:
        if i < round(nxt):
            if not cap.grab(): return
            i += 1; continue
        ok, img = cap.read()
        if not ok: return
        nxt += step
        h = int(img.shape[0] * WIDTH / img.shape[1])
        yield i / fps, cv2.cvtColor(cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        i += 1


def main(vid: str) -> None:
    src = DATA / 'raw' / f'{vid}.mp4'
    out = DATA / 'board' / f'{vid}.npz'; out.parent.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(src)); fps = cap.get(5); n = int(cap.get(7))
    # pass 1 (1 per second): which pixels are edges when two fighters are squared up on screen. The detector
    # marks those frames; the clock is the overlay whose edges co-vary with them. Name banners, logos and replay
    # graphics don't (they come with walkouts, replays and interviews), so they drop out.
    from ultralytics import YOLO
    det = YOLO(str(Path(os.environ.get('MMA_WORK', 'work') + '/models/mma-detector/best.pt')))
    E, D = [], []
    cap1 = cv2.VideoCapture(str(src))
    step = fps; nxt = 0.0; i = 0
    while i < n:
        if i < round(nxt):
            if not cap1.grab(): break
            i += 1; continue
        ok, img = cap1.read()
        if not ok: break
        nxt += step; i += 1
        r = det.predict(img, device='mps', conf=.3, imgsz=640, verbose=False)[0]
        hh = (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1]).cpu().numpy() / img.shape[0]
        D.append(float(((hh > .2) & (hh < .8)).sum() >= 2))
        h = int(img.shape[0] * WIDTH / img.shape[1])
        E.append(edges(cv2.cvtColor(cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)))
    E = np.array(E, np.float32); D = np.array(D, np.float32)
    # persistence within the action: across consecutive seconds that both show two fighters, how often a pixel is
    # an edge in both, beyond chance. The clock is fixed on screen during action; canvas logos slide as the
    # camera pans, crowds flicker, banners are absent.
    pair = (D[1:] > 0) & (D[:-1] > 0)
    if pair.sum() < 10:
        print(vid, 'too little action found'); np.savez(out, t=np.zeros(0), score=np.zeros(0), bbox=np.zeros(4)); return
    A, B = E[1:][pair], E[:-1][pair]
    p = (A.mean(0) + B.mean(0)) / 2
    cov = (A * B).mean(0) - p * p
    H = cov.shape[0]
    # the clock: the clock-sized window (24 % x 12 % of the frame) in the lower 40 % with the most persistent edges
    W_ = cov.shape[1]; bw, bh = int(.24 * W_), int(.12 * H)
    mass = np.clip(cov, 0, None); mass[: int(.6 * H)] = 0
    ii = cv2.integral(mass)
    sums = ii[bh:, bw:] - ii[:-bh, bw:] - ii[bh:, :-bw] + ii[:-bh, :-bw]
    y, x = np.unravel_index(np.argmax(sums), sums.shape)
    x0, y0, x1, y1 = x, y, x + bw, y + bh
    win = cov[y0:y1, x0:x1]
    template = win > max(.05, .4 * win.max())   # the clock's own edges: the most persistent pixels
    if template.sum() < 15:
        print(vid, 'no clock found'); np.savez(out, t=np.zeros(0), score=np.zeros(0), bbox=np.zeros(4)); return
    freq = cov
    # pass 2 (10 per second): share of the template's edges present in each frame
    T, S = [], []
    for t, g in frames(cap, fps, n, RATE):
        e = edges(g)[y0:y1, x0:x1]
        T.append(t); S.append((e & template).sum() / max(1, template.sum()))
    scale = cap.get(3) / WIDTH
    np.savez(out, t=np.array(T, np.float32), score=np.array(S, np.float32),
             bbox=np.array([x0, y0, x1, y1]) * scale, template=template)
    print(vid, 'clock box', (np.array([x0, y0, x1, y1]) * scale).astype(int).tolist(), 'live share', round(float((np.array(S) > .6).mean()), 3))


if __name__ == '__main__':
    main(sys.argv[1])
