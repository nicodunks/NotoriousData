"""Pose for one fight: 10 frames a second, the fighters the UFC-trained detector finds (up to three), RTMW-x's
133 points for each, the colour of each fighter's shorts, and a small colour histogram of the whole frame (for
finding camera cuts later). Nothing is decided here; identity, shots and standing come later from what is saved.

    python pose.py <fight id> [--start S] [--end S] [--out NAME]

Writes $DATA/pose/<fight id or NAME>.npz. Run from anywhere; paths are absolute.
"""
from __future__ import annotations

import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(2)          # several of these run side by side; OpenCV's default (all cores each) thrashes
import onnxruntime as ort
import torch
from rtmlib import RTMPose
from ultralytics import YOLO

WORK = Path(os.environ.get('MMA_WORK', 'work'))
DATA = WORK / 'mcgregor'
DETECTOR = WORK / 'models/mma-detector/best.pt'
RTMW = Path.home() / '.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'
RATE = int(__import__('os').environ.get('POSE_RATE', 10))   # frames kept per second (30 = every frame, for strikes)
MAX_PEOPLE = 3     # the two fighters, plus room for a referee the detector lets through
THREADS = 5


def shorts_colour(img: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Median Lab colour of the box between the hips and the knees (where the shorts are); NaN if not visible."""
    pts = kp[[11, 12, 13, 14]]
    if (pts[:, 2] < .4).any():
        return np.full(3, np.nan, np.float32)
    hip_y, knee_y = pts[:2, 1].mean(), pts[2:, 1].mean()
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = hip_y, hip_y + .55 * (knee_y - hip_y)          # upper half of the thigh: shorts, rarely skin
    h, w = img.shape[:2]
    x0, x1, y0, y1 = int(max(0, x0)), int(min(w, x1)), int(max(0, y0)), int(min(h, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.full(3, np.nan, np.float32)
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0).astype(np.float32)


def frame_hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
    return (h / h.sum()).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('fight')
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--live', action='store_true', help='only frames the clock marks as live (live.py), padded 1 s')
    a = ap.parse_args()
    src = DATA / 'raw' / f'{a.fight}.mp4'
    out = DATA / 'pose' / f'{a.out or a.fight}{"" if RATE == 10 else f"_{RATE}"}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print('have', out); return

    torch.set_num_threads(2)
    det = YOLO(str(DETECTOR))
    opts = ort.SessionOptions(); opts.intra_op_num_threads = THREADS; opts.inter_op_num_threads = 1
    pose = RTMPose(str(RTMW), model_input_size=(288, 384), backend='onnxruntime', device='cpu')
    # CoreML (Apple GPU / Neural Engine): ~45x faster than CPU here, same keypoints (max 1 px, scores +-0.04)
    pose.session = ort.InferenceSession(str(RTMW), sess_options=opts, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    first = int(a.start * fps); last = n if a.end is None else min(n, int(a.end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    step = fps / RATE
    allowed = None
    if a.live:
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from live import live
        bt, bon, _ = live(a.fight, margin=1.0)
        allowed = lambda tt: bool(bon[min(len(bon) - 1, int(np.searchsorted(bt, tt)))])
    k = -1                                   # index of this sample on the 10-per-second grid
    T, BOX, KP, COL, HIST = [], [], [], [], []
    t0 = time.perf_counter(); i = first; nxt = float(first)
    while i < last:
        if i < round(nxt):
            if not cap.grab(): break
            i += 1; continue
        nxt += step; k += 1
        if allowed is not None and not allowed(i / fps):
            if not cap.grab(): break
            i += 1; continue
        ok, img = cap.read()
        if not ok: break
        r = det.predict(img, device='mps', conf=.3, imgsz=640, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
        order = np.argsort(-c)[:MAX_PEOPLE]; b, c = b[order], c[order]
        boxes = np.full((MAX_PEOPLE, 5), np.nan, np.float32)
        kps = np.full((MAX_PEOPLE, 133, 3), np.nan, np.float32)
        cols = np.full((MAX_PEOPLE, 3), np.nan, np.float32)
        if len(b):
            xy, sc = pose(img, b.tolist())
            for j in range(len(b)):
                boxes[j, :4], boxes[j, 4] = b[j], c[j]
                kps[j] = np.c_[xy[j], sc[j]]
                cols[j] = shorts_colour(img, kps[j])
        T.append(i / fps); BOX.append(boxes); KP.append(kps); COL.append(cols); HIST.append(frame_hist(img))
        i += 1
        if len(T) % 500 == 0:
            el = time.perf_counter() - t0
            print(f'{a.fight} {i - first}/{last - first} frames, {len(T)} kept, {len(T) / el:.1f}/s', flush=True)
    np.savez_compressed(out, t=np.array(T, np.float32), boxes=np.array(BOX), kps=np.array(KP).astype(np.float16),
                        shorts=np.array(COL), hist=np.array(HIST), fps=fps, width=W, height=H, rate=RATE,
                        source=str(src), start=a.start)
    print('DONE', out, len(T), 'frames', round(time.perf_counter() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
