"""Video clips of the left: real footage at half speed, McGregor's tracked skeleton drawn over him in red.

Chosen by rule, not taste: per fight, the fastest straight left whose arm was seen throughout; plus, per set-up
type, the fastest example. Each clip runs from 0.8 s before onset to 0.6 s after full extension, cropped to the
two fighters (fixed crop for the clip), 640 px wide, H.264.

    python clips.py   -> results/left/clips/*.mp4, results/left/clips.json
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

OUT = HERE / 'results' / 'left' / 'clips'
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (15, 19), (19, 17), (16, 22), (22, 20)]
RED = (40, 45, 194); BLUE = (184, 80, 35)


def pick(evs):
    mc = [e for e in evs if e['who'] == 'mcg' and e['kind'] == 'straight' and e.get('peak_speed')]
    chosen = {}
    for e in sorted(mc, key=lambda e: -e['peak_speed']):
        chosen.setdefault(('fight', e['fight']), e)
        chosen.setdefault(('setup', e['setup']), e)
    uniq = {}
    for (kind, key), e in chosen.items():
        uniq.setdefault((e['fight'], e['t_peak']), e)
    return sorted(uniq.values(), key=lambda e: (e['date'], e['t_peak']))


def render(e, d_cache):
    W_ = F.windows()[e['fight']]; vid = W_.get('video', e['fight'])
    if e['fight'] not in d_cache:
        d = np.load(F.DATA / 'pose' / f'{vid}_30.npz')
        d_cache[e['fight']] = (d['t'], d['boxes'], d['kps'].astype(np.float32), F.load(e['fight'], keep='upright', suffix='_30'))
    t_all, boxes, _, fd = d_cache[e['fight']]
    t0, t1 = e['t_onset'] - .8, e['t_peak'] + .6
    sel = (fd['t'] >= t0) & (fd['t'] <= t1)
    if sel.sum() < 10:
        return None
    M_ = fd['mcg'][sel]; O_ = fd['opp'][sel]; ts = fd['t'][sel]
    who = e['who']; Tk = M_ if who == 'mcg' else O_; col = RED if who == 'mcg' else BLUE
    wrist = 9 if e['hand'] == 'left' else 10
    pts = np.concatenate([M_[..., :2][M_[..., 2] > .4], O_[..., :2][O_[..., 2] > .4]])
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4')); fps = cap.get(5)
    H, Wd = int(cap.get(4)), int(cap.get(3))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(x1 - x0, (y1 - y0) * 16 / 9) * 1.25; h = w * 9 / 16
    w, h = min(w, Wd), min(h, H)
    X0 = int(np.clip(cx - w / 2, 0, Wd - w)); Y0 = int(np.clip(cy - h / 2, 0, H - h))
    name = f"{e['who']}_{e['fight']}_{e['t_peak']:.1f}".replace('.', '_') + '.mp4'
    OUT.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(OUT / name), (640, 360), fps=fps / 2, codec='libx264', quality=None,
                                         output_params=['-crf', '27', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    writer.send(None)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    while True:
        tt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, im = cap.read()
        if not ok or tt > t1: break
        j = int(np.argmin(np.abs(ts - tt)))
        if abs(ts[j] - tt) < .02:
            k = Tk[j]
            for a, b in BONES:
                if k[a, 2] > .4 and k[b, 2] > .4:
                    cv2.line(im, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), col, 3, cv2.LINE_AA)
            if k[wrist, 2] > .4: cv2.circle(im, tuple(k[wrist, :2].astype(int)), 7, col, -1, cv2.LINE_AA)
        crop = cv2.resize(im[Y0:Y0 + int(h), X0:X0 + int(w)], (640, 360), interpolation=cv2.INTER_AREA)
        writer.send(np.ascontiguousarray(crop[:, :, ::-1]))
    writer.close()
    return {'file': f'clips/{name}', 'who': e['who'], 'fight': e['fight'], 'date': e['date'], 'opponent': e['opponent'], 'setup': e['setup'],
            'peak_speed': e['peak_speed'], 't_peak': e['t_peak']}


if __name__ == '__main__':
    evs = json.loads((HERE / 'results' / 'left' / 'events.json').read_text())
    cache = {}; out = []
    import shutil; shutil.rmtree(OUT, ignore_errors=True)
    opp = {}
    for e in sorted((e for e in evs if e['who'] == 'opp' and e['kind'] == 'straight' and e.get('peak_speed')), key=lambda e: -e['peak_speed']):
        opp.setdefault(e['fight'], e)
    for e in pick(evs) + sorted(opp.values(), key=lambda e: e['date']):
        r = render(e, cache)
        if r: out.append(r); print(r['file'], r['setup'], round(r['peak_speed'], 1), flush=True)
    (HERE / 'results' / 'left' / 'clips.json').write_text(json.dumps(out, indent=1))
