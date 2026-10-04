"""Matched clip pairs, early (left panel) against late (right panel), the measured thing drawn on the body.

Chosen by rule, not taste:
  stance  2.4 s side-on windows (both fighters about the same size on screen, well apart, McGregor in southpaw);
          from each era, the windows whose head height and guard are closest to that era's median: typical, not best.
  left    straight lefts with the punching arm seen throughout; from each era, the ones whose fist speed is closest
          to the era median, paired by closest starting range. Synced at full extension.

    python pairs.py   -> results/pairs/*.mp4, results/pairs/pairs.json
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

OUT = HERE / 'results' / 'pairs'; OUT.mkdir(parents=True, exist_ok=True)
TORSO_CM = json.loads((HERE / 'results' / 'stance' / 'findings.json').read_text())['torso_cm']
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 19), (19, 17), (16, 22), (22, 20)]
WHITE, ACC, GREY, INK, GOLD = (245, 245, 245), (40, 45, 210), (175, 175, 175), (30, 28, 26), (60, 190, 240)
PW, PH = 560, 420
FONT = cv2.FONT_HERSHEY_SIMPLEX
LAST_SYNC = 0
CACHE = {}


def fight(f):
    if f not in CACHE:
        CACHE[f] = F.load(f, keep='upright', suffix='_30')
    return CACHE[f]


def at(d, tt):
    i = int(np.argmin(np.abs(d['t'] - tt)))
    return i if abs(d['t'][i] - tt) < .05 else None


def pt(k, i):
    return tuple(np.round(k[i, :2]).astype(int)) if k[i, 2] >= .4 else None


def skeleton(im, k, col, w):
    for a, b in BONES:
        p, q = pt(k, a), pt(k, b)
        if p and q: cv2.line(im, p, q, col, w, cv2.LINE_AA)


def label(im, text, xy, col=WHITE, s=.55, bg=True):
    (tw, th), _ = cv2.getTextSize(text, FONT, s, 1)
    x, y = xy
    if bg: cv2.rectangle(im, (x - 4, y - th - 5), (x + tw + 4, y + 5), (20, 20, 20), -1)
    cv2.putText(im, text, (x, y), FONT, s, col, 1, cv2.LINE_AA)


def head_guard(im, k, col, scale_px):
    """Shoulder line, nose height above it, and wrist drop below it, in cm."""
    ls, rs, nose = pt(k, 5), pt(k, 6), pt(k, 0)
    if not (ls and rs and nose): return None
    sy = (ls[1] + rs[1]) / 2; x0, x1 = min(ls[0], rs[0]) - 40, max(ls[0], rs[0]) + 40
    cv2.line(im, (x0, int(sy)), (x1, int(sy)), col, 1, cv2.LINE_AA)
    cv2.line(im, (nose[0], int(sy)), nose, col, 2, cv2.LINE_AA)
    cv2.circle(im, nose, 4, col, -1, cv2.LINE_AA)
    head_cm = (sy - nose[1]) / scale_px * TORSO_CM
    wr = [pt(k, 9), pt(k, 10)]
    drops = []
    for w in wr:
        if w:
            cv2.line(im, (w[0], int(sy)), w, col, 1, cv2.LINE_AA); cv2.circle(im, w, 4, col, -1, cv2.LINE_AA)
            drops.append((w[1] - sy) / scale_px * TORSO_CM)
    return head_cm, (np.mean(drops) if drops else np.nan)


def mph_sign(im, mph, sub='screen speed · torso lengths a second'):
    """Big speed sign, bottom left: the fist's peak speed on screen, in the fighter's own torso lengths per second
    (the argument arrives in the old picture-plane mph and is converted back). Not a real-world speed."""
    if mph is None or not np.isfinite(mph): return
    v = mph / (TORSO_CM / 100 * 2.237)
    H_, W_ = im.shape[:2]
    txt = f'{v:.1f}'
    sc = 1.5 * W_ / 640
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, sc, 2)
    (sw, _), _ = cv2.getTextSize(sub, FONT, .45 * W_ / 640, 1)
    x, y = int(14 * W_ / 640), H_ - int(16 * W_ / 640)
    cv2.rectangle(im, (x - 10, y - th - int(30 * W_ / 640)), (x + max(tw, sw) + 12, y + 12), (20, 20, 20), -1)
    cv2.putText(im, sub, (x, y - th - int(12 * W_ / 640)), FONT, .45 * W_ / 640, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(im, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, sc, (255, 255, 255), 2, cv2.LINE_AA)


def peak_mph(d, who, hand_idx, t0, t1):
    """Peak wrist speed of one arm between t0 and t1, in mph, from that man's own scale (in the picture plane)."""
    k = d[who]; sel = np.flatnonzero((d['t'] >= t0) & (d['t'] <= t1))
    if len(sel) < 4: return None
    sc = M.shot_scale(k, d['shot'])
    w = k[sel, hand_idx, :2].astype(float); w[k[sel, hand_idx, 2] < .4] = np.nan
    v = np.linalg.norm(np.diff(w, axis=0), axis=1) / np.diff(d['t'][sel]) / sc[sel[1:]] * TORSO_CM / 100 * 2.237
    v = v[np.isfinite(v) & (np.diff(d['shot'][sel]) == 0)]
    return float(np.percentile(v, 90)) if len(v) >= 3 else None


def crop_box(ks, H, W):
    seen_ = [k[k[:, 2] >= .4, :2] for k in ks if (k[:, 2] >= .4).any()]
    if not seen_:                                               # nothing tracked: the whole frame, at the panel's shape
        h = min(H, W * PH / PW); w = h * PW / PH
        return int((W - w) / 2), int((H - h) / 2), int(w), int(h)
    pts = np.concatenate(seen_)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(x1 - x0, (y1 - y0) * PW / PH) * 1.18; h = w * PH / PW
    w, h = min(w, W), min(h, H)
    return int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h)), int(w), int(h)


def panel(f, t0, t1, tsync, draw, title, sub):
    """Frames of one panel from t0 to t1 (s), already annotated; returns list of images and the sync index."""
    W_ = F.windows()[f]; vid = W_.get('video', f); d = fight(f)
    cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4')); fps = cap.get(5)
    H, Wd = int(cap.get(4)), int(cap.get(3))
    sel = (d['t'] >= t0) & (d['t'] <= t1)
    box = crop_box(list(d['mcg'][sel]) + list(d['opp'][sel]), H, Wd)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    out, sync = [], 0
    while True:
        tt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, im = cap.read()
        if not ok or tt > t1: break
        i = at(d, tt)
        if i is not None:
            draw(im, d, i, tt)
        x, y, w, h = box
        c = cv2.resize(im[y:y + h, x:x + w], (PW, PH), interpolation=cv2.INTER_AREA)
        c = (c * .82).astype(np.uint8)                                   # dim the footage a touch so the lines read
        label(c, title, (12, 26), s=.6); label(c, sub, (12, 50), GREY, s=.45)
        if abs(tt - tsync) < .5 / fps: sync = len(out)
        out.append(c)
    global LAST_SYNC
    LAST_SYNC = sync
    return out, fps


def write(name, left, right, fps, caption):
    n = max(len(left), len(right))
    path = OUT / f'{name}.mp4'
    wr = imageio_ffmpeg.write_frames(str(path), (PW * 2 + 8, PH + 40), fps=fps, codec='libx264', macro_block_size=8,
                                     output_params=['-crf', '26', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    wr.send(None)
    for j in range(n):
        a = left[min(j, len(left) - 1)]; b = right[min(j, len(right) - 1)]
        fr = np.full((PH + 40, PW * 2 + 8, 3), 20, np.uint8)
        fr[:PH, :PW] = a; fr[:PH, PW + 8:] = b
        label(fr, caption, (12, PH + 26), WHITE, .55, bg=False)
        wr.send(np.ascontiguousarray(fr[:, :, ::-1]))
    wr.close()
    return f'pairs/{name}.mp4'


def stance_windows():
    """Candidate 2.4 s side-on southpaw windows per fight, with McGregor's median head height and guard (cm)."""
    W = F.windows(); cands = []
    for f in W:
        if 'exclude' in W[f] or not (F.DATA / 'pose' / f"{W[f].get('video', f)}_30.npz").exists() or not W[f]['shorts']: continue
        d = fight(f)
        if d['n'] < 300: continue
        t, shot = d['t'], d['shot']
        q = M.per_frame(d['mcg'], d['opp'], shot, t); qo = M.per_frame(d['opp'], d['mcg'], shot, t)
        ratio = q['scale'] / qo['scale']; sep = np.abs(q['hip_x'] * q['scale'] - qo['hip_x'] * qo['scale']) / ((q['scale'] + qo['scale']) / 2)
        good = (ratio > .85) & (ratio < 1.18) & (sep > 1.4) & (q['orthodox'] == -1)
        sh_y = M.midp(d['mcg'], 5, 6)[:, 1]
        head = (sh_y - M.P(d['mcg'], 0)[:, 1]) / q['scale'] * TORSO_CM
        r = M.runs(t, shot, max_gap=.1)
        for g in np.unique(r):
            idx = np.flatnonzero(r == g)
            for a in range(0, len(idx) - 72, 36):
                seg = idx[a:a + 72]
                if good[seg].mean() < .8: continue
                cands.append({'fight': f, 'date': W[f]['date'], 't0': float(t[seg[0]]), 't1': float(t[seg[-1]]),
                              'head': float(np.nanmedian(head[seg])), 'guard': float(np.nanmedian(-q['guard'][seg] * TORSO_CM)),
                              'range': float(np.nanmedian(q['range'][seg]))})
    return cands


def make_stance():
    c = stance_windows()
    out = []
    for era in ('early', 'late'):
        e = [x for x in c if (x['date'] < '2016') == (era == 'early')]
        mh, mg = np.median([x['head'] for x in e]), np.median([x['guard'] for x in e])
        for x in e: x['typ'] = abs(x['head'] - mh) / 5 + abs(x['guard'] - mg) / 8
    early = sorted([x for x in c if x['date'] < '2016'], key=lambda x: x['typ'])
    late = sorted([x for x in c if x['date'] >= '2016'], key=lambda x: x['typ'])
    used_e, used_l = set(), set()
    pairs = []
    for x in early:
        if x['fight'] in used_e: continue
        y = min((y for y in late if y['fight'] not in used_l), key=lambda y: abs(y['range'] - x['range']) + y['typ'], default=None)
        if not y: break
        used_e.add(x['fight']); used_l.add(y['fight']); pairs.append((x, y))
        if len(pairs) == 3: break

    def draw(im, d, i, tt):
        k = d['mcg'][i]; o = d['opp'][i]
        sc = M.shot_scale(d['mcg'], d['shot'])[i]; so = M.shot_scale(d['opp'], d['shot'])[i]
        skeleton(im, o, GREY, 1); skeleton(im, k, WHITE, 2)
        r1 = head_guard(im, o, GREY, so); r2 = head_guard(im, k, ACC, sc)
        if r2:
            label(im, f'head {r2[0]:+.0f} cm', (pt(k, 0)[0] + 10, pt(k, 0)[1] - 6), (60, 70, 230), .7)
        if r1 and pt(o, 0):
            label(im, f'{r1[0]:+.0f} cm', (pt(o, 0)[0] + 10, pt(o, 0)[1] - 6), GREY, .6)

    for j, (x, y) in enumerate(pairs):
        L, fps = panel(x['fight'], x['t0'], x['t1'], x['t0'], draw, f"{x['date'][:4]} vs {F.windows()[x['fight']]['opponent']}", 'standing, side-on')
        R, _ = panel(y['fight'], y['t0'], y['t1'], y['t0'], draw, f"{y['date'][:4]} vs {F.windows()[y['fight']]['opponent']}", 'standing, side-on')
        out.append({'file': write(f'stance_{j}', L, R, fps, 'red: McGregor  ·  grey: opponent  ·  head height above the shoulder line, hands below it'),
                    'early': x, 'late': y})
    return out


def make_left():
    evs = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['who'] == 'mcg' and e['kind'] == 'straight']
    out = []
    for era in (0, 1):
        es = [e for e in evs if (e['date'] >= '2016') == bool(era)]
        med = np.median([e['peak_speed'] for e in es])
        for e in es: e['typ'] = abs(e['peak_speed'] - med)
    early = sorted([e for e in evs if e['date'] < '2016'], key=lambda e: e['typ'])
    late = sorted([e for e in evs if e['date'] >= '2016'], key=lambda e: e['typ'])
    pairs, ue, ul = [], set(), set()
    for x in early:
        if x['fight'] in ue: continue
        y = min((y for y in late if y['fight'] not in ul), key=lambda y: abs(y['range_at_onset'] - x['range_at_onset']) + y['typ'] / 10, default=None)
        if not y: break
        ue.add(x['fight']); ul.add(y['fight']); pairs.append((x, y))
        if len(pairs) == 3: break

    def drawer(e):
        trail = []

        def draw(im, d, i, tt):
            k = d['mcg'][i]; skeleton(im, d['opp'][i], GREY, 1); skeleton(im, k, WHITE, 2)
            fist = pt(k, 9); face = [pt(k, j) for j in range(5) if pt(k, j)]
            if e['t_onset'] - .3 <= tt <= e['t_peak'] + .05 and fist:
                trail.append((fist, tuple(np.mean(face, 0).astype(int)) if face else None))
            def smooth(ps):
                ps = [p for p in ps if p is not None]
                if len(ps) < 3: return ps
                a = np.array(ps, float); k = np.ones(3) / 3
                return [tuple(map(int, p)) for p in np.c_[np.convolve(a[:, 0], k, 'valid'), np.convolve(a[:, 1], k, 'valid')]]
            fp, hp = smooth([a for a, _ in trail]), smooth([b for _, b in trail])
            for a, b in zip(fp[:-1], fp[1:]): cv2.line(im, a, b, ACC, 3, cv2.LINE_AA)
            for a, b in zip(hp[:-1], hp[1:]): cv2.line(im, a, b, GOLD, 2, cv2.LINE_AA)
            if fist: cv2.circle(im, fist, 6, ACC, -1, cv2.LINE_AA)
            if abs(tt - e['t_peak']) < .02: label(im, 'full extension', (12, PH - 20), (60, 70, 230))
        return draw

    for j, (x, y) in enumerate(pairs):
        dx, dy = drawer(x), drawer(y)
        span = .9
        L, fps = panel(x['fight'], x['t_peak'] - span - .3, x['t_peak'] + .5, x['t_peak'], dx, f"{x['date'][:4]} vs {x['opponent']}", x['setup'])
        for fr in L[max(0, LAST_SYNC - 1):]: mph_sign(fr, x['peak_speed'] * TORSO_CM / 100 * 2.237)
        R, _ = panel(y['fight'], y['t_peak'] - span - .3, y['t_peak'] + .5, y['t_peak'], dy, f"{y['date'][:4]} vs {y['opponent']}", y['setup'])
        for fr in R[max(0, LAST_SYNC - 1):]: mph_sign(fr, y['peak_speed'] * TORSO_CM / 100 * 2.237)
        out.append({'file': write(f'left_{j}', L, R, fps, 'red: the fist\'s path  ·  gold: the head\'s path  ·  from 0.3 s before the punch to full extension'),
                    'early': {k: x[k] for k in ('fight', 'date', 'opponent', 'setup', 'peak_speed', 't_peak')},
                    'late': {k: y[k] for k in ('fight', 'date', 'opponent', 'setup', 'peak_speed', 't_peak')}})
    return out


def make_counters():
    """Pull counter (left panel) beside slip counter (right panel), the most typical of each: closest to that set-up's
    median head travel (back for the pull, down for the slip). Gold traces the head from 0.8 s before the punch
    starts, red the fist from the start of the punch to full extension. Synced at the start of the punch."""
    evs = {(e['fight'], round(e['t_onset'], 3)): e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['who'] == 'mcg'}
    sl = [r for r in json.loads((HERE / 'results' / 'left' / 'slips.json').read_text())['events'] if r['who'] == 'mcg']
    for r in sl: r['ev'] = evs.get((r['fight'], round(r['t_onset'], 3)))
    pulls = [r for r in sl if r['setup'] == 'pull counter' and r['ev']]
    dips = [r for r in sl if r['setup'] == 'slip counter' and r['ev']]
    mb, md = np.median([r['head_back'] for r in pulls]), np.median([r['head_down'] for r in dips])
    pulls.sort(key=lambda r: abs(r['head_back'] - mb)); dips.sort(key=lambda r: abs(r['head_down'] - md))
    pairs, used = [], set()
    for x in pulls:
        y = next((y for y in dips if y['fight'] not in used and y['fight'] != x['fight']), None)
        if not y or x['fight'] in used: continue
        used |= {x['fight'], y['fight']}; pairs.append((x, y))
        if len(pairs) == 3: break

    def drawer(r):
        e = r['ev']; trail = []

        def draw(im, d, i, tt):
            k = d['mcg'][i]; skeleton(im, d['opp'][i], GREY, 1); skeleton(im, k, WHITE, 2)
            fist = pt(k, 9); face = [pt(k, j) for j in range(5) if pt(k, j)]
            hd = tuple(np.mean(face, 0).astype(int)) if face else None
            if e['t_onset'] - .8 <= tt <= e['t_peak'] + .05:
                trail.append((fist if tt >= e['t_onset'] else None, hd))
            def smooth(ps):
                ps = [p for p in ps if p is not None]
                if len(ps) < 3: return ps
                a = np.array(ps, float); kk = np.ones(3) / 3
                return [tuple(map(int, p)) for p in np.c_[np.convolve(a[:, 0], kk, 'valid'), np.convolve(a[:, 1], kk, 'valid')]]
            fp, hp = smooth([a for a, _ in trail]), smooth([b for _, b in trail])
            for a, b in zip(hp[:-1], hp[1:]): cv2.line(im, a, b, GOLD, 3, cv2.LINE_AA)
            for a, b in zip(fp[:-1], fp[1:]): cv2.line(im, a, b, ACC, 3, cv2.LINE_AA)
            if hd: cv2.circle(im, hd, 5, GOLD, -1, cv2.LINE_AA)
            if abs(tt - e['t_onset']) < .02: label(im, 'the left starts', (12, PH - 20), (60, 70, 230))
        return draw

    out = []
    for j, (x, y) in enumerate(pairs):
        L, fps = panel(x['fight'], x['ev']['t_onset'] - 1.0, x['ev']['t_peak'] + .4, x['ev']['t_onset'], drawer(x),
                       f"Pull counter · {x['date'][:4]} vs {x['ev']['opponent']}", f"head back {x['head_back'] * TORSO_CM:.0f} cm, then the left")
        R, _ = panel(y['fight'], y['ev']['t_onset'] - 1.0, y['ev']['t_peak'] + .4, y['ev']['t_onset'], drawer(y),
                     f"Slip counter · {y['date'][:4]} vs {y['ev']['opponent']}", f"head down {y['head_down'] * TORSO_CM:.0f} cm, the left on the way down")
        out.append({'file': write(f'counter_{j}', L, R, fps / 2, 'gold: his head, from 0.8 s before  ·  red: the fist  ·  half speed'),
                    'pull': {k: x[k] for k in ('fight', 'date', 't_onset', 'head_back', 'lead_ms')},
                    'slip': {k: y[k] for k in ('fight', 'date', 't_onset', 'head_down', 'lead_ms')}})
    return out


def make_measures():
    """One clip per stance measure: four panels, one per era (2012-13, 2014-15, 2016, 2018-21), each cropped to
    McGregor with the measure drawn on his body and its value live (running median over 9 frames, so keypoint
    jitter doesn't flicker the number). Each panel is the 2.4 s side-on window from that era whose median of the
    measure is closest to the era's median window and whose values stay steady inside it (half-weighted 10-90 %
    range): typical for that measure in that era, not mid-punch. Panels play in step at 30 /s."""
    import stance_deep as SD
    KEYS = ('blade', 'width', 'length', 'crouch', 'rear_knee', 'lean', 'weight', 'step_speed', 'shoulder')
    ERAS_ = [('2012–13', '2012', '2014'), ('2014–15', '2014', '2016'), ('2016', '2016', '2017'), ('2021', '2021', '2022')]
    cands = stance_windows()
    for c in cands:
        dd = fight(c['fight']); sel_ = (dd['t'] >= c['t0']) & (dd['t'] <= c['t1'])
        m, _ = SD.measures(dd['mcg'], dd['opp'], dd['t'], dd['shot'])
        c['m'] = {k: float(np.nanmedian(m[k][sel_])) if np.isfinite(m[k][sel_]).any() else np.nan for k in KEYS}
        c['iqr'] = {k: float(np.subtract(*np.nanpercentile(m[k][sel_], [90, 10]))) if np.isfinite(m[k][sel_]).sum() > 10 else np.nan for k in KEYS}
        c['conf'] = float(np.median(dd['mcg'][sel_][:, :17, 2]))
    MW, MH = 300, 360

    def load(st):
        f, t0, t1 = st['fight'], st['t0'], st['t1']
        d = fight(f); W_ = F.windows()[f]; vid = W_.get('video', f)
        cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4')); fps = cap.get(5); H, Wd = int(cap.get(4)), int(cap.get(3))
        sel = (d['t'] >= t0) & (d['t'] <= t1)
        pts_ = np.concatenate([k[:17][k[:17, 2] >= .4, :2] for k in d['mcg'][sel]])      # body points only
        x0, y0 = np.percentile(pts_, 2, 0); x1, y1 = np.percentile(pts_, 98, 0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = min((y1 - y0) * 1.3, H); w = h * MW / MH
        if w > Wd: w = Wd; h = w * MH / MW
        box = int(np.clip(cx - w / 2, 0, Wd - w)), int(np.clip(cy - h / 2, 0, H - h)), int(w), int(h)
        frames = []
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        while True:
            tt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000; ok, im = cap.read()
            if not ok or tt > t1: break
            frames.append((tt, im))
        return d, fps, box, frames

    RED = ACC
    def P(k, j): return np.array(k[j, :2], float) if k[j, 2] >= .4 else None
    def ip(p): return tuple(np.round(p).astype(int))

    def draw(name, im, i, d):
        """Draw the measure; return its value in display units (or nan)."""
        sc = M.shot_scale(d['mcg'], d['shot']); hipx = M.midp(d['mcg'], 11, 12)[:, 0]
        k = d['mcg'][i]; o = d['opp'][i]; s_ = sc[i]
        cm = lambda px: px / s_ * TORSO_CM
        fwd = np.sign(np.nanmean(o[[11, 12], 0]) - np.nanmean(k[[11, 12], 0])) or 1
        lead = dict(a=16, k=14, h=12, s=6); rear = dict(a=15, k=13, h=11, s=5)      # southpaw
        skeleton(im, k, WHITE, 2)
        hip = (P(k, 11) + P(k, 12)) / 2 if P(k, 11) is not None and P(k, 12) is not None else None
        sh = (P(k, 5) + P(k, 6)) / 2 if P(k, 5) is not None and P(k, 6) is not None else None
        la, ra = P(k, lead['a']), P(k, rear['a'])
        if name == 'width' and la is not None and ra is not None:
            cv2.line(im, ip(la), ip(ra), RED, 3, cv2.LINE_AA); return cm(np.linalg.norm(la - ra))
        if name == 'length' and la is not None and ra is not None:
            y = int(max(la[1], ra[1]) + 14); cv2.line(im, (int(ra[0]), y), (int(la[0]), y), RED, 3, cv2.LINE_AA)
            for a in (la, ra): cv2.line(im, (int(a[0]), y - 8), (int(a[0]), y + 8), RED, 2, cv2.LINE_AA)
            return cm(abs(la[0] - ra[0]))
        if name == 'crouch' and hip is not None and la is not None and ra is not None:
            gy = (la[1] + ra[1]) / 2; cv2.line(im, (int(hip[0]), int(gy)), ip(hip), RED, 3, cv2.LINE_AA)
            cv2.line(im, (int(hip[0]) - 30, int(gy)), (int(hip[0]) + 30, int(gy)), RED, 1, cv2.LINE_AA)
            cv2.circle(im, ip(hip), 5, RED, -1, cv2.LINE_AA); return cm(gy - hip[1])
        if name == 'rear_knee' and all(P(k, rear[j]) is not None for j in ('h', 'k', 'a')):
            h_, k_, a_ = P(k, rear['h']), P(k, rear['k']), P(k, rear['a'])
            cv2.line(im, ip(h_), ip(k_), RED, 3, cv2.LINE_AA); cv2.line(im, ip(k_), ip(a_), RED, 3, cv2.LINE_AA)
            u, v = h_ - k_, a_ - k_
            return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        if name == 'lean' and hip is not None and sh is not None:
            cv2.line(im, ip(hip), ip(sh), RED, 3, cv2.LINE_AA)
            L_ = np.linalg.norm(sh - hip); cv2.line(im, ip(hip), (int(hip[0]), int(hip[1] - L_)), GREY, 1, cv2.LINE_AA)
            return float(np.degrees(np.arctan2((sh[0] - hip[0]) * fwd, hip[1] - sh[1])))
        if name == 'weight' and hip is not None and la is not None and ra is not None:
            y = int(max(la[1], ra[1]) + 14); cv2.line(im, (int(ra[0]), y), (int(la[0]), y), GREY, 2, cv2.LINE_AA)
            cv2.line(im, ip(hip), (int(hip[0]), y), RED, 1, cv2.LINE_AA); cv2.circle(im, (int(hip[0]), y), 6, RED, -1, cv2.LINE_AA)
            return (hip[0] - ra[0]) / (la[0] - ra[0]) if abs(la[0] - ra[0]) > 1 else np.nan
        if name == 'step_speed' and hip is not None:
            j0, j1 = max(0, i - 3), min(len(d['t']) - 1, i + 3)
            v = (hipx[j1] - hipx[j0]) / (d['t'][j1] - d['t'][j0]) if d['shot'][j0] == d['shot'][j1] else np.nan
            if np.isfinite(v):
                cv2.arrowedLine(im, ip(hip), (int(hip[0] + v * .25), int(hip[1])), RED, 3, cv2.LINE_AA, tipLength=.25)
                return abs(cm(v))
            return np.nan
        if name == 'blade' and P(k, 5) is not None and P(k, 6) is not None:
            a_, b_ = P(k, 5), P(k, 6)
            cv2.line(im, ip(a_), ip(b_), RED, 4, cv2.LINE_AA)
            for p_ in (a_, b_): cv2.circle(im, ip(p_), 6, RED, -1, cv2.LINE_AA)
            full = d.setdefault('_full', np.nanpercentile(np.linalg.norm(M.P(d['mcg'], 5) - M.P(d['mcg'], 6), axis=-1) / sc, 95))
            return float(np.degrees(np.arcsin(np.clip(abs(a_[0] - b_[0]) / s_ / full, 0, 1))))
        if name == 'shoulder' and P(k, lead['s']) is not None and P(k, rear['s']) is not None:
            ls_, rs_ = P(k, lead['s']), P(k, rear['s'])
            cv2.line(im, ip(rs_), ip(ls_), RED, 3, cv2.LINE_AA)
            for p_ in (ls_, rs_): cv2.circle(im, ip(p_), 5, RED, -1, cv2.LINE_AA)
            cv2.line(im, (int(rs_[0]) - 40, int(rs_[1])), (int(rs_[0]) + 40, int(rs_[1])), GREY, 1, cv2.LINE_AA)
            return cm(rs_[1] - ls_[1])
        return np.nan

    FMT = {'blade': 'shoulders {:.0f}° side-on', 'width': 'width {:.0f} cm', 'length': 'length {:.0f} cm', 'crouch': 'hips {:.0f} cm up', 'rear_knee': 'rear knee {:.0f}°',
           'lean': 'lean {:+.0f}° (+ toward him)', 'weight': 'hips {:.2f} of the way to the lead foot', 'step_speed': 'feet {:.0f} cm/s',
           'shoulder': 'lead shoulder {:+.0f} cm vs rear'}
    out = {}
    TOP = 34
    for name in KEYS:
        panels, info = [], []
        for lab, y0_, y1_ in ERAS_:
            ok_c = [c for c in cands if y0_ <= c['date'] < y1_ and np.isfinite(c['m'][name]) and np.isfinite(c['iqr'][name])]
            # clearly seen: the window's median keypoint confidence in the top half of the era (no cage post in front)
            if ok_c:
                cm_ = np.median([c['conf'] for c in ok_c]); ok_c = [c for c in ok_c if c['conf'] >= cm_]
            if not ok_c: panels.append(None); info.append(None); continue
            target = float(np.median([c['m'][name] for c in ok_c]))
            spread = float(np.std([c['m'][name] for c in ok_c])) or 1
            # best-scoring window whose picture is real footage throughout (no wipe or graphic: every sampled crop
            # has texture)
            for st in sorted(ok_c, key=lambda c: abs(c['m'][name] - target) / spread + .5 * c['iqr'][name] / spread):
                d, fps, box, frames = load(st)
                x, y, w, h = box
                if all(np.std(cv2.cvtColor(im[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)) > 28 for _, im in frames[::6]): break
            ims, vals, ts = [], [], []
            for tt, im0 in frames:
                im = im0.copy(); i = at(d, tt)
                vals.append(draw(name, im, i, d) if i is not None else np.nan); ims.append(im); ts.append(tt - frames[0][0])
            vals = np.array(vals, float); sm = np.full_like(vals, np.nan)
            for j in range(len(vals)):
                w_ = vals[max(0, j - 4):j + 5]
                if np.isfinite(w_).any(): sm[j] = np.nanmedian(w_)
            x, y, w, h = box
            crops = []
            for im, v in zip(ims, sm):
                c = (cv2.resize(im[y:y + h, x:x + w], (MW, MH), interpolation=cv2.INTER_AREA) * .8).astype(np.uint8)
                if np.isfinite(v): label(c, FMT[name].format(v), (8, MH - 12), (235, 235, 235), .45)
                crops.append(c)
            panels.append((np.array(ts), crops))
            info.append({'era': lab, 'fight': st['fight'], 'date': st['date'], 'opponent': F.windows()[st['fight']]['opponent'],
                         't0': st['t0'], 'window_median': st['m'][name], 'era_median': target})
        n = int(round(max(p[0][-1] for p in panels if p) * 30)) + 1
        path = OUT / f'measure_{name}.mp4'
        Wt = MW * 4 + 6 * 3
        wr = imageio_ffmpeg.write_frames(str(path), (Wt, MH + TOP), fps=30, codec='libx264', macro_block_size=2,
                                         output_params=['-crf', '28', '-preset', 'slow', '-movflags', '+faststart'])
        wr.send(None)
        for j in range(n):
            fr = np.full((MH + TOP, Wt, 3), 20, np.uint8)
            for e, (p, inf) in enumerate(zip(panels, info)):
                x0_ = e * (MW + 6)
                if p:
                    ts, crops = p; k = int(np.argmin(np.abs(ts - (j / 30) % (ts[-1] + 1 / 30))))
                    fr[TOP:, x0_:x0_ + MW] = crops[k]
                    label(fr, f"{inf['era']}  vs {inf['opponent'].split()[-1]}", (x0_ + 6, 23), WHITE, .5, bg=False)
            wr.send(np.ascontiguousarray(fr[:, :, ::-1]))
        wr.close()
        out[name] = {'file': f'pairs/measure_{name}.mp4', 'eras': info}
        print(name, [(i['era'], i['fight'], round(i['window_median'], 2)) for i in info if i], flush=True)
    return out


def make_fatigue():
    """Same fight, early (round 1) beside late (minute 8 on): the typical side-on stretch of each window, chosen by
    rule (median guard and stance width of that fight-window), guard and width drawn in cm, steps flashed."""
    import fatigue as FA
    cands = stance_windows()
    for c in cands:
        c['ft'] = float(FA.fight_clock(c['fight'], np.array([c['t0']]))[0])
    out = []
    for f in ('2016_diaz2', '2018_khabib', '2016_diaz1'):
        cs = [c for c in cands if c['fight'] == f]
        e = [c for c in cs if c['ft'] < 300]; l = [c for c in cs if c['ft'] >= 480]
        if not e or not l: continue
        pick = []
        for grp in (e, l):
            mg, mw = np.median([c['guard'] for c in grp]), np.median([c['range'] for c in grp])
            pick.append(min(grp, key=lambda c: abs(c['guard'] - mg) / 5 + abs(c['range'] - mw)))
        d = fight(f)
        st_times = set()

        def drawer(minute):
            def draw(im, d, i, tt):
                k = d['mcg'][i]; sc = M.shot_scale(d['mcg'], d['shot'])[i]
                skeleton(im, d['opp'][i], GREY, 1); skeleton(im, k, WHITE, 2)
                r = head_guard(im, k, ACC, sc)
                la, ra = pt(k, 15), pt(k, 16)
                if la and ra:
                    cv2.line(im, la, ra, GOLD, 3, cv2.LINE_AA)
                    wcm = np.linalg.norm(np.subtract(la, ra)) / sc * TORSO_CM
                    label(im, f'feet {wcm:.0f} cm apart', (min(la[0], ra[0]), max(la[1], ra[1]) + 26), GOLD, .55)
                if r and np.isfinite(r[1]):
                    w_ = pt(k, 9) or pt(k, 10)
                    if w_: label(im, f'hands {abs(r[1]):.0f} cm below', (w_[0] + 10, w_[1] + 6), (60, 70, 230), .55)
            return draw
        L, fps = panel(f, pick[0]['t0'], pick[0]['t1'], pick[0]['t0'], drawer(0), f"Minute {pick[0]['ft'] / 60 + 1:.0f} · {d and F.windows()[f]['opponent']}", 'round 1')
        R, _ = panel(f, pick[1]['t0'], pick[1]['t1'], pick[1]['t0'], drawer(1), f"Minute {pick[1]['ft'] / 60 + 1:.0f} · {F.windows()[f]['opponent']}", f"round {int(pick[1]['ft'] // 300) + 1}")
        out.append({'file': write(f'fatigue_{len(out)}', L, R, fps, 'red: hands below the shoulder line  ·  gold: the feet  ·  same fight, early and late'),
                    'fight': f, 'early_min': pick[0]['ft'] / 60, 'late_min': pick[1]['ft'] / 60})
    return out


def write_single(name, frames_, fps, caption):
    path = OUT / f'{name}.mp4'
    wr = imageio_ffmpeg.write_frames(str(path), (PW, PH + 40), fps=fps, codec='libx264', macro_block_size=8,
                                     output_params=['-crf', '25', '-preset', 'slow', '-movflags', '+faststart'])
    wr.send(None)
    for a in frames_:
        fr = np.full((PH + 40, PW, 3), 20, np.uint8); fr[:PH] = a
        label(fr, caption, (12, PH + 26), WHITE, .5, bg=False)
        wr.send(np.ascontiguousarray(fr[:, :, ::-1]))
    wr.close()
    return f'pairs/{name}.mp4'


def fist_drawer(who, hand_idx, t_hit, mph, trail_from=.5, head=False):
    trail, htrail = [], []
    def draw(im, d, i, tt):
        k = d[who]; o = d['opp' if who == 'mcg' else 'mcg']
        skeleton(im, o[i], GREY, 1); skeleton(im, k[i], WHITE, 2)
        p = pt(k[i], hand_idx)
        if t_hit - trail_from <= tt <= t_hit + .05 and p: trail.append(p)
        if head:
            face = [pt(k[i], j) for j in range(5) if pt(k[i], j)]
            if t_hit - .9 <= tt <= t_hit + .05 and face: htrail.append(tuple(np.mean(face, 0).astype(int)))
        for a, b in zip(htrail[:-1], htrail[1:]): cv2.line(im, a, b, GOLD, 3, cv2.LINE_AA)
        for a, b in zip(trail[:-1], trail[1:]): cv2.line(im, a, b, ACC, 4, cv2.LINE_AA)
        if p and abs(tt - t_hit) < .5: cv2.circle(im, p, 7, ACC, -1, cv2.LINE_AA)
    return draw


def make_drops():
    """The lefts that dropped them: knockdowns and knockouts confirmed by eye, each with the fist's path and the
    fist's peak speed (picture plane, mph)."""
    W = F.windows()
    DROPS = [('2012_buchinger', (737.55, 737.95), 'knockout, round 1 · the pull counter'), ('2014_poirier', (115.2, 116.6), 'knockout, round 1'),
             ('2015_mendes', (793.2, 794.6), 'knockout, round 2'), ('2016_alvarez', (177.6, 177.95), 'knockdown, round 1')]
    ev = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['who'] == 'mcg']
    out = []
    for f, (a, b), sub in DROPS:
        CACHE[f] = F.load(f, keep='any', suffix='_30')          # knockouts aren't upright: every identified frame
        d = fight(f)
        e = next((e for e in ev if e['fight'] == f and a <= e['t_peak'] <= b), None)
        if e: t_hit, mph = e['t_peak'], e['peak_speed'] * TORSO_CM / 100 * 2.237
        else:
            sel = np.flatnonzero((d['t'] >= a) & (d['t'] <= b)); sc = M.shot_scale(d['mcg'], d['shot'])
            w = d['mcg'][sel, 9, :2].astype(float); w[d['mcg'][sel, 9, 2] < .4] = np.nan
            v = np.linalg.norm(np.diff(w, axis=0), axis=1) / np.diff(d['t'][sel]) / sc[sel[1:]]
            j = int(np.nanargmax(v)) if np.isfinite(v).any() else 0
            t_hit = float(d['t'][sel[j + 1]]) if len(sel) > 1 else (a + b) / 2
            mph = peak_mph(d, 'mcg', 9, t_hit - .35, t_hit + .05)
        if f in ('2014_poirier', '2015_mendes'): mph = None          # hook toward the camera / not tracked: no honest speed
        frs, fps = panel(f, t_hit - 2.2, t_hit + 2.0, t_hit, fist_drawer('mcg', 9, t_hit, mph, head=(f == '2012_buchinger')), f"{W[f]['date'][:4]} vs {W[f]['opponent']}", sub)
        for fr in frs[max(0, LAST_SYNC - 1):]: mph_sign(fr, mph)
        out.append({'file': write_single(f'drop_{len(out)}', frs, fps, 'red: his left fist  ·  speed measured in the picture plane'),
                    'fight': f, 'date': W[f]['date'], 'opponent': W[f]['opponent'], 't_hit': t_hit, 'mph': mph, 'sub': sub})
        print('drop', f, round(t_hit, 2), mph and round(mph, 1), flush=True)
    return out


def two(name, specs, caption):
    """Two panels side by side; each spec: (fight, who, hand_idx, t_peak, mph or None, title, sub, after, head)."""
    W = F.windows(); panes = []
    for f, who, hidx, tp, mph, title, sub, after, head in specs:
        CACHE[f] = F.load(f, keep='any', suffix='_30')
        frs, fps = panel(f, tp - 1.7, tp + after, tp, fist_drawer(who, hidx, tp, mph, head=head), title, sub)
        for fr in frs[max(0, LAST_SYNC - 1):]: mph_sign(fr, mph)
        panes.append(frs)
    return write(name, panes[0], panes[1], fps, caption)


def make_vs():
    """His left beside his opponent's rear straight, same fight, both at full speed, the fist's peak speed signed."""
    ev = json.loads((HERE / 'results' / 'left' / 'events.json').read_text())
    E = lambda f, who, tp: next(e for e in ev if e['fight'] == f and e['who'] == who and abs(e['t_peak'] - tp) < .2)
    mph = lambda e: e['peak_speed'] * TORSO_CM / 100 * 2.237
    W = F.windows(); out = []
    for f, tm, to, sub_m, after_m in (('2016_alvarez', 308.1, 387.9, 'his fastest left', .8), ('2016_alvarez', 177.84, 337.3, 'the left that dropped him', 1.8),
                                      ('2015_siver', 246.9, 342.6, 'his fastest left', .8)):
        em, eo = E(f, 'mcg', tm), E(f, 'opp', to)
        opp = W[f]['opponent']
        out.append({'file': two(f'vs_{len(out)}', [(f, 'mcg', 9, em['t_peak'], mph(em), f"McGregor · {W[f]['date'][:4]}", sub_m, after_m, False),
                                                     (f, 'opp', 10 if eo['hand'] == 'right' else 9, eo['t_peak'], mph(eo), f"{opp}", 'his fastest rear straight', .8, False)],
                                'red: the fist  ·  speed in the picture plane  ·  real speed'),
                    'fight': f, 'mcg_mph': mph(em), 'opp_mph': mph(eo), 'sub': sub_m})
        print('vs', f, round(mph(em), 1), round(mph(eo), 1), flush=True)
    return out


def make_counter_ivan_eddie():
    """The pull counter that finished Buchinger beside the end of Alvarez: Alvarez throws, McGregor's left meets him,
    and the finish follows. The Alvarez close-up is barely tracked, so it is shown as footage with timed labels."""
    W = F.windows()
    CACHE['2012_buchinger'] = F.load('2012_buchinger', keep='any', suffix='_30')
    L, fps = panel('2012_buchinger', 737.87 - 1.7, 737.87 + 1.6, 737.87, fist_drawer('mcg', 9, 737.87, None, head=True),
                   'Pull counter · 2012 vs Ivan Buchinger', 'head back, then the left')
    CACHE['2016_alvarez'] = F.load('2016_alvarez', keep='any', suffix='_30')
    def cue(im, d, i, tt): pass
    t0 = 548.5
    R, _ = panel('2016_alvarez', t0, t0 + 3.3, t0, cue, 'The finish · 2016 vs Eddie Alvarez', 'he throws; the left meets him')
    for j, fr in enumerate(R):
        tt = t0 + j / fps
        if 549.5 <= tt < 549.9: label(fr, 'Alvarez throws', (PW // 2 - 70, PH - 30), (230, 230, 230), .75)
        elif 549.9 <= tt < 550.5: label(fr, "McGregor's left", (PW // 2 - 80, PH - 30), (80, 90, 240), .75)
        elif 550.5 <= tt < 551.6: label(fr, 'and the finish', (PW // 2 - 70, PH - 30), (230, 230, 230), .75)
    return {'file': write('counter_ie', L, R, fps, 'gold: his head, from 0.9 s before  ·  red: the fist  ·  real speed')}


def _old_counter_ivan_eddie():
    """The pull counter that finished Buchinger beside the slip counter on Alvarez: head traced gold, fist red."""
    return {'file': two('counter_ie', [('2012_buchinger', 'mcg', 9, 737.87, None, 'Pull counter · 2012 vs Ivan Buchinger', 'head back, then the left', 1.6, True),
                                        ('2016_alvarez', 'mcg', 9, 268.0, None, 'Slip counter · 2016 vs Eddie Alvarez', 'head down, the left on the way', .9, True)],
                        'gold: his head, from 0.9 s before  ·  red: the fist  ·  real speed')}


def make_blade():
    """Bladed against squared: from Mendes (2015) and Alvarez (2016), the side-on stretch where his shoulders were
    most side-on relative to his opponent's, steadily. Both shoulder lines drawn, angles live."""
    import stance_deep as SD
    cands = [c for c in stance_windows() if c['fight'] in ('2015_mendes', '2016_alvarez')]
    for c in cands:
        dd = fight(c['fight']); sel_ = (dd['t'] >= c['t0']) & (dd['t'] <= c['t1'])
        m1, _ = SD.measures(dd['mcg'], dd['opp'], dd['t'], dd['shot']); m2, _ = SD.measures(dd['opp'], dd['mcg'], dd['t'], dd['shot'])
        c['bm'], c['bo'] = float(np.nanmedian(m1['blade'][sel_])), float(np.nanmedian(m2['blade'][sel_]))
        c['bsd'] = float(np.nanstd(m1['blade'][sel_]))
    picks = []
    for f in ('2015_mendes', '2016_alvarez'):
        cs = [c for c in cands if c['fight'] == f and np.isfinite(c['bm']) and np.isfinite(c['bo'])]
        picks.append(max(cs, key=lambda c: (c['bm'] - c['bo']) - .5 * c['bsd']))
    full = {}

    def drawer(f):
        d = fight(f)
        fw = {w: np.nanpercentile(np.linalg.norm(M.P(d[w], 5) - M.P(d[w], 6), axis=-1) / M.shot_scale(d[w], d['shot']), 95) for w in ('mcg', 'opp')}
        def draw(im, d, i, tt):
            for w, col, txtcol in (('opp', GREY, GREY), ('mcg', ACC, (60, 70, 230))):
                k = d[w][i]; skeleton(im, k, WHITE if w == 'mcg' else GREY, 2 if w == 'mcg' else 1)
                a_, b_ = pt(k, 5), pt(k, 6)
                if a_ and b_:
                    cv2.line(im, a_, b_, col, 4, cv2.LINE_AA)
                    sc = M.shot_scale(d[w], d['shot'])[i]
                    ang = np.degrees(np.arcsin(np.clip(abs(a_[0] - b_[0]) / sc / fw[w], 0, 1)))
                    label(im, f'{ang:.0f}° side-on', (min(a_[0], b_[0]), min(a_[1], b_[1]) - 14), txtcol, .6)
        return draw
    W = F.windows()
    L, fps = panel(picks[0]['fight'], picks[0]['t0'], picks[0]['t1'], picks[0]['t0'], drawer(picks[0]['fight']), f"{W[picks[0]['fight']]['date'][:4]} vs {W[picks[0]['fight']]['opponent']}", 'his shoulders against his opponent\'s')
    R, _ = panel(picks[1]['fight'], picks[1]['t0'], picks[1]['t1'], picks[1]['t0'], drawer(picks[1]['fight']), f"{W[picks[1]['fight']]['date'][:4]} vs {W[picks[1]['fight']]['opponent']}", 'his shoulders against his opponent\'s')
    print('blade', [(p['fight'], round(p['bm']), round(p['bo'])) for p in picks])
    return {'file': write('blade_pair', L, R, fps, 'red: his shoulder line  ·  grey: his opponent\'s  ·  90° is fully side-on, 0° squared up'),
            'picks': [{k: p[k] for k in ('fight', 't0', 'bm', 'bo')} for p in picks]}


if __name__ == '__main__':
    old = json.loads((OUT / 'pairs.json').read_text()) if (OUT / 'pairs.json').exists() else {}
    which = sys.argv[1:] or ['stance', 'left', 'counters', 'measures']
    res = dict(old)
    for w in which: res[w] = {'stance': make_stance, 'left': make_left, 'counters': make_counters, 'measures': make_measures, 'fatigue': make_fatigue, 'drops': make_drops, 'vs': make_vs, 'counter_ie': make_counter_ivan_eddie, 'blade': make_blade}[w]()
    (OUT / 'pairs.json').write_text(json.dumps(res, indent=1, default=float))
    for p in res.get('counters', []): print(p['file'], p['pull']['fight'], p['slip']['fight'])
