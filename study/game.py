"""Which one is McGregor? One round per fight, chosen by rule, not taste: the straight left in that fight after which
the opponent's head snapped back furthest (the clearest sign it landed), McGregor's side alternating from
round to round where the fight allows, with the 5 s around it (2.6 s before full
extension, 1.4 s after) tracked continuously in one shot. Both men's tracked points for the drawing; a reveal clip
with McGregor in red.

    python game.py -> results/game/rounds.json, results/game/round_<i>.mp4
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
import fatigue as FA    # noqa: E402

OUT = HERE / 'results' / 'game'; OUT.mkdir(parents=True, exist_ok=True)
TC = json.loads((HERE / 'results' / 'stance' / 'findings.json').read_text())['torso_cm']
PTS = list(range(23))                       # body 0-16 (face 0-4 included) and feet 17-22
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 19), (19, 17), (16, 22), (22, 20)]
PRE, POST, N = 2.6, 1.4, 10


def main():
    W = F.windows()
    evs = [e for e in json.loads((HERE / 'results' / 'left' / 'events.json').read_text()) if e['who'] == 'mcg' and e['kind'] == 'straight']
    rounds = []
    for f in sorted({e['fight'] for e in evs}, key=lambda f: W[f]['date']):
        d = F.load(f, keep='upright', suffix='_30')
        t, shot = d['t'], d['shot']
        cands_ = []
        for e in sorted([e for e in evs if e['fight'] == f], key=lambda e: -(e.get('opp_head_snap') or -9)):
            sel = (t >= e['t_peak'] - PRE) & (t <= e['t_peak'] + POST)
            if sel.sum() < .8 * (PRE + POST) * d['rate'] or len(np.unique(shot[sel])) != 1: continue
            k = d['mcg'][sel][:, PTS]; o = d['opp'][sel][:, PTS]
            if (k[:, [5, 6, 11, 12, 15, 16], 2] >= .4).mean() < .75 or (o[:, [5, 6, 11, 12, 15, 16], 2] >= .4).mean() < .75: continue
            mcx = np.nanmean(d['mcg'][sel][0, [5, 6, 11, 12], 0]); ocx = np.nanmean(d['opp'][sel][0, [5, 6, 11, 12], 0])
            cands_.append((e, sel, bool(mcx < ocx)))
        if not cands_: print(f, 'no clean moment'); continue
        # sides alternate fight to fight so 'always left' can't win: the best-landing clean left with McGregor on the
        # wanted side, else the best-landing clean left
        want_left = len(rounds) % 2 == 0
        e, sel, _ = next((c for c in cands_ if c[2] == want_left), cands_[0])
        idx = np.flatnonzero(sel)
        vid = W[f].get('video', f)
        cap = cv2.VideoCapture(str(F.DATA / 'raw' / f'{vid}.mp4')); Wd, H = cap.get(3), cap.get(4)
        # the frame region both men occupy, 16:9, for the drawing and the clip
        allp = np.concatenate([d[w][idx][:, PTS][d[w][idx][:, PTS, 2] >= .4][:, :2] for w in ('mcg', 'opp')])
        x0, y0 = np.percentile(allp, 1, 0); x1, y1 = np.percentile(allp, 99, 0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w = max(x1 - x0, (y1 - y0) * 16 / 9) * 1.25; h = w * 9 / 16
        w, h = min(w, Wd), min(h, H); w = min(w, h * 16 / 9); h = w * 9 / 16
        bx, by = float(np.clip(cx - w / 2, 0, Wd - w)), float(np.clip(cy - h / 2, 0, H - h))
        def norm(k):
            out = []
            for p in k:
                out.append([round(float((p[0] - bx) / w), 4), round(float((p[1] - by) / h), 4)] if p[2] >= .4 else None)
            return out
        seq = [{'t': round(float(t[i] - t[idx[0]]), 3), 'm': norm(d['mcg'][i][PTS]), 'o': norm(d['opp'][i][PTS])} for i in idx]
        # which figure starts on the left; mirror the round (drawing and clip together) when that keeps the sides even
        mx = np.nanmean([p[0] for p in seq[0]['m'][5:13] if p]); ox = np.nanmean([p[0] for p in seq[0]['o'][5:13] if p])
        want_left = len(rounds) % 2 == 0
        mirror = bool((mx < ox) != want_left)
        if mirror:
            for fr in seq:
                for who in ('m', 'o'): fr[who] = [[round(1 - p[0], 4), p[1]] if p else None for p in fr[who]]
            mx, ox = 1 - mx, 1 - ox
        ft = float(FA.fight_clock(f, np.array([e['t_peak']]))[0]) if not W[f].get('no_clock') else None
        rnd = int(ft // 300) + 1 if ft is not None else None
        # reveal clip: real speed, the same box, McGregor red, opponent grey
        path = OUT / f'round_{len(rounds)}.mp4'
        wr = imageio_ffmpeg.write_frames(str(path), (640, 360), fps=cap.get(5), codec='libx264', macro_block_size=8,
                                         output_params=['-crf', '26', '-preset', 'slow', '-movflags', '+faststart'])
        wr.send(None)
        cap.set(cv2.CAP_PROP_POS_MSEC, t[idx[0]] * 1000)
        while True:
            tt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000; ok, im = cap.read()
            if not ok or tt > t[idx[-1]]: break
            i = int(np.argmin(np.abs(t - tt)))
            if abs(t[i] - tt) < .05:
                for who, col, wd in (('opp', (190, 190, 190), 2), ('mcg', (40, 45, 225), 3)):
                    k = d[who][i]
                    for a, b in BONES:
                        if k[a, 2] >= .4 and k[b, 2] >= .4:
                            cv2.line(im, tuple(np.round(k[a, :2]).astype(int)), tuple(np.round(k[b, :2]).astype(int)), col, wd, cv2.LINE_AA)
            c = cv2.resize(im[int(by):int(by + h), int(bx):int(bx + w)], (640, 360), interpolation=cv2.INTER_AREA)
            if mirror: c = c[:, ::-1]
            wr.send(np.ascontiguousarray(c[:, :, ::-1]))
        wr.close()
        rounds.append({'fight': f, 'date': W[f]['date'], 'opponent': W[f]['opponent'], 'event': W[f].get('event', ''),
                       'round': rnd, 'fist_ms': round(float(e['peak_speed']) * TC / 100, 1), 'setup': e.get('setup'),
                       'mcg_left': bool(mx < ox), 'mirrored': mirror, 'snap': round(float(e.get('opp_head_snap') or 0), 2), 'clip': f'clips/game_{len(rounds)}.mp4', 'seq': seq, 'aspect': 16 / 9})
        print(f, 'round', rnd, 'snap', round(e.get('opp_head_snap') or 0, 2), 'mcg on the', 'left' if mx < ox else 'right', flush=True)
    # ten rounds, spread over the career
    if len(rounds) > N:
        keep = np.unique(np.round(np.linspace(0, len(rounds) - 1, N)).astype(int))
        for j, r in enumerate(rounds):
            if j not in keep: (OUT / f'round_{j}.mp4').unlink(missing_ok=True)
        rounds = [rounds[j] for j in keep]
    for j, r in enumerate(rounds):
        src = OUT / f"round_{r['clip'].split('_')[-1].split('.')[0]}.mp4"
        r['file'] = src.name
    (OUT / 'rounds.json').write_text(json.dumps(rounds, separators=(',', ':')))
    print(len(rounds), 'rounds')


if __name__ == '__main__':
    main()
