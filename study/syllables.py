"""McGregor's syllables (keypoint-MoSeq): occurrences, stacked figures, clips, a map, and use over his career.

    python syllables.py sheets        -> results/syllables/sheet_<s>.jpg   (to watch and name each syllable)
    python syllables.py build         -> results/syllables/syllables_site.json, clips/*.mp4 (needs names.json)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import frames as F      # noqa: E402

D = F.DATA / 'moseq'
OUT = HERE / 'results' / 'syllables'; OUT.mkdir(parents=True, exist_ok=True)
idx = json.loads((D / 'index.json').read_text())
recs = {r['name']: r for r in idx['recordings']}
raw = np.load(D / 'data.npz')
syl = json.loads((D / 'syllables.json').read_text())['syllables']
NAMES = idx['bodyparts']
# body-part index in the moseq arrays -> COCO index used by the page's skeleton drawer
COCO = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22]
MIN_SHARE = .01


def occurrences():
    occ = []
    for name, s in syl.items():
        s = np.asarray(s); ch = np.r_[0, np.flatnonzero(np.diff(s)) + 1, len(s)]
        for a, b in zip(ch[:-1], ch[1:]):
            occ.append({'rec': name, 'syl': int(s[a]), 'a': int(a), 'b': int(b)})
    return occ


def canon(rec, a, b, n=16):
    """Occurrence resampled to n frames, hips at origin of the occurrence's first frame, torso 1."""
    xy = raw[f'c__{rec}'][a:b].astype(float); c = raw[f'w__{rec}'][a:b]
    xy[c < .4] = np.nan
    hip = np.nanmean(xy[:, [7, 8]], axis=1); sh = np.nanmean(xy[:, [1, 2]], axis=1)
    torso = np.nanmedian(np.linalg.norm(sh - hip, axis=1))
    if not np.isfinite(torso) or torso <= 0: return None
    p = (xy - np.nanmedian(hip, 0)) / torso
    ti = np.linspace(0, len(p) - 1, n)
    out = np.full((n, 23, 2), np.nan)
    for j, k in enumerate(COCO):
        for d_ in (0, 1):
            v = p[:, j, d_]; ok = np.isfinite(v)
            if ok.sum() >= 2: out[:, k, d_] = np.interp(ti, np.flatnonzero(ok), v[ok])
    return out


def top(occ):
    u, c = np.unique([o['syl'] for o in occ for _ in range(o['b'] - o['a'])], return_counts=True)
    f = c / c.sum(); order = np.argsort(-f)
    return [(int(u[i]), float(f[i])) for i in order if f[i] >= MIN_SHARE]


def sheets():
    occ = occurrences()
    for s, share in top(occ):
        os_ = [o for o in occ if o['syl'] == s and 6 <= o['b'] - o['a'] <= 30]
        rng = np.random.default_rng(s); pick = rng.choice(len(os_), min(6, len(os_)), replace=False)
        rows = []
        for i in pick:
            o = os_[i]; r = recs[o['rec']]; W = F.windows()[r['fight']]
            cap = cv2.VideoCapture(str(F.DATA / 'raw' / f"{W.get('video', r['fight'])}.mp4"))
            tiles = []
            for fr in np.linspace(o['a'], o['b'] - 1, 4).astype(int):
                cap.set(cv2.CAP_PROP_POS_MSEC, (r['t0'] + fr / r['rate']) * 1000); ok, im = cap.read()
                xy = raw[f"c__{o['rec']}"][fr].copy(); cc = raw[f"w__{o['rec']}"][fr]
                fwd = 1 if True else -1
                pts = xy[cc > .4]
                if not ok or not len(pts): tiles.append(np.zeros((200, 200, 3), np.uint8)); continue
                # coords were mirrored when the opponent was on the left: undo by checking which side the hips are in the frame
                xs = pts[:, 0]
                if np.median(xs) < 0: xy[:, 0] *= -1; pts = xy[cc > .4]
                x0, y0 = pts.min(0) - 50; x1, y1 = pts.max(0) + 50
                crop = im[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
                tiles.append(cv2.resize(crop, (200, 200)) if crop.size else np.zeros((200, 200, 3), np.uint8))
            strip = np.hstack(tiles)
            cv2.putText(strip, f"{r['fight']} {(r['t0'] + o['a'] / r['rate']):.1f}s {(o['b'] - o['a']) / r['rate']:.2f}s", (4, 14), 0, .4, (0, 255, 255), 1)
            rows.append(strip)
        cv2.imwrite(str(OUT / f'sheet_{s}.jpg'), np.vstack(rows))
        print(s, f'{share:.1%}', len(os_), 'occurrences', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'sheets':
        sheets()


NAMES_BY_ID = {2: 'The long stance', 3: 'Tall and light', 0: 'Stepping in', 1: 'Lunge and throw', 5: 'Shuffle forward',
               7: 'Square up and walk', 4: 'Feet together', 6: 'Kick'}
DESC = {2: 'Feet wide, weight back, lead hand out: measuring.', 3: 'A short, upright stance on small steps.',
        0: 'The lead foot reaches and the body follows.', 1: 'The torso drives forward over the lead leg: most punches live here.',
        5: 'Both feet slide forward together, closing distance.', 7: 'Squaring to the opponent and walking: a reset.',
        4: 'The feet come together: a step-through or a switch.', 6: 'A leg leaves the floor at speed.'}


def render_clip(o, path):
    import imageio_ffmpeg
    r = recs[o['rec']]; W = F.windows()[r['fight']]
    cap = cv2.VideoCapture(str(F.DATA / 'raw' / f"{W.get('video', r['fight'])}.mp4")); fps = cap.get(5)
    a, b = max(0, o['a'] - 8), min(r['n'], o['b'] + 8)
    xy_all = raw[f"c__{o['rec']}"]; cc = raw[f"w__{o['rec']}"]
    xs = xy_all[a:b][cc[a:b] > .4]
    flip = np.median(xs[:, 0]) < 0
    pts = np.abs(xs) if flip else xs
    pts = np.c_[np.abs(xs[:, 0]) if flip else xs[:, 0], xs[:, 1]]
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = (y1 - y0) * 1.25; w = h * 3 / 4
    H_, W_ = int(cap.get(4)), int(cap.get(3))
    w, h = min(w, W_), min(h, H_)
    X0, Y0 = int(np.clip(cx - w / 2, 0, W_ - w)), int(np.clip(cy - h / 2, 0, H_ - h))
    wr = imageio_ffmpeg.write_frames(str(path), (240, 320), fps=fps / 2, codec='libx264', macro_block_size=8,
                                     output_params=['-crf', '27', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    wr.send(None)
    BONES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (11, 14), (14, 13), (12, 16), (16, 15)]
    for fr in range(a, b):
        cap.set(cv2.CAP_PROP_POS_MSEC, (r['t0'] + fr / r['rate']) * 1000); ok, im = cap.read()
        if not ok: break
        k = xy_all[fr].copy(); c = cc[fr]
        if flip: k[:, 0] *= -1
        inside = o['a'] <= fr < o['b']
        col = (40, 45, 210) if inside else (200, 200, 200)
        for p, q in BONES:
            if c[p] > .4 and c[q] > .4:
                cv2.line(im, tuple(k[p].astype(int)), tuple(k[q].astype(int)), col, 2, cv2.LINE_AA)
        crop = cv2.resize(im[Y0:Y0 + int(h), X0:X0 + int(w)], (240, 320), interpolation=cv2.INTER_AREA)
        wr.send(np.ascontiguousarray(crop[:, :, ::-1]))
    wr.close()


def build():
    from datetime import date
    from scipy.stats import ttest_ind
    from sklearn.manifold import TSNE
    occ = occurrences(); tops = [s for s in top(occ) if s[0] in NAMES_BY_ID]
    EMB = {f"{r['rec']}:{r['a']}": (r['x'], r['y']) for r in json.loads((OUT / 'embed.json').read_text())['points']}
    keep = {s for s, _ in tops}
    (OUT / 'clips').mkdir(exist_ok=True)
    # usage per fight: share of his side-on frames in each syllable
    fights = sorted({recs[r]['fight'] for r in syl}, key=lambda f: F.windows()[f]['date'])
    use = {f: {} for f in fights}; tot = {f: 0 for f in fights}
    for o in occ:
        f = recs[o['rec']]['fight']; n = o['b'] - o['a']; tot[f] += n
        use[f][o['syl']] = use[f].get(o['syl'], 0) + n
    fights = [f for f in fights if tot[f] >= 30 * 30]                 # at least 30 s of side-on footage
    out = {'syllables': [], 'fights': [{'fight': f, 'date': F.windows()[f]['date'], 'opponent': F.windows()[f]['opponent'], 'seconds': tot[f] / 30} for f in fights]}
    ps = []
    rng = np.random.default_rng(0)
    feats, labels = [], []
    for s, share in tops:
        os_ = [o for o in occ if o['syl'] == s and 5 <= o['b'] - o['a'] <= 40]
        seqs = [(o, canon(o['rec'], o['a'], o['b'])) for o in os_]
        seqs = [(o, q) for o, q in seqs if q is not None and np.isfinite(q[:, [5, 6, 11, 12]]).mean() > .8]
        for o, q in seqs:
            feats.append(np.nan_to_num(np.r_[q[[0, 8, 15], :, :][:, [0, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]].ravel()])); labels.append(s)
        pick = rng.choice(len(seqs), min(40, len(seqs)), replace=False)
        stack = [np.round(seqs[i][1], 2) for i in pick]
        med = np.nanmedian(np.array([q for _, q in seqs]), 0)
        per = [use[f].get(s, 0) / tot[f] for f in fights]
        e = [v for f, v in zip(fights, per) if F.windows()[f]['date'] < '2016']; l = [v for f, v in zip(fights, per) if F.windows()[f]['date'] >= '2016']
        tt = ttest_ind(l, e, equal_var=False); ps.append(float(tt.pvalue))
        # three clips: typical length, spread over the career
        dur = np.array([o['b'] - o['a'] for o, _ in seqs]); md = np.median(dur)
        # ten clips, placed by the map: the four occurrences nearest the syllable's centre (its most typical
        # movement), then six spread over its region by farthest-point sampling, so a circle almost anywhere has video
        placed = [(o, EMB[f"{o['rec']}:{o['a']}"]) for o, _ in seqs if f"{o['rec']}:{o['a']}" in EMB]
        xy = np.array([p_ for _, p_ in placed]); cen = np.median(xy, 0)
        order_ = np.argsort(np.linalg.norm(xy - cen, axis=1))
        pick_i = list(order_[:4])
        while len(pick_i) < min(10, len(placed)):
            dmin = np.min(np.linalg.norm(xy[:, None] - xy[pick_i][None], axis=2), axis=1)
            dmin[np.linalg.norm(xy - cen, axis=1) > np.percentile(np.linalg.norm(xy - cen, axis=1), 85)] = -1   # not the stragglers
            pick_i.append(int(np.argmax(dmin)))
        chosen = [placed[i][0] for i in pick_i]
        clips = []
        for j, o in enumerate(chosen):
            p = OUT / 'clips' / f'syl{s}_{j}.mp4'; render_clip(o, p)
            r = recs[o['rec']]; clips.append({'file': f'clips/{p.name}', 'label': f"{r['date'][:4]} · vs {F.windows()[r['fight']]['opponent']}", 'key': f"{o['rec']}:{o['a']}"})
        out['syllables'].append({'id': s, 'name': NAMES_BY_ID[s], 'desc': DESC[s], 'share': share, 'n': len(os_),
                                 'median_s': float(md / 30), 'stack': [x.tolist() for x in stack], 'median': np.round(med, 2).tolist(),
                                 'per_fight': per, 'early': float(np.mean(e)), 'late': float(np.mean(l)), 'p': float(tt.pvalue),
                                 'effect_sd': float((np.mean(l) - np.mean(e)) / (np.std(per, ddof=1) or 1)), 'clips': clips})
        print(s, NAMES_BY_ID[s], f'{share:.1%}', f'{np.mean(e):.3f} -> {np.mean(l):.3f} p {tt.pvalue:.3f}', flush=True)
    ps = np.array(ps); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r_, i in enumerate(order): run = max(run, min(1, (m - r_) * ps[i])); adj[i] = run
    for x, a in zip(out['syllables'], adj):
        x['holm_p'] = float(a); x['reported'] = bool(a < .05 and abs(x['effect_sd']) >= .4)
    emb = json.loads((OUT / 'embed.json').read_text())
    clip_of = {c['key']: c for x in out['syllables'] for c in x['clips']}
    tc = json.loads((HERE / 'results' / 'stance' / 'findings.json').read_text())['torso_cm']
    pts, inst = [], []
    for r in emb['points']:
        if r['syl'] not in keep: continue
        q = canon(r['rec'], r['a'], r['b'], n=12)
        if q is None: continue
        hip = np.nanmean(q[:, [11, 12]], axis=1)
        dur = (r['b'] - r['a']) / recs[r['rec']]['rate']
        speed = float(np.nansum(np.linalg.norm(np.diff(hip, axis=0), axis=1)) * tc / dur) if dur > 0 else np.nan
        rec = recs[r['rec']]; key = f"{r['rec']}:{r['a']}"
        pts.append([r['x'], r['y'], r['syl'], int(rec['date'][:4]), round(speed, 1) if np.isfinite(speed) else None,
                    clip_of[key]['file'] if key in clip_of else None, f"{rec['date'][:4]} · vs {F.windows()[rec['fight']]['opponent']}"])
        inst.append(np.nan_to_num(np.round(q[:, COCO] * 100), nan=-9999).astype(int).ravel().tolist())
    out['map'] = pts
    out['map_knn'] = emb['knn_agreement']; out['map_chance'] = emb['chance']
    (OUT / 'syllables_inst.json').write_text(json.dumps(inst, separators=(',', ':')))
    for x in out['syllables']:
        x['stack'] = json.loads(json.dumps(x['stack']).replace('NaN', 'null'))
    (OUT / 'syllables_site.json').write_text(json.dumps(out, separators=(',', ':')).replace('NaN', 'null'))
    print('holm', [round(a, 3) for a in adj])


if __name__ == '__main__' and sys.argv[1] == 'build':
    build()
