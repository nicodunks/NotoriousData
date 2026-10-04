"""The syllable map: every occurrence of the kept syllables placed by how it moves.

Features per occurrence, from the fitted keypoint-MoSeq model (results.h5): mean latent state, mean absolute and
mean signed latent velocity, mean and mean absolute body-centre velocity (torso lengths per frame, x3), and log
duration; standardised, then t-SNE (perplexity 40). Run in the MoSeq venv (h5py):

    MPLBACKEND=Agg ../../../../moseq-venv/bin/python syl_embed.py -> results/syllables/embed.json
"""
import os
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).parent
D = Path(os.environ.get('MMA_WORK', 'work') + '/mcgregor/moseq')
RES = sorted((D / 'project').glob('2026_*'))[-1] / 'results.h5'
KEEP = {2, 3, 0, 1, 5, 7, 4, 6}
f = h5py.File(RES, 'r'); raw = np.load(D / 'data.npz')
X, rows = [], []
for name in f.keys():
    s = np.asarray(f[name]['syllable']); z = np.asarray(f[name]['latent_state']); c = np.asarray(f[name]['centroid'])
    xy = raw[f'c__{name}'].astype(float); tor = np.nanmedian(np.linalg.norm(xy[:, [1, 2]].mean(1) - xy[:, [7, 8]].mean(1), axis=1))
    ch = np.r_[0, np.flatnonzero(np.diff(s)) + 1, len(s)]
    dz = np.gradient(z, axis=0); dc = np.gradient(c, axis=0) / tor
    for a, b in zip(ch[:-1], ch[1:]):
        if s[a] not in KEEP or b - a < 3: continue
        X.append(np.r_[z[a:b].mean(0), np.abs(dz[a:b]).mean(0), dz[a:b].mean(0), dc[a:b].mean(0) * 3, np.abs(dc[a:b]).mean(0) * 3, np.log(b - a)])
        rows.append({'rec': name, 'a': int(a), 'b': int(b), 'syl': int(s[a])})
X = np.array(X); X = (X - X.mean(0)) / (X.std(0) + 1e-9); L = np.array([r['syl'] for r in rows])
E = TSNE(2, perplexity=40, random_state=0, init='pca').fit_transform(X)
E = (E - E.min(0)) / (E.max(0) - E.min(0))
for r, (x, y) in zip(rows, E): r['x'], r['y'] = round(float(x), 4), round(float(y), 4)
acc = float(cross_val_score(KNeighborsClassifier(10), E, L, cv=5).mean())
(HERE / 'results' / 'syllables' / 'embed.json').write_text(json.dumps({'knn_agreement': acc, 'chance': float(np.max(np.bincount(L)) / len(L)), 'points': rows}))
print(len(rows), 'occurrences; 10-nearest-neighbour syllable agreement on the map', round(acc, 3))
