"""Fit keypoint-MoSeq to McGregor's movement.

AR-only warm-up, then the full model; stickiness (kappa) set so the median syllable lasts about half a second.

    moseq-venv/bin/python moseq_fit.py [kappa_ar] [kappa_full]  -> $DATA/moseq/project, results.h5, syllables.json
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import keypoint_moseq as kpms

DATA = Path(os.environ.get('MMA_WORK', 'work') + '/mcgregor/moseq')
PROJ = DATA / 'project'
idx = json.loads((DATA / 'index.json').read_text())
names = idx['bodyparts']
raw = np.load(DATA / 'data.npz')
coords = {r['name']: np.nan_to_num(raw[f"c__{r['name']}"]) for r in idx['recordings']}
confs = {r['name']: np.nan_to_num(raw[f"w__{r['name']}"]) for r in idx['recordings']}
k_ar = float(sys.argv[1]) if len(sys.argv) > 1 else 1e5
k_full = float(sys.argv[2]) if len(sys.argv) > 2 else 1e4

kpms.setup_project(str(PROJ), overwrite=True)
skel = [['l_shoulder', 'r_shoulder'], ['l_shoulder', 'l_elbow'], ['l_elbow', 'l_wrist'], ['r_shoulder', 'r_elbow'], ['r_elbow', 'r_wrist'],
        ['l_shoulder', 'l_hip'], ['r_shoulder', 'r_hip'], ['l_hip', 'r_hip'], ['l_hip', 'l_knee'], ['l_knee', 'l_ankle'], ['r_hip', 'r_knee'],
        ['r_knee', 'r_ankle'], ['l_ankle', 'l_heel'], ['l_heel', 'l_toe'], ['r_ankle', 'r_heel'], ['r_heel', 'r_toe']]
kpms.update_config(str(PROJ), bodyparts=names, use_bodyparts=names, skeleton=skel, anterior_bodyparts=['nose'],
                   posterior_bodyparts=['l_hip', 'r_hip'], fps=30, outlier_scale_factor=6.0)
config = lambda: kpms.load_config(str(PROJ))
data, metadata = kpms.format_data(coords, confs, **config())
pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, str(PROJ))
cs = np.cumsum(pca.explained_variance_ratio_)
latent = int(np.searchsorted(cs, .9) + 1)
print('latent dims for 90 % variance:', latent, flush=True)
kpms.update_config(str(PROJ), latent_dim=latent)
model = kpms.init_model(data, pca=pca, **config())
model = kpms.update_hypparams(model, kappa=k_ar)
model, name = kpms.fit_model(model, data, metadata, str(PROJ), ar_only=True, num_iters=50)
model, data, metadata, it = kpms.load_checkpoint(str(PROJ), name, iteration=50)
model = kpms.update_hypparams(model, kappa=k_full)
model = kpms.fit_model(model, data, metadata, str(PROJ), name, ar_only=False, start_iter=it, num_iters=it + 250)[0]
kpms.reindex_syllables_in_checkpoint(str(PROJ), name)
model, data, metadata, it = kpms.load_checkpoint(str(PROJ), name)
res = kpms.extract_results(model, metadata, str(PROJ), name)
out = {}
durs = []
for rec, r in res.items():
    s = np.asarray(r['syllable']); out[rec] = s.tolist()
    ch = np.flatnonzero(np.diff(s)) + 1
    durs += np.diff(np.r_[0, ch, len(s)]).tolist()
(DATA / 'syllables.json').write_text(json.dumps({'model': name, 'kappa': [k_ar, k_full], 'latent_dim': latent, 'syllables': out}))
print('median syllable duration (s):', np.median(durs) / 30, 'n syllables used:', len(set(np.concatenate([np.asarray(v) for v in out.values()]))), flush=True)
