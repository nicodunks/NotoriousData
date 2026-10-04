"""The left-hand claims, tested under the same rule as the stance: one number per fight, McGregor against his
opponent in that fight, Wilcoxon signed-rank over fights, Holm across every claim on the tab; and, for change over
the career, early (2012-15) vs late (2016-21) by Welch's t-test in the same Holm family.

    python leftfindings.py -> results/left/findings.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind, wilcoxon

HERE = Path(__file__).parent
L = HERE / 'results' / 'left'
summ = json.loads((L / 'summary.json').read_text())
sl = json.loads((L / 'slips.json').read_text())
lead = json.loads((L / 'leadhand.json').read_text())
sel = json.loads((L / 'selection.json').read_text())
torso_cm = json.loads((HERE / 'results' / 'stance' / 'findings.json').read_text())['torso_cm']


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * ps[i])); out[i] = run
    return out


pairs = {}
pairs['left_rate'] = [(f['fight'], f['date'], f['left_rate'], f['opp_rate']) for f in summ['fights'] if f['upright_min'] and f['upright_min'] >= 1]
pairs['rear_share'] = [(r['fight'], r['date'], r['mcg']['rear_share'], r['opp']['rear_share']) for r in sel['fights']]
pairs['jab_rate'] = [(r['fight'], r['date'], r['mcg'].get('lead straight', 0), r['opp'].get('lead straight', 0)) for r in sel['fights']]
by = {}
for r in sl['events']:
    by.setdefault((r['fight'], r['who']), []).append(r)
pairs['big_head_move'] = [(f, by[(f, 'mcg')][0]['date'], np.mean([e['head_move'] >= .4 for e in by[(f, 'mcg')]]), np.mean([e['head_move'] >= .4 for e in by[(f, 'opp')]]))
                          for f in sorted({k[0] for k in by}) if (f, 'mcg') in by and (f, 'opp') in by]
recs = lead['records']; byl = {}
for r in recs: byl.setdefault((r['fight'], r['who']), []).append(r)
pairs['jab_before'] = [(f, byl[(f, 'mcg')][0]['date'], np.mean([e['lead_jabbed'] for e in byl[(f, 'mcg')]]), np.mean([e['lead_jabbed'] for e in byl[(f, 'opp')]]))
                       for f in sorted({k[0] for k in byl}) if (f, 'mcg') in byl and (f, 'opp') in byl]
recs_by_f = {}
for r in recs: recs_by_f.setdefault((r['fight'], r['who']), []).append(r['lead_motion'])
pairs['lead_motion'] = [(f, byl[(f, 'mcg')][0]['date'], float(np.median(recs_by_f[(f, 'mcg')])), float(np.median(recs_by_f[(f, 'opp')])))
                        for f in sorted({k[0] for k in byl}) if (f, 'mcg') in byl and (f, 'opp') in byl]
pres = json.loads((L / 'pressure.json').read_text())
# Families, one per question on the tab; Holm within each.
#   sig: what set his left apart from the rear hand thrown back at him (paired by fight)
#   era: what changed in how he used it, 2012-15 vs 2016-21 (Welch on per-fight values, McGregor alone)
#   cnt: the two counters against each other (Mann-Whitney on punches; one test)
out, fam = {}, {'sig': [], 'era': [], 'cnt': []}
for k, rows in pairs.items():
    d = np.array([m - o for _, _, m, o in rows])
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    out[k] = {'kind': 'vs opponents', 'family': 'sig', 'n': len(rows), 'his_higher_in': int((d > 0).sum()), 'mcg': float(np.median([m for _, _, m, _ in rows])),
              'opp': float(np.median([o for _, _, _, o in rows])), 'median_diff': float(np.median(d)), 'p': p,
              'effect_sd': float(np.median(d) / (np.std(d, ddof=1) or 1)), 'per_fight': rows}
    fam['sig'].append(k)
mc = [f for f in summ['fights'] if f['upright_min'] and f['upright_min'] >= 1]
slip_pf = {r['fight']: r['mcg'].get('slip_share') for r in sl['per_fight']}
era = {'left_rate': [(f['date'], f['left_rate']) for f in mc],
       'left_hook': [(r['date'], r['mcg'].get('rear hook', 0)) for r in sel['fights']],
       'jab_rate': [(r['date'], r['mcg'].get('lead straight', 0)) for r in sel['fights']],
       'counter_share': [(f['date'], f['counter_share']) for f in mc if f['counter_share'] is not None and f['lefts'] >= 4],
       'moved_first': [(f['date'], slip_pf[f['fight']]) for f in mc if slip_pf.get(f['fight']) is not None and f['lefts'] >= 4],
       'first_share': [(r['date'], r['mcg']['first_share']) for r in pres['fights']],
       'strike_counter_share': [(r['date'], r['mcg']['counter_share']) for r in pres['fights']],
       'head_share': [(r['date'], r['mcg']['head_share']) for r in pres['fights']],
       'backing_share': [(r['date'], r['mcg']['backing_share']) for r in pres['fights']]}
for k, rows in era.items():
    rows = [(d_, v) for d_, v in rows if v is not None and np.isfinite(v)]
    e = [v for d_, v in rows if d_ < '2016']; l = [v for d_, v in rows if d_ >= '2016']
    tt = ttest_ind(l, e, equal_var=False)
    sd = np.std([v for _, v in rows], ddof=1)
    out[f'era_{k}'] = {'kind': 'early vs late', 'family': 'era', 'early': float(np.mean(e)), 'late': float(np.mean(l)), 'p': float(tt.pvalue),
                       'effect_sd': float((np.mean(l) - np.mean(e)) / sd), 'n': [len(e), len(l)], 'per_fight': rows}
    fam['era'].append(f'era_{k}')
# the two counters: how long after the head's furthest point the left starts
from scipy.stats import mannwhitneyu
pc = [r['lead_ms'] for r in sl['events'] if r['who'] == 'mcg' and r['setup'] == 'pull counter' and r.get('lead_ms') is not None]
sc = [r['lead_ms'] for r in sl['events'] if r['who'] == 'mcg' and r['setup'] == 'slip counter' and r.get('lead_ms') is not None]
pool = np.r_[pc, sc]
out['counter_timing'] = {'kind': 'pull vs slip', 'family': 'cnt', 'pull_ms': float(np.median(pc)), 'slip_ms': float(np.median(sc)), 'n': [len(pc), len(sc)],
                         'p': float(mannwhitneyu(pc, sc).pvalue), 'effect_sd': float((np.median(pc) - np.median(sc)) / (np.std(pool, ddof=1) or 1)),
                         'pull': pc, 'slip': sc}
fam['cnt'].append('counter_timing')
keys = [k for f in fam.values() for k in f]
for f, ks in fam.items():
    for k, ph in zip(ks, holm([out[k]['p'] for k in ks])):
        out[k]['holm_p'] = float(ph); out[k]['reported'] = bool(ph < .05 and abs(out[k]['effect_sd']) >= .4)
# descriptive: the set-up mix, every left, his and theirs
mix = {}
for who in ('mcg', 'opp'):
    rs = [r for r in recs if r['who'] == who]
    mix[who] = {s: sum(r['setup'] == s for r in rs) for s in ['straight lead', 'caught coming in', 'behind the paw', 'counter, head still', 'slip counter', 'pull counter']}
    mix[who]['n'] = len(rs)
for era_, test in (('early', lambda d: d < '2016'), ('late', lambda d: d >= '2016')):
    rs = [r for r in recs if r['who'] == 'mcg' and test(r['date'])]
    mix[era_] = {s: sum(r['setup'] == s for r in rs) for s in mix['mcg'] if s != 'n'}; mix[era_]['n'] = len(rs)
out['_mix'] = mix
out['_setups'] = lead['setups']
out['_posted'] = {w: float(np.mean([r['posted'] for r in recs if r['who'] == w])) for w in ('mcg', 'opp')}
out['_baseline_big_move'] = sl['baseline']['big_move_share']
out['_torso_cm'] = torso_cm
(L / 'findings.json').write_text(json.dumps(out, indent=1, default=float))
for k in keys:
    v = out[k]
    if v['kind'] == 'vs opponents':
        print(f"{k:15s} him {v['mcg']:.2f} opp {v['opp']:.2f} higher {v['his_higher_in']}/{v['n']} p {v['p']:.4f} holm {v['holm_p']:.3f} d {v['effect_sd']:+.2f} {'REPORTED' if v['reported'] else ''}")
    elif v['kind'] == 'pull vs slip':
        print(f"{k:15s} pull {v['pull_ms']:.0f} ms slip {v['slip_ms']:.0f} ms n {v['n']} p {v['p']:.4f} d {v['effect_sd']:+.2f} {'REPORTED' if v['reported'] else ''}")
    else:
        print(f"{k:15s} {v['early']:.2f} -> {v['late']:.2f} p {v['p']:.3f} holm {v['holm_p']:.3f} d {v['effect_sd']:+.2f} {'REPORTED' if v['reported'] else ''}")
