"""The study as one page: results/report.html. Every number comes from results/results.json and the per-frame
files; the prose lives in TEXT below and is written after reading the results.

    python report.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import figures as G      # noqa: E402
import stance as S       # noqa: E402

R = json.loads((HERE / 'results' / 'results.json').read_text())
W = json.loads((HERE / 'windows.json').read_text())
TEXT = json.loads((HERE / 'report_text.json').read_text())

HYP = [  # key in tests, quantity, axis label, unit, number format, plain-language "higher means"
    ('H1 stance width', 'stance_width', 'Stance width', ' torso lengths', '{:.2f}', 'feet wider apart'),
    ('H2 on the toes (rear heel up)', 'rear_heel_up', 'Rear-heel lift', ' foot lengths', '{:.2f}', 'rear heel higher'),
    ('H3 bounce', 'bounce', 'Bounce (1.5–4 Hz hip motion)', ' torso lengths', '{:.3f}', 'more bounce'),
    ('H4 guard height', 'guard', 'Guard height', ' torso lengths', '{:.2f}', 'hands higher'),
    ('H5 kicks per minute', 'kicks_per_min', 'Kicks per standing minute', '', '{:.1f}', 'more kicks'),
    ('H6 stance switches per minute', 'switches_per_min', 'Stance switches per standing minute', '', '{:.1f}', 'more switching'),
    ('H7 range', 'range', 'Range (hip to hip)', ' torso lengths', '{:.2f}', 'farther apart'),
]


def verdict(t: dict) -> tuple[str, str]:
    """Primary test: McGregor minus opponent, trend over fight date, Holm-corrected permutation p."""
    rho, p, hp, sign = t['spearman_rho'], t['perm_p'], t['holm_p'], t['predicted_sign']
    if rho is None or not np.isfinite(rho):
        return 'untested', 'Not enough fights'
    same = np.sign(rho) == sign
    if hp < .05:
        return ('yes', 'Supported') if same else ('no', 'Reversed')
    if p < .05:
        return ('lean', 'Suggestive') if same else ('no', 'Reversed, weakly')
    return ('flat', 'No clear change')


def fmt_ci(ci, f='{:+.2f}'):
    return '–'.join(f.format(c) for c in ci) if ci and np.isfinite(ci).all() else 'n/a'


SHORT_NAME = {'Khabib Nurmagomedov': 'Khabib'}


def fingerprints(fights) -> str:
    """McGregor's average standing pose per fight, in order, same scale."""
    cells = []
    for f in fights:
        p = HERE / 'results' / 'perframe' / f"{f['fight']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        pts = S.canonical(d['mcg_kps'].astype(np.float32), d['opp_kps'].astype(np.float32), d['shot'], d['t'])
        if len(pts) < 30:
            continue
        avg = S.average(pts)
        svg = (f'<svg viewBox="-80 -150 160 250" class="fig-skel" role="img" aria-label="Average stance, {f["date"][:4]} vs {G.esc(f["opponent"])}">'
               f'<line x1="-70" x2="70" y1="{np.nanmax(avg[[15, 16], 1]) * 62 + 6:.1f}" y2="{np.nanmax(avg[[15, 16], 1]) * 62 + 6:.1f}" class="ground"/>'
               + G.skeleton(avg, 'mcg', 62, 0, 0) + '</svg>')
        cells.append(f'<figure class="print"><div class="print-art">{svg}</div><figcaption><b>{f["date"][:4]}</b>'
                     f'<span>{G.esc(SHORT_NAME.get(f["opponent"], f["opponent"].split()[-1]))}</span></figcaption></figure>')
    return ''.join(cells)


def era_overlay(fights, split) -> str:
    """Early vs late average stance on top of each other."""
    groups = {'early': [], 'late': []}
    for f in fights:
        p = HERE / 'results' / 'perframe' / f"{f['fight']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        pts = S.canonical(d['mcg_kps'].astype(np.float32), d['opp_kps'].astype(np.float32), d['shot'], d['t'])
        # weight fights equally: sample up to 400 frames each
        if len(pts) > 400:
            pts = pts[np.random.default_rng(0).choice(len(pts), 400, replace=False)]
        groups['late' if date.fromisoformat(f['date']) >= split else 'early'].append(pts)
    if not groups['early'] or not groups['late']:
        return ''
    e, l = S.average(np.concatenate(groups['early'])), S.average(np.concatenate(groups['late']))
    return ('<svg viewBox="-110 -170 220 290" class="fig-overlay" role="img" aria-label="Average stance before and after March 2016">'
            + G.skeleton(e, 'early', 80, 0, 0, w=3, dot=3.4) + G.skeleton(l, 'mcg', 80, 0, 0, w=3, dot=3.4) + '</svg>')


# Opponents whose fighting stance is public knowledge (only those we are sure of), to check the instrument.
KNOWN = {'Max Holloway': 'orthodox', 'Chad Mendes': 'orthodox', 'Eddie Alvarez': 'orthodox', 'Dennis Siver': 'orthodox',
         'Khabib Nurmagomedov': 'orthodox', 'Donald Cerrone': 'orthodox', 'Dustin Poirier': 'southpaw', 'Nate Diaz': 'southpaw'}


def instrument_rows(fights) -> str:
    rows = []
    for f in fights:
        st = KNOWN.get(f['opponent'])
        v = f.get('opp', {}).get('orthodox_share')
        if not st or v is None or not np.isfinite(v):
            continue
        right = v if st == 'orthodox' else 1 - v
        rows.append(f'<tr><td>{f["date"][:4]}</td><td>{G.esc(f["opponent"])}</td><td>{st}</td>'
                    f'<td class="num">{v:.0%}</td><td class="num">{right:.0%}</td></tr>')
    mc = [1 - f['mcg']['orthodox_share'] for f in fights if f.get('mcg') and np.isfinite(f['mcg'].get('orthodox_share', np.nan))]
    rows.append(f'<tr><td>all</td><td>Conor McGregor</td><td>southpaw</td><td class="num">{1 - np.median(mc):.0%}</td><td class="num">{np.median(mc):.0%}</td></tr>')
    return ''.join(rows)


ST = json.loads((HERE / 'results' / 'stance' / 'stance.json').read_text())
CLAIMS = [  # the popular story, claim by claim -> the measure that tests it, the direction it predicts
    ('A wide karate stance that narrowed', 'width', -1, 'Ankle to ankle, torso lengths'),
    ('Light on his toes, then flat-footed', 'heel', -1, 'Rear heel lift'),
    ('Bouncing, then still', 'bounce', -1, 'Hip motion at 1.5–4 Hz'),
    ('Moving less', 'step_speed', -1, 'Hip speed, torso lengths / s'),
    ('Upright, then leaning forward', 'lean', +1, 'Torso tilt toward the opponent, °'),
    ('Head pushed out over the front', 'head_fwd', +1, 'Head ahead of hips'),
    ('Shoulders hunched, chin down', 'hunch', -1, 'Nose above the shoulder line'),
    ('Weight onto the front foot', 'weight', +1, 'Hips between the feet: 0 rear, 1 lead'),
    ('A deeper crouch', 'crouch', -1, 'Hip height above the ankles'),
]
STANCE_CHARTS = [('width', 'Stance width', '', '{:.2f}', 'wider'), ('lean', 'Torso lean', '°', '{:.1f}', 'leaning toward the opponent'),
                 ('head_fwd', 'Head ahead of hips', '', '{:.2f}', 'head further forward'), ('weight', 'Weight over the lead foot', '', '{:.2f}', 'weight further forward'),
                 ('hunch', 'Neck height (shoulder hunch)', '', '{:.2f}', 'longer neck, less hunched'), ('crouch', 'Hip height (crouch)', '', '{:.2f}', 'standing taller'),
                 ('lead_knee', 'Lead knee angle', '°', '{:.0f}', 'straighter lead leg'), ('rear_knee', 'Rear knee angle', '°', '{:.0f}', 'straighter rear leg'),
                 ('heel', 'Heel lift', '', '{:.2f}', 'more on the toes'), ('bounce', 'Bounce', '', '{:.3f}', 'more bounce'),
                 ('step_speed', 'Footwork speed', '', '{:.2f}', 'moving more'), ('guard', 'Guard height', '', '{:.2f}', 'hands higher')]


def stance_verdict(k, sign):
    t = ST['tests'][k]; m, o, d = t['mcg'], t['opp'], t['diff']; sd = t.get('mcg_side', {})
    moved = m['p'] < .05 and np.sign(m['rho']) == sign
    shared = o['p'] < .1 and np.sign(o['rho']) == np.sign(m['rho'])
    survives_view = sd.get('p', 1) < .05 and np.sign(sd.get('rho', 0)) == np.sign(m['rho'])
    if moved and survives_view and not shared: return 'lean', 'Changed in him'
    if moved and shared: return 'flat', 'Changed, but so did they'
    if m['p'] < .05 and np.sign(m['rho']) != sign: return 'no', 'Opposite way'
    return 'flat', 'No change in him'


def stance_section() -> str:
    rows = []
    for claim, k, sign, unit in CLAIMS:
        t = ST['tests'][k]; cls, word = stance_verdict(k, sign); sd = t.get('mcg_side', {})
        side = f"{sd['rho']:+.2f}<small>p {sd['p']:.3f}</small>" if sd else 'n/a'
        f = '{:.3f}' if k == 'bounce' else '{:.1f}' if k in ('lean',) else '{:.2f}'
        rows.append(f'<tr><th scope="row">{G.esc(claim)}<small>{G.esc(unit)}</small></th>'
                    f'<td class="num">{f.format(t["mcg"]["early"])} → {f.format(t["mcg"]["late"])}</td>'
                    f'<td class="num">{t["mcg"]["rho"]:+.2f}<small>p {t["mcg"]["p"]:.3f}</small></td>'
                    f'<td class="num">{side}</td>'
                    f'<td class="num">{t["opp"]["rho"]:+.2f}<small>p {t["opp"]["p"]:.3f}</small></td>'
                    f'<td><span class="chip {cls}">{word}</span></td></tr>')
    fights = [r for r in ST['fights'] if r['seconds'] >= 30]
    charts = []
    for k, lab, unit, fmt, up in STANCE_CHARTS:
        t = ST['tests'][k]['mcg']
        charts.append(f'<article class="hyp small"><header><h3>{G.esc(lab)}</h3><span class="chip {"lean" if t["p"] < .05 else "flat"}">ρ {t["rho"]:+.2f}</span></header>'
                      f'<div class="chart-wrap">{G.timeline(fights, k, lab, unit, "2016-03-05", fmt=fmt, height=200)}</div>'
                      f'<p class="axis-note">Up = {G.esc(up)} · p {t["p"]:.3f}, him alone</p></article>')
    return ''.join(rows), ''.join(charts)


FEET = json.loads((HERE / 'results' / 'stance' / 'feet.json').read_text())


def feet_rows() -> str:
    lab = {'heel_lift': ('Heel lift', 'toe minus heel height, per foot length', '{:.2f}'),
           'heel_up_share': ('Heel clearly up', 'share of foot-frames', '{:.0%}'),
           'ankle_bounce': ('Ankle bounce', 'vertical ankle motion at 1.5–4 Hz', '{:.3f}'),
           'planted_share': ('Both feet planted', 'share of standing time', '{:.1%}')}
    rows = []
    for k, (l, u, f) in lab.items():
        t = FEET['tests'][k]; mo = t['mcg_more_in']
        rows.append(f'<tr><th scope="row">{l}<small>{u}</small></th><td class="num">{f.format(t["mcg"]["early"])} → {f.format(t["mcg"]["late"])}</td>'
                    f'<td class="num">{t["mcg"]["rho"]:+.2f}<small>p {t["mcg"]["p"]:.3f}</small></td>'
                    f'<td class="num">{f.format(t["opp"]["early"])} → {f.format(t["opp"]["late"])}</td>'
                    f'<td class="num">{mo[0]} of {mo[1]}</td></tr>')
    return ''.join(rows)


PR = json.loads((HERE / 'results' / 'left' / 'pressure.json').read_text())


def pressure_section() -> str:
    items = [('first_share', 'Strikes that started an exchange', '{:.0%}', 'he went first more often'),
             ('counter_share', 'Strikes that answered one (counters)', '{:.0%}', 'more counter-punching'),
             ('moving_in_share', 'Strikes thrown moving forward', '{:.0%}', 'more strikes on the way in'),
             ('backing_share', 'Strikes thrown moving backward', '{:.0%}', 'more strikes on the retreat'),
             ('advancing', 'Time spent moving forward', '{:.0%}', 'more time advancing'),
             ('head_share', 'Hand strikes at head height', '{:.0%}', 'more head-hunting')]
    out = []
    for k, lab, fmt, up in items:
        t = PR['tests'][k]
        out.append(f'<article class="hyp small"><header><h3>{G.esc(lab)}</h3><span class="chip {"lean" if t["mcg"]["p"] < .05 else "flat"}">ρ {t["mcg"]["rho"]:+.2f}</span></header>'
                   f'<p class="read mono">him {fmt.format(t["mcg"]["early"])} → {fmt.format(t["mcg"]["late"])} · opponents {fmt.format(t["opp"]["early"])} → {fmt.format(t["opp"]["late"])} · his higher in {t["mcg_more_in"][0]} of {t["mcg_more_in"][1]}</p>'
                   f'<div class="chart-wrap">{G.timeline(PR["fights"], k, lab, "", "2016-03-05", fmt=fmt, height=190)}</div>'
                   f'<p class="axis-note">Up = {G.esc(up)} · p {t["mcg"]["p"]:.2f}, him alone</p></article>')
    return ''.join(out)


def page() -> str:
    tests = R['tests']['primary']
    fights = [f for f in R['fights'] if f.get('mcg')]
    split = date.fromisoformat(R['split'])
    used = [f for f in fights if f['standing_seconds'] >= R['min_standing']]
    total_live = sum(f.get('live_seconds', 0) for f in R['fights'])
    total_standing = sum(f['standing_seconds'] for f in R['fights'])
    rows, charts = [], []
    for name, key, label, unit, f_, hi in HYP:
        t = tests.get(name, {}).get('diff')
        if not t:
            continue
        cls, word = verdict(t)
        tm = tests[name]['mcg']; to = tests[name]['opp']
        rows.append(
            f'<tr><th scope="row">{G.esc(label)}<small>{G.esc(TEXT["predictions"][key])}</small></th>'
            f'<td><span class="chip {cls}">{word}</span></td>'
            f'<td class="num">{t["spearman_rho"]:+.2f}</td><td class="num">{t["perm_p"]:.3f}</td><td class="num">{t["holm_p"]:.3f}</td>'
            f'<td class="num">{f_.format(t["late_minus_early"]) if np.isfinite(t["late_minus_early"]) else "n/a"}<small>{fmt_ci(t["late_minus_early_ci"])}</small></td>'
            f'<td class="num">{tm["spearman_rho"]:+.2f}</td><td class="num">{to["spearman_rho"]:+.2f}</td></tr>')
        charts.append(
            f'<article class="hyp" id="{key}"><header><h3>{G.esc(label)}</h3><span class="chip {cls}">{word}</span></header>'
            f'<p class="read">{G.esc(TEXT["readings"].get(key, ""))}</p>'
            f'<div class="chart-wrap">{G.timeline(used, key, label, unit, R["split"], fmt=f_)}</div>'
            f'<p class="axis-note">Up = {hi}. Each dot is one fight; whiskers are 95 % block-bootstrap intervals.</p></article>')
    fight_rows = ''.join(
        f'<tr><td>{f["date"]}</td><td>{G.esc(f["opponent"])}</td><td>{G.esc(W[f["fight"]].get("event", ""))}</td>'
        f'<td class="num">{f.get("live_seconds", 0):.0f}</td><td class="num">{f["standing_seconds"]:.0f}</td>'
        f'<td>{"in tests" if f["standing_seconds"] >= R["min_standing"] else "shown only"}{"" if W[f["fight"]].get("official", True) else ", fan upload"}</td></tr>'
        for f in R['fights'])
    excluded = ''.join(f'<li><b>{k[:4]} {G.esc(v["opponent"])}</b>: {G.esc(v["exclude"])}</li>' for k, v in W.items() if 'exclude' in v)
    surprises = ''.join(f'<article class="surprise"><h3>{G.esc(s["title"])}</h3><p>{G.esc(s["body"])}</p>{s.get("figure", "")}</article>' for s in TEXT['surprises'])
    prereg = (HERE / 'PREREGISTRATION.md').read_text().split('## Changes after registration')[1]
    prereg_items = ''.join(f'<li>{G.esc(x.strip()[3:].replace("**", ""))}</li>' for x in prereg.strip().split('\n') if x[:3].strip().rstrip('.').isdigit() or x.startswith(tuple(f'{i}.' for i in range(1, 10))))
    fill = dict(
        lede=TEXT['lede'], answer=TEXT['answer'], n_fights=str(len(R['fights'])), n_used=str(len(used)),
        live_min=f'{total_live / 60:.0f}', standing_min=f'{total_standing / 60:.0f}', prints=fingerprints(fights),
        overlay=era_overlay(used, split), overlay_note=G.esc(TEXT['overlay_note']), rows=''.join(rows),
        charts=''.join(charts), surprises=surprises, fight_rows=fight_rows, excluded=excluded,
        method=TEXT['method'], fool=TEXT['fool'], prereg=prereg_items, closing=TEXT['closing'],
        instrument=instrument_rows(fights), instrument_note=TEXT['instrument_note'],
        stance_rows=stance_section()[0], stance_charts=stance_section()[1], stance_read=TEXT['stance_read'], feet_rows=feet_rows(), feet_read=TEXT['feet_read'], pressure=pressure_section(),
        first_mcg=f"{PR['tests']['first_share']['mcg']['early']:.0%} → {PR['tests']['first_share']['mcg']['late']:.0%}", counter_mcg=f"{PR['tests']['counter_share']['mcg']['early']:.0%} → {PR['tests']['counter_share']['mcg']['late']:.0%}")
    out = TEMPLATE
    for k, v in fill.items():
        out = out.replace('{{' + k + '}}', v)
    return out


TEMPLATE = (HERE / 'report_template.html').read_text()

if __name__ == '__main__':
    out = HERE / 'results' / 'report.html'
    out.write_text(page())
    print('wrote', out)
