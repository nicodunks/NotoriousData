"""The left-hand page: results/left/report.html, every number from results/left/summary.json and events.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import figures as G      # noqa: E402
import frames as F       # noqa: E402
import stance as S       # noqa: E402

SUM = json.loads((HERE / 'results' / 'left' / 'summary.json').read_text())
EVS = json.loads((HERE / 'results' / 'left' / 'events.json').read_text())
W = json.loads((HERE / 'windows.json').read_text())


def average_left() -> str:
    """McGregor's straight left averaged over every detected one: 5 moments, onset-150 ms to peak+100 ms."""
    moments = [('−150 ms', -.15, 'on'), ('onset', 0, 'on'), ('halfway', .5, 'mid'), ('full extension', 0, 'pk'), ('+100 ms', .1, 'pk')]
    stacks = {m[0]: [] for m in moments}
    cache = {}
    for e in EVS:
        if e['who'] != 'mcg' or e['kind'] != 'straight':
            continue
        f = e['fight']
        if f not in cache:
            d = F.load(f, keep='upright', suffix='_30')
            cache[f] = (d['t'], S.canonical_all(d['mcg'], d['opp'], d['shot'], d['t']))
        t, pts = cache[f]
        for name, off, base in moments:
            tt = e['t_onset'] + off if base == 'on' else e['t_peak'] + off if base == 'pk' else (e['t_onset'] + e['t_peak']) / 2
            i = int(np.argmin(np.abs(t - tt)))
            if abs(t[i] - tt) < .05:
                stacks[name].append(pts[i])
    cells = []
    for name, *_ in moments:
        if len(stacks[name]) < 10:
            continue
        avg = S.average(np.array(stacks[name]), min_seen=8)
        svg = (f'<svg viewBox="-110 -150 220 250" class="fig-skel" role="img" aria-label="Average left, {name}">'
               + G.skeleton(avg, 'mcg', 62, 0, 0, w=2.6, dot=3) + '</svg>')
        cells.append(f'<figure class="print"><div class="print-art">{svg}</div><figcaption><b>{name}</b><span>n={len(stacks[name])}</span></figcaption></figure>')
    return ''.join(cells)


def rate_chart() -> str:
    rows = [f for f in SUM['fights'] if f['upright_min'] and f['upright_min'] >= 1]
    width, row_h, L = 620, 26, 150
    hi = max(max(f['left_rate'], f['opp_rate']) for f in rows) * 1.1
    X = lambda v: L + v / hi * (width - L - 50)
    h = len(rows) * row_h + 36
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="Straight lefts per upright minute, per fight">']
    for v in range(0, int(hi) + 1):
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="0" y2="{h - 26}" class="grid"/><text x="{X(v):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{v}</text>')
    for k, f in enumerate(rows):
        y = k * row_h + 14
        out.append(f'<text x="{L - 10}" y="{y + 4}" class="tick" text-anchor="end">{f["date"][:4]} {G.esc(f["opponent"].split()[-1])}</text>')
        out.append(f'<line x1="{X(0)}" x2="{X(f["opp_rate"]):.1f}" y1="{y + 5}" y2="{y + 5}" class="bar opp"><title>{G.esc(f["opponent"])} rear straight: {f["opp_rate"]:.1f}/min</title></line>')
        out.append(f'<line x1="{X(0)}" x2="{X(f["left_rate"]):.1f}" y1="{y - 3}" y2="{y - 3}" class="bar mcg"><title>McGregor left: {f["left_rate"]:.1f}/min ({f["lefts"]} in {f["upright_min"]:.1f} min)</title></line>')
    out.append('</svg>')
    return ''.join(out)


def mechanics_rows() -> str:
    labels = {'peak_speed': ('Peak fist speed', 'torso lengths / s', '{:.1f}'), 'duration_ms': ('Onset to full extension', 'ms', '{:.0f}'),
              'reach': ('Fist ahead of own hips at full extension', 'torso lengths', '{:.2f}'),
              'hip_drive': ('Hips carried forward during the punch', 'torso lengths', '{:.2f}'),
              'lead_step': ('Lead foot stepped in', 'torso lengths', '{:.2f}'),
              'head_fwd_at_peak': ('Head ahead of hips at full extension', 'torso lengths', '{:.2f}'),
              'range_at_onset': ('Hip-to-hip distance when it starts', 'torso lengths', '{:.2f}'),
              'lead_high_at_peak': ('Other hand height at full extension', 'torso lengths vs shoulder', '{:.2f}')}
    rows = []
    for k, (lab, unit, f) in labels.items():
        v = SUM['features'].get(k)
        if not v: continue
        sig = v['ci'][0] > 0 or v['ci'][1] < 0
        rows.append(f'<tr><th scope="row">{lab}<small>{unit}</small></th><td class="num">{f.format(v["mcg"])}</td><td class="num">{f.format(v["opp"])}</td>'
                    f'<td class="num">{v["median_diff"]:+.2f}<small>{v["ci"][0]:+.2f} to {v["ci"][1]:+.2f}</small></td>'
                    f'<td class="num">{v["mcg_higher_in"]}/{v["n_fights"]}</td><td><span class="chip {"yes" if sig else "flat"}">{"differs" if sig else "same"}</span></td></tr>')
    return ''.join(rows)


def setup_chart() -> str:
    names = list(SUM['setups']['mcg'])
    width, row_h, L = 620, 34, 170
    h = len(names) * row_h + 34
    X = lambda v: L + v / .6 * (width - L - 60)
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="How the punch was set up, share of punches">']
    for v in (0, .2, .4, .6):
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="0" y2="{h - 26}" class="grid"/><text x="{X(v):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{v:.0%}</text>')
    for k, s in enumerate(names):
        y = k * row_h + 16
        out.append(f'<text x="{L - 10}" y="{y + 4}" class="tick" text-anchor="end">{s}</text>')
        for who, dy in (('mcg', -5), ('opp', 7)):
            v = SUM['setups'][who][s]
            out.append(f'<line x1="{X(v["ci"][0]):.1f}" x2="{X(v["ci"][1]):.1f}" y1="{y + dy}" y2="{y + dy}" class="{who} whisker"/>')
            out.append(f'<circle cx="{X(v["share"]):.1f}" cy="{y + dy}" r="5" class="{who} pt"><title>{"McGregor left" if who == "mcg" else "opponents, rear straight"}: {v["share"]:.0%} ({v["n"]}), 95% CI {v["ci"][0]:.0%} to {v["ci"][1]:.0%}</title></circle>')
    out.append('</svg>')
    return ''.join(out)


TIME = json.loads((HERE / 'results' / 'left' / 'time.json').read_text())
CLIPS = json.loads((HERE / 'results' / 'left' / 'clips.json').read_text())
OVER_TIME = [  # key, label, unit suffix, format, what "up" means, plain reading
    ('peak_speed', 'Peak fist speed', '', '{:.1f}', 'faster (torso lengths / s)'),
    ('lean_onset', 'Torso lean when the punch starts', '°', '{:.0f}', 'leaning further toward the opponent'),
    ('lean_change', 'Lean added during the punch', '°', '{:.0f}', 'throws more bodyweight forward'),
    ('lead_knee_onset', 'Lead knee angle at onset', '°', '{:.0f}', 'straighter lead leg (180° = straight)'),
    ('rear_knee_peak', 'Rear knee angle at full extension', '°', '{:.0f}', 'straighter rear leg, more push'),
    ('shoulder_turn', 'Shoulder turn through the punch', '', '{:.2f}', 'shoulders open more (2-D width change)'),
    ('lead_step', 'Lead foot step during the punch', '', '{:.2f}', 'steps in more (torso lengths)'),
    ('reach', 'Fist ahead of hips at full extension', '', '{:.2f}', 'longer (torso lengths)'),
    ('range_at_onset', 'Distance when it starts', '', '{:.2f}', 'starts from farther away'),
]


def time_charts() -> str:
    fights = [f for f in TIME['fights'] if f['n']['mcg'] >= 2]
    out = []
    for k, lab, unit, fmt, up in OVER_TIME:
        tr = TIME['trend'].get(k, {}); er = TIME['era'].get(k, {})
        note = f'ρ {tr["rho"]:+.2f}, p {tr["p"]:.2f} over {tr["n"]} fights' if tr else ''
        era = (f'2012–15 {fmt.format(er["early"])}{unit} → 2016–21 {fmt.format(er["late"])}{unit}' if er else '')
        cls = 'lean' if tr and tr['p'] < .1 else 'flat'
        word = 'trend' if tr and tr['p'] < .1 else 'steady'
        out.append(f'<article class="hyp small"><header><h3>{G.esc(lab)}</h3><span class="chip {cls}">{word}</span></header>'
                   f'<p class="read mono">{G.esc(era)}<br>{G.esc(note)}</p>'
                   f'<div class="chart-wrap">{G.timeline(fights, k, lab, unit, "2016-03-05", fmt=fmt, height=200)}</div>'
                   f'<p class="axis-note">Up = {G.esc(up)}.</p></article>')
    return ''.join(out)


def clip_grid() -> str:
    cells = []
    for c in [c for c in CLIPS if c['who'] == 'mcg']:
        cells.append(f'<figure class="clip"><video src="{c["file"]}#t=0.1" muted loop playsinline preload="metadata" aria-label="McGregor left, {c["date"][:4]} vs {G.esc(c["opponent"])}"></video>'
                     f'<figcaption><b>{c["date"][:4]} · {G.esc(c["opponent"].split()[-1])}</b><span>{G.esc(c["setup"])} · {c["peak_speed"]:.0f}</span></figcaption></figure>')
    return ''.join(cells)


def hero_clip() -> str:
    best = max((c for c in CLIPS if c['who'] == 'mcg'), key=lambda c: c['peak_speed'])
    return (f'<figure class="hero-clip"><video src="{best["file"]}" autoplay muted loop playsinline></video>'
            f'<figcaption>{best["date"][:4]} vs {G.esc(best["opponent"])} · {G.esc(best["setup"])} · half speed, tracked skeleton in red</figcaption></figure>')


def pair_grid() -> str:
    rows = []
    mc = {}
    for c in sorted((c for c in CLIPS if c['who'] == 'mcg'), key=lambda c: -c['peak_speed']):
        mc.setdefault(c['fight'], c)
    for o in sorted((c for c in CLIPS if c['who'] == 'opp'), key=lambda c: c['date']):
        m = mc.get(o['fight'])
        if not m: continue
        v = lambda c, who: (f'<figure class="clip"><video src="{c["file"]}#t=0.1" muted loop playsinline preload="metadata" aria-label="{who}, {c["date"][:4]}"></video>'
                            f'<figcaption><b>{who}</b><span>{G.esc(c["setup"])} · {c["peak_speed"]:.0f}</span></figcaption></figure>')
        rows.append(f'<div class="vs"><h3>{o["date"][:4]} · McGregor vs {G.esc(o["opponent"])}</h3><div class="vs-pair">'
                    + v(m, 'McGregor, left') + v(o, f'{G.esc(o["opponent"].split()[-1])}, rear hand') + '</div></div>')
    return ''.join(rows)


SLIPS = json.loads((HERE / 'results' / 'left' / 'slips.json').read_text())
LEAD = json.loads((HERE / 'results' / 'left' / 'leadhand.json').read_text())
SEL = json.loads((HERE / 'results' / 'left' / 'selection.json').read_text())


def head_paths() -> str:
    """Head paths relative to his hips in the 0.8 s before the left: pull counters, slip counters, the rest."""
    W_, H_ = 560, 360; s = 330; ox, oy = W_ * .62, H_ * .3
    out = [f'<svg viewBox="0 0 {W_} {H_}" class="chart" role="img" aria-label="Head paths before the left, relative to his hips">']
    out.append(f'<line x1="{ox}" x2="{ox}" y1="10" y2="{H_ - 30}" class="grid"/><line x1="20" x2="{W_ - 20}" y1="{oy}" y2="{oy}" class="grid"/>')
    out.append(f'<text x="{W_ - 24}" y="{oy - 8}" class="tick" text-anchor="end">toward opponent →</text><text x="24" y="{oy - 8}" class="tick">← away (pull)</text>'
               f'<text x="{ox + 8}" y="{H_ - 34}" class="tick">↓ down (dip)</text><text x="{ox + 8}" y="22" class="tick">start of window</text>')
    order = [('other', 'opp', .35), ('slip counter', 'early', .9), ('pull counter', 'mcg', .9)]
    for kind, cls, op in order:
        for e in SLIPS['events']:
            if e['who'] != 'mcg': continue
            k = e['setup'] if e['setup'] in ('pull counter', 'slip counter') else 'other'
            if k != kind: continue
            pts = [(ox + x * s, oy + y * s) for x, y in e['path'] if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
            if len(pts) < 5: continue
            d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
            out.append(f'<path d="{d}" class="hp {cls}" style="opacity:{op}"/><circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3" class="{cls} pt" style="opacity:{op}"/>')
    out.append('</svg>')
    return ''.join(out)


def counter_rows() -> str:
    rows = []
    names = {'pull counter': 'Pull counter', 'slip counter': 'Slip counter', 'counter, head still': 'Counter, head still',
             'caught coming in': 'Caught coming in', 'behind the paw': 'Behind the paw', 'straight lead': 'Straight lead'}
    f = lambda v, fmt: fmt.format(v) if v is not None and np.isfinite(v) else '–'
    for k, lab in names.items():
        r = LEAD['setups'][k]
        rows.append(f'<tr><th scope="row">{lab}<small>{r["n"]} punches, {r["fights"]} fights</small></th>'
                    f'<td class="num">{f(r["head_back"], "{:.2f}")}</td><td class="num">{f(r["head_down"], "{:.2f}")}</td>'
                    f'<td class="num">{f(r["lean_swing"], "{:.0f}°")}</td><td class="num">{f(r["lead_ms"], "{:.0f} ms")}</td>'
                    f'<td class="num">{f(r["speed"], "{:.1f}")}</td><td class="num">{f(r["range"], "{:.2f}")}</td></tr>')
    b = SLIPS['baseline']
    rows.append(f'<tr><th scope="row">No punch at all<small>{b["n"]} random moments of him standing</small></th>'
                f'<td class="num">{b["head_back"]:.2f}</td><td class="num">{b["head_down"]:.2f}</td><td class="num">{b["lean_swing"]:.0f}°</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>')
    return ''.join(rows)


def selection_charts() -> str:
    fights = SEL['fights']
    items = [('rear straight', 'Rear straight (his left)', '', '{:.1f}', 'more per upright minute'),
             ('rear hook', 'Rear hook (his left hook)', '', '{:.1f}', 'more per upright minute'),
             ('lead straight', 'Lead straight (jab)', '', '{:.1f}', 'more per upright minute'),
             ('lead hook', 'Lead hook', '', '{:.1f}', 'more per upright minute'),
             ('kicks', 'Kicks', '', '{:.1f}', 'more per upright minute'),
             ('rear_share', 'Share of hand strikes from the rear hand', '', '{:.2f}', 'more rear-handed')]
    out = []
    for k, lab, unit, fmt, up in items:
        t = SEL['tests'][k]
        out.append(f'<article class="hyp small"><header><h3>{G.esc(lab)}</h3><span class="chip {"lean" if t["mcg"]["p"] < .1 else "flat"}">ρ {t["mcg"]["rho"]:+.2f}</span></header>'
                   f'<p class="read mono">him {fmt.format(t["mcg"]["early"])} → {fmt.format(t["mcg"]["late"])} · opponents {fmt.format(t["opp"]["early"])} → {fmt.format(t["opp"]["late"])} · his higher in {t["mcg_more_in"][0]} of {t["mcg_more_in"][1]}</p>'
                   f'<div class="chart-wrap">{G.timeline(fights, k, lab, unit, "2016-03-05", fmt=fmt, height=190)}</div>'
                   f'<p class="axis-note">Up = {G.esc(up)} · p {t["mcg"]["p"]:.3f}, him alone</p></article>')
    return ''.join(out)


def page() -> str:
    tpl = (HERE / 'leftreport_template.html').read_text()
    r = SUM['rate']; c = SUM['trends']['counter_share']
    fill = {'n_lefts': str(SUM['n_lefts']), 'n_hooks': str(SUM['n_hooks']), 'n_opp': str(SUM['n_opp_rear']),
            'rate_mcg': f'{r["mcg"]:.1f}', 'rate_opp': f'{r["opp"]:.1f}', 'rate_ratio': f'{r["mcg"] / r["opp"]:.0f}',
            'rate_ci': f'{r["ci"][0]:+.1f} to {r["ci"][1]:+.1f}', 'rate_n': f'{r["mcg_higher_in"]} of {r["n_fights"]}',
            'rate_chart': rate_chart(), 'avg_left': average_left(), 'mechanics': mechanics_rows(), 'setups': setup_chart(),
            'counter_rho': f'{c["rho"]:+.2f}', 'counter_p': f'{c["p"]:.2f}',
            'time_charts': time_charts(), 'head_paths': head_paths(), 'counter_rows': counter_rows(), 'selection': selection_charts(),
            'big_mcg': f"{SLIPS['big_move_share']['mcg']:.0%}", 'big_base': f"{SLIPS['baseline']['big_move_share']:.0%}", 'big_opp': f"{SLIPS['big_move_share']['opp']:.0%}",
            'jab_mcg': f"{LEAD['jab_rate']['mcg']['value']:.1f}", 'jab_opp': f"{LEAD['jab_rate']['opp']['value']:.1f}",
            'jabbed_mcg': f"{LEAD['lead']['lead_jabbed']['mcg']['value']:.0%}", 'jabbed_opp': f"{LEAD['lead']['lead_jabbed']['opp']['value']:.0%}",
            'leadmove_mcg': f"{LEAD['lead']['lead_motion']['mcg']['value']:.1f}", 'leadmove_opp': f"{LEAD['lead']['lead_motion']['opp']['value']:.1f}",
            'rear_mcg': f"{SEL['tests']['rear_share']['mcg']['early']:.0%}", 'rear_opp': f"{SEL['tests']['rear_share']['opp']['early']:.0%}", 'clips': clip_grid(), 'pairs': pair_grid(), 'hero_clip': hero_clip(),
            'counter_points': ', '.join(f'{p["date"][:4]} {W[p["fight"]]["opponent"].split()[-1]} {p["value"]:.0%}' for p in c['points'])}
    for k, v in fill.items():
        tpl = tpl.replace('{{' + k + '}}', v)
    return tpl


if __name__ == '__main__':
    (HERE / 'results' / 'left' / 'report.html').write_text(page())
    print('wrote results/left/report.html')
