"""The McGregor page, a single-page writeup: results/site/index.html plus its data files and clips.

Tabs: Stance · The left hand · Syllables · Methods. Every number is read from the results files.

    python build_site.py
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE / 'results'
SITE = RES / 'site'
ERAS = [('2012–13', '2012', '2014'), ('2014–15', '2014', '2016'), ('2016', '2016', '2017'), ('2018–21', '2017', '2022')]
esc = lambda s: html.escape(str(s), quote=True)


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def era_chart(series_m, series_o, label, unit, fmt='{:.0f}', star='', height=230, invert_note=''):
    """One dot per era (mean of that era's fights, 95 % interval), McGregor red solid, opponents grey dashed."""
    W, H, L, R, T, B = 520, height, 56, 18, 28, 40
    def agg(series):
        out = []
        for name, a, b in ERAS:
            v = [x for d, x in series if a <= d < b and x is not None and np.isfinite(x)]
            if len(v) >= 2: out.append((name, float(np.mean(v)), 1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)), len(v)))
            elif len(v) == 1: out.append((name, float(v[0]), 0.0, 1))
            else: out.append((name, None, None, 0))
        return out
    m, o = agg(series_m), agg(series_o)
    vals = [v + s for _, v, c, _ in m + o if v is not None for s in (-c, c)]
    lo, hi = min(vals), max(vals); pad = (hi - lo) * .15 or 1; lo, hi = lo - pad, hi + pad
    if unit.strip() == '%': lo, hi = max(lo, 0), min(hi, 100)
    X = lambda i: L + (i + .5) * (W - L - R) / len(ERAS)
    Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    ticks = np.linspace(lo + pad * .5, hi - pad * .5, 4)
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="{esc(label)} by era, McGregor and opponents">']
    s.append(f'<text x="{L - 44}" y="{T - 10}" class="axl">{esc(label)}</text>')
    for tv in ticks:
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="g"/><text x="{L - 8}" y="{Y(tv) + 4:.1f}" class="tk" text-anchor="end">{fmt.format(tv)}</text>')
    for i, (name, *_ ) in enumerate(ERAS):
        s.append(f'<text x="{X(i):.1f}" y="{H - 14}" class="tk" text-anchor="middle">{name}</text>')
    for series, cls, dx in ((o, 'o', 6), (m, 'm', -6)):
        pts = [(X(i) + dx, Y(v), c, n, name) for i, (name, v, c, n) in enumerate(series) if v is not None]
        s.append('<polyline class="ln ' + cls + '" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y, *_ in pts) + '"/>')
        for x, y, c, n, name in pts:
            if c: s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{max(T, y - c / (hi - lo) * (H - T - B)):.1f}" y2="{min(H - B, y + c / (hi - lo) * (H - T - B)):.1f}" class="ci {cls}"/>')
            who = 'McGregor' if cls == 'm' else 'Opponents'
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if cls == "m" else 4.5}" class="pt {cls}"><title>{who}, {name}: {fmt.format(series[[e[0] for e in series].index(name)][1])}{unit} ({n} fights)</title></circle>')
    if star:
        s.append(f'<text x="{X(len(ERAS) - 1):.1f}" y="{T - 6}" class="star" text-anchor="middle">{star}</text>')
    s.append('</svg>')
    return ''.join(s)


def rate_bars(rows, unit_note):
    W, rh, L = 520, 24, 128
    hi = max(max(r[2], r[3]) for r in rows) * 1.08
    X = lambda v: L + v / hi * (W - L - 24)
    H = len(rows) * rh + 30
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="Straight lefts per minute, McGregor against his opponent, per fight">']
    for v in range(0, int(hi) + 1):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="0" y2="{H - 24}" class="g"/><text x="{X(v):.1f}" y="{H - 8}" class="tk" text-anchor="middle">{v}</text>')
    for i, (name, d, m, o) in enumerate(rows):
        y = i * rh + 14
        s.append(f'<text x="{L - 10}" y="{y + 4}" class="tk" text-anchor="end">{esc(name)}</text>')
        s.append(f'<line x1="{X(0)}" x2="{X(o):.1f}" y1="{y + 4}" y2="{y + 4}" class="bar o"><title>opponent: {o:.1f} {unit_note}</title></line>')
        s.append(f'<line x1="{X(0)}" x2="{X(m):.1f}" y1="{y - 3}" y2="{y - 3}" class="bar m"><title>McGregor: {m:.1f} {unit_note}</title></line>')
    s.append('</svg>')
    return ''.join(s)


def head_paths(sl):
    W, H = 520, 330; s = 300; ox, oy = W * .6, H * .28
    out = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="Head paths in the 0.8 s before each left, relative to his hips">']
    out.append(f'<line x1="{ox}" x2="{ox}" y1="8" y2="{H - 26}" class="g"/><line x1="16" x2="{W - 16}" y1="{oy}" y2="{oy}" class="g"/>')
    out.append(f'<text x="{W - 18}" y="{oy + 18}" class="tk" text-anchor="end">toward the opponent →</text><text x="18" y="{oy - 8}" class="tk">← away</text><text x="{ox + 8}" y="{H - 30}" class="tk">↓ down</text>')
    for who, cls, op in (('opp', 'o', .45), ('mcg', 'm', .7)):
        for e in sl['events']:
            if e['who'] != who: continue
            pts = [(ox + x * s, oy + y * s) for x, y in e['path'] if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
            if len(pts) < 5: continue
            out.append('<path class="hp ' + cls + f'" style="opacity:{op}" d="M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts) + '"/>')
    out.append('</svg>')
    return ''.join(out)


SETUPS = [('straight lead', 'Straight lead'), ('caught coming in', 'Caught coming in'), ('behind the paw', 'Lead hand out first'),
          ('counter, head still', 'Counter, head still'), ('slip counter', 'Slip counter'), ('pull counter', 'Pull counter')]


def mix_chart(mix):
    """Share of lefts by set-up: McGregor 2012-15 (ink), 2016-21 (red), opponents' rear straights (grey)."""
    W, rh, L = 520, 46, 150
    rows = [('early', 'k', '2012–15'), ('late', 'm', '2016–21'), ('opp', 'o', 'opponents')]
    hi = max(mix[w][k] / mix[w]['n'] for w, _, _ in rows for k, _ in SETUPS) * 1.1
    X = lambda v: L + v / hi * (W - L - 40)
    H = len(SETUPS) * rh + 30
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="How each straight left was set up, share of lefts">']
    for v in np.arange(0, hi, .1):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="0" y2="{H - 24}" class="g"/><text x="{X(v):.1f}" y="{H - 8}" class="tk" text-anchor="middle">{v:.0%}</text>')
    for i, (k, lab) in enumerate(SETUPS):
        y = i * rh + 12
        s.append(f'<text x="{L - 10}" y="{y + 16}" class="tk" text-anchor="end">{esc(lab)}</text>')
        for j, (w, cls, wl) in enumerate(rows):
            v = mix[w][k] / mix[w]['n']; yy = y + 6 + j * 9
            s.append(f'<line x1="{X(0)}" x2="{max(X(v), X(0) + 1.5):.1f}" y1="{yy}" y2="{yy}" class="bar {cls}"><title>{wl}: {mix[w][k]} of {mix[w]["n"]} ({v:.0%})</title></line>')
            if j == 1 or (w == 'opp' and False): pass
        s.append(f'<text x="{X(mix["late"][k] / mix["late"]["n"]) + 6:.1f}" y="{y + 19}" class="tk">{mix["late"][k]}/{mix["late"]["n"]}</text>')
    s.append('</svg>')
    return ''.join(s)


def counter_traces(sl, tc):
    """The 0.8 s before the left, his head against his hips: how far back (pull counters) and how far down (slip
    counters), each punch faint, the median bold. Paths resampled to a common clock."""
    W, H, L, R, T, B = 520, 250, 46, 18, 22, 38
    grid = np.linspace(0, 1, 25)
    def traces(setup, comp, sign):
        out = []
        for e in sl['events']:
            if e['who'] != 'mcg' or e['setup'] != setup: continue
            a = np.array([[np.nan if v is None else v for v in p] for p in e['path']], float)
            if len(a) < 6: continue
            x = np.linspace(0, 1, len(a)); y = sign * a[:, comp] * tc
            ok = np.isfinite(y)
            if ok.sum() < 6: continue
            out.append(np.interp(grid, x[ok], y[ok]))
        return np.array(out)
    pull, slip = traces('pull counter', 0, -1), traces('slip counter', 1, 1)
    hi = max(14, float(np.nanmax(np.r_[np.median(pull, 0), np.median(slip, 0)])) * 1.6); lo = -4
    X = lambda u: L + u * (W - L - R); Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="Head movement in the 0.8 s before pull counters and slip counters"><defs><clipPath id="ctc"><rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}"/></clipPath></defs>']
    for v in range(0, int(hi) + 1, 5):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g"/><text x="{L - 8}" y="{Y(v) + 4:.1f}" class="tk" text-anchor="end">{v}</text>')
    for u, lab in ((0, '−0.8 s'), (.5, '−0.4 s'), (1, 'the left starts')):
        s.append(f'<text x="{X(u):.1f}" y="{H - 12}" class="tk" text-anchor="{"start" if u == 0 else "end" if u == 1 else "middle"}">{lab}</text>')
    s.append(f'<line x1="{X(1):.1f}" x2="{X(1):.1f}" y1="{T}" y2="{H - B}" class="g" stroke-dasharray="3 4"/>')
    s.append(f'<text x="{L - 40}" y="{T - 8}" class="axl">Head travel (cm)</text>')
    for arr, cls in ((slip, 'k'), (pull, 'm')):
        for tr in arr:
            s.append(f'<polyline class="hp {cls}" clip-path="url(#ctc)" style="opacity:.22" points="' + ' '.join(f'{X(u):.1f},{Y(v):.1f}' for u, v in zip(grid, tr)) + '"/>')
    for arr, cls in ((slip, 'k'), (pull, 'm')):
        md = np.median(arr, 0)
        s.append(f'<polyline class="ln {cls}" points="' + ' '.join(f'{X(u):.1f},{Y(v):.1f}' for u, v in zip(grid, md)) + '"/>')
    k = int(np.argmax(np.median(pull, 0)))
    s.append(f'<circle cx="{X(grid[k]):.1f}" cy="{Y(np.median(pull, 0)[k]):.1f}" r="5" class="pt m"/>')
    s.append('</svg>')
    return s and ''.join(s), len(pull), len(slip)


def dumbbells(test, label, unit, fmt='{:.0f}'):
    """Per fight: round 1 (open dot) to minute 8 on (filled dot), McGregor red, opponent grey below him."""
    rows = [r for r in test['per_fight'] if all(v is not None and np.isfinite(v) for v in r[2:])]
    W, rh, L, R, T = 520, 34, 120, 22, 30
    vals = [v for r in rows for v in r[2:]]; lo, hi = min(vals), max(vals); pad = (hi - lo) * .1 or 1; lo, hi = lo - pad, hi + pad
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    H = T + len(rows) * rh + 26
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="{esc(label)}: round one against minute eight on, per fight">',
         f'<text x="4" y="16" class="axl">{esc(label)}</text>']
    for tv in np.linspace(lo + pad, hi - pad, 4):
        s.append(f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{T - 6}" y2="{H - 22}" class="g"/><text x="{X(tv):.1f}" y="{H - 6}" class="tk" text-anchor="middle">{fmt.format(tv)}</text>')
    for i, (f, d, me, ml, oe, ol) in enumerate(rows):
        y = T + i * rh + 10
        name = f"{d[:4]} {f.split('_')[1].rstrip('0123456789').title()} {f.split('_')[1][len(f.split('_')[1].rstrip('0123456789')):]}".strip()
        s.append(f'<text x="{L - 12}" y="{y + 9}" class="tk" text-anchor="end">{esc(name)}</text>')
        for (e, l, cls, dy) in ((me, ml, 'm', 0), (oe, ol, 'o', 12)):
            s.append(f'<line x1="{X(e):.1f}" x2="{X(l):.1f}" y1="{y + dy}" y2="{y + dy}" class="ln {cls}" style="stroke-dasharray:none;stroke-width:{2 if cls == "m" else 1.5}"/>')
            s.append(f'<circle cx="{X(e):.1f}" cy="{y + dy}" r="{4.5 if cls == "m" else 3.5}" class="hollow {cls}"><title>{"McGregor" if cls == "m" else "Opponent"}, round 1: {fmt.format(e)}{unit}</title></circle>')
            s.append(f'<circle cx="{X(l):.1f}" cy="{y + dy}" r="{4.5 if cls == "m" else 3.5}" class="pt {cls}"><title>{"McGregor" if cls == "m" else "Opponent"}, minute 8 on: {fmt.format(l)}{unit}</title></circle>')
    s.append('</svg>')
    return ''.join(s)


def minute_curve(fight, key, label, unit, fmt='{:.0f}', opp_name='Diaz', small=False):
    """One fight, minute by minute (3-minute window), both fighters; rounds marked; minute 8 marked."""
    m, o = fight['minutes']['mcg'], fight['minutes']['opp']
    xs = sorted({int(k) for k in m} | {int(k) for k in o})
    W, H, L, R, T, B = (300, 190, 34, 10, 30, 26) if small else (520, 220, 46, 16, 26, 36)
    vals = [v[key] for d_ in (m, o) for v in d_.values() if v[key] is not None]
    lo, hi = min(vals), max(vals); pad = (hi - lo) * .12 or 1; lo, hi = lo - pad, hi + pad
    X = lambda v: L + v / max(xs) * (W - L - R); Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="{esc(label)}, minute by minute">', f'<text x="4" y="16" class="axl" style="font-size:20px">{esc(label)}</text>']
    for tv in np.linspace(lo + pad, hi - pad, 4):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="g"/><text x="{L - 8}" y="{Y(tv) + 4:.1f}" class="tk" text-anchor="end">{fmt.format(tv)}</text>')
    for r_ in range(5, max(xs) + 1, 5):
        s.append(f'<line x1="{X(r_):.1f}" x2="{X(r_):.1f}" y1="{T}" y2="{H - B}" class="g" stroke-dasharray="2 4"/>')
    for r_ in range(0, max(xs), 5):
        s.append(f'<text x="{X(r_ + 2.5):.1f}" y="{H - 10}" class="tk" text-anchor="middle">R{r_ // 5 + 1}</text>')
    if max(xs) >= 8: s.append(f'<line x1="{X(8):.1f}" x2="{X(8):.1f}" y1="{T}" y2="{H - B}" class="mark8"/><text x="{X(8) + 4:.1f}" y="{H - B - 6}" class="tk">min 8</text>')
    for d_, cls in ((o, 'o'), (m, 'm')):
        pts = [(X(int(k)), Y(v[key])) for k, v in sorted(d_.items(), key=lambda kv: int(kv[0])) if v[key] is not None]
        s.append(f'<polyline class="ln {cls}" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts) + '"/>')
    s.append('</svg>')
    return ''.join(s)


def counter_story(sl, tc, pull_ms):
    """Pull against slip, told plainly: the median head path for each (band: middle half), labelled on the lines,
    the pull's furthest-back point and the gap to the punch marked."""
    W, H, L, R, T, B = 640, 300, 50, 120, 30, 44
    grid = np.linspace(0, 1, 25)
    def traces(setup, comp, sign):
        out = []
        for e in sl['events']:
            if e['who'] != 'mcg' or e['setup'] != setup: continue
            a = np.array([[np.nan if v is None else v for v in p] for p in e['path']], float)
            if len(a) < 6: continue
            x = np.linspace(0, 1, len(a)); y = sign * a[:, comp] * tc; ok = np.isfinite(y)
            if ok.sum() >= 6: out.append(np.interp(grid, x[ok], y[ok]))
        return np.array(out)
    pull, slip = traces('pull counter', 0, -1), traces('slip counter', 1, 1)
    lo, hi = -2, 12
    X = lambda u: L + u * (W - L - R); Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="Pull counter against slip counter: the head in the 0.8 s before the left">']
    for v in (0, 5, 10):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g"/><text x="{L - 8}" y="{Y(v) + 4:.1f}" class="tk" text-anchor="end">{v} cm</text>')
    s.append(f'<line x1="{X(1):.1f}" x2="{X(1):.1f}" y1="{T - 10}" y2="{H - B}" class="mark8"/><text x="{X(1):.1f}" y="{T - 14}" class="tk" text-anchor="middle">the left starts</text>')
    for u, lab in ((0, '0.8 s before'), (.5, '0.4 s')):
        s.append(f'<text x="{X(u):.1f}" y="{H - 16}" class="tk" text-anchor="{"start" if u == 0 else "middle"}">{lab}</text>')
    for arr, cls, name in ((slip, 'k', 'slip: head down'), (pull, 'm', 'pull: head back')):
        q1, md, q3 = np.percentile(arr, 25, 0), np.median(arr, 0), np.percentile(arr, 75, 0)
        col = 'var(--red)' if cls == 'm' else 'var(--ink)'
        band = ' '.join(f'{X(u):.1f},{Y(v):.1f}' for u, v in zip(grid, q3)) + ' ' + ' '.join(f'{X(u):.1f},{Y(v):.1f}' for u, v in zip(grid[::-1], q1[::-1]))
        s.append(f'<polygon points="{band}" style="fill:{col};opacity:.09"/>')
        s.append(f'<polyline class="ln {cls}" style="stroke-width:3" points="' + ' '.join(f'{X(u):.1f},{Y(v):.1f}' for u, v in zip(grid, md)) + '"/>')
        s.append(f'<text x="{X(1) + 8:.1f}" y="{Y(md[-1]) + 5:.1f}" class="axl" style="fill:{col};font-weight:600">{name}</text>')
        if cls == 'm':
            k = int(np.argmax(md)); xk, yk = X(grid[k]), Y(md[k])
            s.append(f'<circle cx="{xk:.1f}" cy="{yk:.1f}" r="6" class="pt m"/>')
            s.append(f'<text x="{xk:.1f}" y="{yk - 14:.1f}" class="tk" text-anchor="middle" style="fill:var(--red)">furthest back, {md[k]:.0f} cm</text>')
            yb = Y(hi - .5)
            s.append(f'<path d="M{xk:.1f},{yk - 30:.1f} V{yb:.1f} H{X(1):.1f} V{yb + 8:.1f}" style="fill:none;stroke:var(--red);stroke-width:1.5"/>')
            s.append(f'<text x="{(xk + X(1)) / 2:.1f}" y="{yb - 6:.1f}" class="tk" text-anchor="middle" style="fill:var(--red)">{pull_ms} ms</text>')
    s.append('</svg>')
    return ''.join(s), len(pull), len(slip)


def rate_pairs(rows, unit, title):
    """Horizontal paired bars, rows: (label, mcg, opp, note)."""
    W, rh, L = 640, 50, 170
    hi = max(max(r[1], r[2]) for r in rows) * 1.25
    X = lambda v: L + v / hi * (W - L - 70)
    H = len(rows) * rh + 34
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="{esc(title)}"><text x="4" y="16" class="axl">{esc(title)}</text>']
    for i, (lab, m, o, note) in enumerate(rows):
        y = 34 + i * rh
        s.append(f'<text x="{L - 10}" y="{y + 14}" class="tk" text-anchor="end" style="font-size:13px">{esc(lab)}</text>')
        s.append(f'<line x1="{L}" x2="{X(m):.1f}" y1="{y + 6}" y2="{y + 6}" class="bar m" style="stroke-width:12"/><text x="{X(m) + 8:.1f}" y="{y + 10}" class="tk" style="fill:var(--red)">{m:.0f}{unit}</text>')
        s.append(f'<line x1="{L}" x2="{max(X(o), L + 2):.1f}" y1="{y + 22}" y2="{y + 22}" class="bar o" style="stroke-width:12"/><text x="{max(X(o), L + 2) + 8:.1f}" y="{y + 26}" class="tk">{o:.0f}{unit}</text>')
    s.append('</svg>')
    return ''.join(s)


def speed_chart(rows):
    """Per fight: median fist speed of his left (red) and the opponent's rear straight (grey), mph."""
    W, rh, L, R = 640, 30, 130, 30
    lo, hi = 8, 20
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    H = len(rows) * rh + 50
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="Fist speed per fight, his left against the opponent rear straight">']
    for v in (10, 12, 14, 16, 18):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="8" y2="{H - 30}" class="g"/><text x="{X(v):.1f}" y="{H - 12}" class="tk" text-anchor="middle">{v} mph</text>')
    for i, (lab, m, o) in enumerate(rows):
        y = 20 + i * rh
        s.append(f'<text x="{L - 12}" y="{y + 4}" class="tk" text-anchor="end" style="font-size:13px">{esc(lab)}</text>')
        s.append(f'<line x1="{X(m):.1f}" x2="{X(o):.1f}" y1="{y}" y2="{y}" class="g" style="stroke:#bbb;stroke-width:2"/>')
        s.append(f'<circle cx="{X(o):.1f}" cy="{y}" r="6" class="pt o"/><circle cx="{X(m):.1f}" cy="{y}" r="7" class="pt m"/>')
    s.append('</svg>')
    return ''.join(s)


def break_chart(bp, label, unit, fan=8):
    """Every half-minute of standing footage (each fight centred on its own average), the fitted flat-then-change
    line, the evidence's break and the fan's minute 8."""
    pts = bp['points']; tau, b = bp['tau_min'], bp['slope_per_min']
    by = {}
    for f, t, v in pts: by.setdefault(f, []).append(v)
    mu = {f: np.mean(v) for f, v in by.items()}
    xs = [t for _, t, _ in pts]; ys = [v - mu[f] for f, _, v in pts]
    W, H, L, R, T, B = 640, 280, 50, 16, 34, 40
    tmax = max(xs) + .5
    lo, hi = np.percentile(ys, 2), np.percentile(ys, 98); pad = (hi - lo) * .1; lo, hi = lo - pad, hi + pad
    X = lambda t: L + t / tmax * (W - L - R); Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" class="era" role="img" aria-label="{esc(label)} through the fight"><text x="4" y="16" class="axl">{esc(label)}</text>']
    for tv in np.linspace(lo + pad, hi - pad, 3):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="g"/><text x="{L - 8}" y="{Y(tv) + 4:.1f}" class="tk" text-anchor="end">{tv:+.0f}</text>')
    for m_ in range(0, int(tmax) + 1, 5):
        s.append(f'<text x="{X(m_):.1f}" y="{H - 14}" class="tk" text-anchor="middle">min {m_}</text>')
    for x, y in zip(xs, ys):
        if lo <= y <= hi: s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" style="fill:#9a9a9a;opacity:.55"/>')
    tt = np.linspace(0, tmax, 80); yh = b * np.maximum(0, tt - tau)
    yh = yh - np.mean([b * max(0, x - tau) for x in xs])
    s.append('<polyline class="ln m" style="stroke-width:3" points="' + ' '.join(f'{X(t):.1f},{Y(min(hi, max(lo, v))):.1f}' for t, v in zip(tt, yh)) + '"/>')
    s.append(f'<line x1="{X(fan):.1f}" x2="{X(fan):.1f}" y1="{T}" y2="{H - B}" class="mark8"/><text x="{X(fan) + 4:.1f}" y="{T + 12}" class="tk">the fans\' minute 8</text>')
    if tau < tmax - .5:
        s.append(f'<line x1="{X(tau):.1f}" x2="{X(tau):.1f}" y1="{T}" y2="{H - B}" style="stroke:var(--red);stroke-width:1.5"/><text x="{X(tau) + 4:.1f}" y="{T + 28}" class="tk" style="fill:var(--red)">the data: minute {tau:.0f}</text>')
    s.append('</svg>')
    return ''.join(s)


def main():
    if SITE.exists(): shutil.rmtree(SITE)
    (SITE / 'clips').mkdir(parents=True)
    st = json.loads((RES / 'stance' / 'findings.json').read_text())
    lf = json.loads((RES / 'left' / 'findings.json').read_text())
    pr = json.loads((RES / 'pairs' / 'pairs.json').read_text())
    sl = json.loads((RES / 'left' / 'slips.json').read_text())
    summ = json.loads((RES / 'left' / 'summary.json').read_text())
    clips = json.loads((RES / 'left' / 'clips.json').read_text())
    tc = st['torso_cm']
    meas = st['measures']
    ser = lambda k, who: [(d, v) for _, d, _, v in meas[k][who]['per_fight']]
    # copy media
    for p in pr['stance'] + pr['left'] + pr.get('counters', []) + pr.get('fatigue', []) + pr.get('drops', []) + pr.get('vs', []) + ([pr['counter_ie']] if 'counter_ie' in pr else []) + ([pr['blade']] if 'blade' in pr else []):
        shutil.copy(RES / p['file'], SITE / 'clips' / Path(p['file']).name)
    for c in clips:
        shutil.copy(RES / 'left' / c['file'], SITE / 'clips' / Path(c['file']).name)
    shutil.copy(RES / 'stance' / 'clouds.json', SITE / 'clouds.json')
    shutil.copy(RES / 'site_extra' / 'hero.json', SITE / 'hero.json')
    shutil.copy(RES / 'pairs' / 'left_study.mp4', SITE / 'clips' / 'left_study.mp4')
    gr = RES / 'game' / 'rounds.json'
    if gr.exists():
        rounds = json.loads(gr.read_text())
        for j, r in enumerate(rounds):
            shutil.copy(RES / 'game' / r['file'], SITE / 'clips' / f'game_{j}.mp4'); r['clip'] = f'clips/game_{j}.mp4'
        (SITE / 'game.json').write_text(json.dumps(rounds, separators=(',', ':')))
    syl = RES / 'syllables' / 'syllables_site.json'
    if syl.exists():
        shutil.copy(syl, SITE / 'syllables.json')
        if (RES / 'syllables' / 'syllables_inst.json').exists(): shutil.copy(RES / 'syllables' / 'syllables_inst.json', SITE / 'syllables_inst.json')
        for f in (RES / 'syllables' / 'clips').glob('*.mp4'):
            shutil.copy(f, SITE / 'clips' / f.name)

    # stance: the null grid
    grid_keys = [('blade', 'Bladed shoulders', '°', '{:.0f}', 'How far the shoulder line is turned from square toward the opponent: 90° fully side-on (karate), 0° squared up (boxing).'),
                 ('width', 'Stance width', ' cm', '{:.0f}', 'Ankle to ankle. Wide is the karate stance.'),
                 ('length', 'Stance length', ' cm', '{:.0f}', 'How far the lead foot sits ahead of the rear, along the line to the opponent.'),
                 ('lean', 'Torso lean', '°', '{:.1f}', 'The hip-to-shoulder line against vertical; + is toward the opponent. The boxer leans in.'),
                 ('shoulder', 'Shoulder tilt', ' cm', '{:.1f}', 'Lead shoulder above the rear one; + is the boxer\'s shoulder raised over the chin.'),
                 ('crouch', 'Hip height', ' cm', '{:.0f}', 'Hips above the ankles. Lower is a deeper crouch.'),
                 ('rear_knee', 'Rear knee', '°', '{:.0f}', 'The angle at the back knee; 180° is a straight leg.'),
                 ('weight', 'Weight on the lead foot', '', '{:.2f}', 'Where the hips sit between the feet: 0 over the rear foot, 1 over the lead.'),
                 ('step_speed', 'Footwork speed', ' cm/s', '{:.0f}', 'How fast the hips travel across the cage. The bounce shows up here.')]
    MOVED = ('blade', 'lean', 'shoulder', 'step_speed')
    moved_keys = [g for g in grid_keys if g[0] in MOVED]; grid_keys = [g for g in grid_keys if g[0] not in MOVED and g[0] != 'blade']
    for k, *_ in grid_keys + moved_keys:
        src = RES / 'pairs' / f'measure_{k}.mp4'
        if src.exists(): shutil.copy(src, SITE / 'clips' / src.name)
    mw = json.loads((RES / 'pairs' / 'pairs.json').read_text()).get('measures', {})
    btns, pans = [], []
    for i, (k, lab, u, f, dfn) in enumerate(grid_keys):
        mm = meas[k]['mcg']
        eras_txt = ' · '.join(f"{e['era']} vs {e['opponent'].split()[-1]}" for e in mw.get(k, {}).get('eras', []) if e)
        btns.append(f'<button role="tab" aria-selected="{"true" if i == 0 else "false"}" data-m="{i}">{esc(lab)}</button>')
        pans.append(f'<div class="mpan" data-m="{i}"{" hidden" if i else ""}>'
                    f'<video class="mstrip" src="clips/measure_{k}.mp4#t=0.1" muted loop playsinline preload="{"metadata" if i == 0 else "none"}" aria-label="{esc(lab)} in each era, drawn on McGregor"></video>'
                    f'<p class="cap">{esc(eras_txt)} · the most typical steady stretch of each era for this measure</p>'
                    f'<p class="mdef"><b>{esc(lab)}.</b> {esc(dfn)} Across the eras: {f.format(mm["early"])}{u} in 2012–15, {f.format(mm["late"])}{u} in 2016–21 (corrected p {mm["holm_p"]:.2f}).</p>'
                    f'<figure class="box">{era_chart(ser(k, "mcg"), ser(k, "opp"), lab + (" (" + u.strip() + ")" if u.strip() else ""), u, f)}'
                    f'<div class="key"><span><i class="m"></i>McGregor</span><span><i class="o"></i>his opponents, same frames</span></div></figure></div>')
    grid = f'<div class="mpick"><div class="ptabs mtabs" role="tablist" aria-label="Measure">{"".join(btns)}</div>{"".join(pans)}</div>'
    bl = meas['blade']; blo = {f: v for f, _, _, v in bl['opp']['per_fight']}
    brows = [(d, v - blo[f], v, blo[f]) for f, d, _, v in bl['mcg']['per_fight'] if f in blo]
    from scipy.stats import ttest_ind as _tt
    be, bl_ = [r for r in brows if r[0] < '2016'], [r for r in brows if r[0] >= '2016']
    blade_p = float(_tt([r[1] for r in bl_], [r[1] for r in be], equal_var=False).pvalue)
    blade_chart = era_chart(ser('blade', 'mcg'), ser('blade', 'opp'), 'Shoulders turned side-on (°)', '°')
    fill_b = {'blade_chart': blade_chart, 'bl_m_e': f"{np.mean([r[2] for r in be]):.0f}", 'bl_m_l': f"{np.mean([r[2] for r in bl_]):.0f}",
              'bl_o_e': f"{np.mean([r[3] for r in be]):.0f}", 'bl_o_l': f"{np.mean([r[3] for r in bl_]):.0f}",
              'bl_d_e': f"{np.mean([r[1] for r in be]):.0f}", 'bl_d_l': f"{abs(np.mean([r[1] for r in bl_])):.0f}",
              'bl_n_e': f"{sum(r[1] > 0 for r in be)} of {len(be)}", 'bl_n_l': f"{sum(r[1] > 0 for r in bl_)} of {len(bl_)}", 'bl_p': f"{blade_p:.2f}",
              'bl_mp': f"{bl['mcg']['p']:.2f}", 'bl_h': f"{bl['mcg']['holm_p']:.2f}"}
    def era_vals(k, who, test):
        v = [x for f, d, _, x in meas[k][who]['per_fight'] if test(f, d)]
        return float(np.mean(v)) if v else np.nan
    early_ = lambda f, d: d < '2016'; poi = lambda f, d: f.startswith('2021_poirier')
    MV = {'blade': ('Bladed shoulders', '°', 'How far the shoulder line is turned toward the opponent: 90° side-on (karate), 0° squared (boxing).'),
          'lean': ('Torso lean', '°', 'The hip-to-shoulder line against vertical; + leans toward the opponent.'),
          'shoulder': ('Shoulder tilt', ' cm', 'Lead shoulder above the rear one.'),
          'step_speed': ('Footwork speed', ' cm/s', 'How fast the hips travel across the cage.')}
    mv_html = []
    for k, (lab, u, dfn) in MV.items():
        me_, ml_, oe_, ol_ = era_vals(k, 'mcg', early_), era_vals(k, 'mcg', poi), era_vals(k, 'opp', early_), era_vals(k, 'opp', poi)
        fmt_ = '{:.0f}' if k != 'shoulder' else '{:+.1f}'
        TT = {'blade': 'He squared up', 'lean': 'He leaned in', 'shoulder': 'His lead shoulder dropped', 'step_speed': 'His feet slowed'}
        mv_html.append(f'<div class="mv"><h3>{esc(TT[k])}</h3><p class="small">{esc(dfn)}</p>'
                       f'<div class="num">{fmt_.format(me_)} → {fmt_.format(ml_)}{u} <small>his opponents {fmt_.format(oe_)} → {fmt_.format(ol_)}{u}</small></div>'
                       f'<video class="mstrip" src="clips/measure_{k}.mp4#t=0.1" muted loop playsinline preload="metadata" aria-label="{esc(lab)} in each era"></video>'
                       f'<figure class="box" style="margin-top:8px">{era_chart(ser(k, "mcg"), ser(k, "opp"), lab + " (" + u.strip() + ")", u, "{:.1f}" if k in ("lean", "shoulder") else "{:.0f}")}'
                       f'<div class="key"><span><i class="m"></i>McGregor</span><span><i class="o"></i>opponents, same frames</span></div></figure></div>')
    moved_html = '<div class="moved">' + ''.join(mv_html) + '</div>'
    hs, gs = meas['hunch']['signature'], meas['guard']['signature']
    head_chart = era_chart(ser('hunch', 'mcg'), ser('hunch', 'opp'), 'Nose above the shoulder line (cm)', ' cm', star=stars(hs['holm_p']))
    guard_chart = era_chart(ser('guard', 'mcg'), ser('guard', 'opp'), 'Wrists above the shoulder line (cm)', ' cm', star=stars(gs['holm_p']))
    reach = meas['reach']['signature']
    nulls_tested = len(meas)

    # left hand
    lr = lf['left_rate']
    rate_rows = [(f"{d[:4]} {f.split('_')[1].rstrip('0123456789').title()} {f.split('_')[1][len(f.split('_')[1].rstrip('0123456789')):]}".strip(), d, m, o) for f, d, m, o in lr['per_fight']]
    rates = rate_bars(rate_rows, 'per minute on the feet')
    rs = lf['rear_share']; bh = lf['big_head_move']
    left_pairs = pr['left']; stance_pairs = [pr['stance'][i] for i in (0, 2, 1)]       # reader's order: pair 3 second
    vs = {}
    for c in clips:
        vs.setdefault((c['fight'], c['who']), c)
    vs_pairs = [(vs[(f, 'mcg')], vs[(f, 'opp')]) for f in sorted({c['fight'] for c in clips}) if (f, 'mcg') in vs and (f, 'opp') in vs]
    vs_pick = sorted(vs_pairs, key=lambda p: -p[0]['peak_speed'])[:3]
    speeds = [e for e in json.loads((RES / 'left' / 'events.json').read_text()) if e['kind'] == 'straight']
    sp_m = np.median([e['peak_speed'] for e in speeds if e['who'] == 'mcg']) * tc / 100
    sp_o = np.median([e['peak_speed'] for e in speeds if e['who'] == 'opp']) * tc / 100

    def pairtabs(name, items, render):
        btn = ''.join(f'<button role="tab" aria-selected="{"true" if i == 0 else "false"}" data-g="{name}" data-i="{i}">Pair {i + 1}</button>' for i in range(len(items)))
        pan = ''.join(f'<div class="pp" data-g="{name}" data-i="{i}"{" hidden" if i else ""}>{render(p)}</div>' for i, p in enumerate(items))
        return f'<div class="pairs"><div class="ptabs" role="tablist">{btn}</div>{pan}</div>'

    vid = lambda src, alt: f'<video src="{src}#t=0.1" muted loop playsinline preload="metadata" aria-label="{esc(alt)}"></video>'
    stance_pairs_html = pairtabs('st', stance_pairs, lambda p: vid('clips/' + Path(p['file']).name, f"{p['early']['date'][:4]} against {p['late']['date'][:4]}"))
    left_pairs_html = pairtabs('lp', left_pairs, lambda p: vid('clips/' + Path(p['file']).name, f"Left, {p['early']['date'][:4]} against {p['late']['date'][:4]}"))
    vs_html = pairtabs('vs', pr.get('vs', []), lambda p: vid('clips/' + Path(p['file']).name, 'His left beside the opponent rear straight')) if pr.get('vs') else pairtabs('vs', vs_pick, lambda p: '<div class="two">' + vid('clips/' + Path(p[0]['file']).name, 'McGregor left') + vid('clips/' + Path(p[1]['file']).name, 'Opponent rear straight') +
                       f'</div><p class="cap">{p[0]["date"][:4]} vs {esc(p[0]["opponent"])} · his fastest left beside {esc(p[0]["opponent"].split()[-1])}\'s fastest rear straight · half speed</p>')
    counter_html = pairtabs('cp', ([pr['counter_ie']] if 'counter_ie' in pr else []) + pr.get('counters', [])[:2], lambda p: vid('clips/' + Path(p['file']).name, 'Pull counter beside slip counter'))
    traces_svg, n_pull, n_slip = counter_story(sl, tc, f"{lf['counter_timing']['pull_ms']:.0f}")
    pf = lambda k: [(d, m) for _, d, m, _ in lf[k]['per_fight']], lambda k: [(d, o) for _, d, _, o in lf[k]['per_fight']]
    pct = lambda rows: [(d, 100 * v) for d, v in rows]
    rear_chart = era_chart(pct(pf[0]('rear_share')), pct(pf[1]('rear_share')), 'Punches from the rear hand (%)', ' %', star=stars(rs['holm_p']))
    still_chart = era_chart(pct(pf[0]('big_head_move')), pct(pf[1]('big_head_move')), 'Big head move before the punch (%)', ' %', star=stars(bh['holm_p']))
    lm = lf['lead_motion']
    cmr = lambda rows: [(d, v * tc) for d, v in rows]
    lead_chart = era_chart(cmr(pf[0]('lead_motion')), cmr(pf[1]('lead_motion')), 'Lead hand travel, last second (cm)', ' cm', star=stars(lm['holm_p']))
    pres = json.loads((RES / 'left' / 'pressure.json').read_text())
    pkeys = [('first_share', 'Strikes that start an exchange', 'era_first_share'), ('counter_share', 'Strikes that answer one', 'era_strike_counter_share'),
             ('head_share', 'Strikes to the head', 'era_head_share'), ('backing_share', 'Strikes while backing up', 'era_backing_share')]
    hunt_grid = ''.join(f'<figure class="cell">{era_chart([(r["date"], 100 * r["mcg"][k]) for r in pres["fights"]], [(r["date"], 100 * r["opp"][k]) for r in pres["fights"]], lab + " (%)", " %", height=200)}'
                        f'<figcaption>{esc(lab)}: {100 * lf[ek]["early"]:.0f} % → {100 * lf[ek]["late"]:.0f} % · corrected p {lf[ek]["holm_p"]:.2f}</figcaption></figure>' for k, lab, ek in pkeys)
    cs = lf['era_counter_share']
    counters_late = era_chart([(d, 100 * v) for d, v in cs['per_fight']], [], 'Lefts thrown as counters (%)', ' %')
    mix = lf['_mix']; sets = lf['_setups']
    evj = json.loads((RES / 'left' / 'evasion.json').read_text()); EP = evj['punches']
    evd = lambda who: [p for p in EP if p['target'] == who and p['move'] in ('pull', 'slip') and p['gap_cm'] > 0]
    ca_, cb_ = sum(p['countered'] for p in evd('mcg')), sum(p['countered'] for p in evd('opp'))
    from scipy.stats import fisher_exact as _fe
    ctr_p = _fe([[ca_, len(evd('mcg')) - ca_], [cb_, len(evd('opp')) - cb_]])[1]
    po_ = evj['pooled']
    drops_html = ''.join(f'<figure><video src="clips/{Path(p["file"]).name}#t=0.1" muted loop playsinline preload="metadata" aria-label="{esc(p["opponent"])}"></video>'
                         f'<figcaption>{p["date"][:4]} vs {esc(p["opponent"])} · {esc(p["sub"])}</figcaption></figure>' for p in pr.get('drops', []))
    pres_ = json.loads((RES / 'left' / 'pressure.json').read_text())
    hunt_one = (f'<figure class="cell box">{era_chart([(r["date"], 100 * r["mcg"]["first_share"]) for r in pres_["fights"]], [(r["date"], 100 * r["opp"]["first_share"]) for r in pres_["fights"]], "Strikes that start an exchange (%)", " %", height=200)}'
                f'<figcaption>Counter-puncher to head hunter? {100 * lf["era_first_share"]["early"]:.0f} % → {100 * lf["era_first_share"]["late"]:.0f} %: no.</figcaption></figure>')
    fa0 = json.loads((RES / 'fatigue' / 'fatigue.json').read_text())
    late_cells = []
    for r in fa0['fights']:
        if r['official_s'] < 360 or 'mcg' not in r['minutes'] or len(r['minutes']['mcg']) < 4: continue
        nm = r['fight'].split('_')[1]; num = nm[len(nm.rstrip('0123456789')):]
        ttl = r['date'][:4] + ' ' + r['opponent'].split()[-1] + (' ' + num if num else '')
        late_cells.append('<figure class="box mini">' + minute_curve(r, 'steps_min', ttl, '', small=True) + '</figure>')
    late_grid = ''.join(late_cells)
    one = lambda path, alt: f'<video src="clips/{Path(path).name}#t=0.1" muted loop playsinline preload="metadata" aria-label="{esc(alt)}"></video>'
    bp = json.loads((RES / 'fatigue' / 'breakpoint.json').read_text())
    ev_rates = [('after a pull', 100 * po_['mcg']['counter_after_pull'], 100 * po_['opp']['counter_after_pull'], ''),
                ('after a slip', 100 * po_['mcg']['counter_after_slip'], 100 * po_['opp']['counter_after_slip'], '')]
    evl = [e for e in json.loads((RES / 'left' / 'events.json').read_text()) if e['kind'] == 'straight']
    mph_ = lambda e: e['peak_speed'] * tc / 100 * 2.237
    sp_rows = []
    for f in sorted({e['fight'] for e in evl}, key=lambda f: f):
        m = [mph_(e) for e in evl if e['fight'] == f and e['who'] == 'mcg']; o = [mph_(e) for e in evl if e['fight'] == f and e['who'] == 'opp']
        if len(m) >= 2 and len(o) >= 2:
            nm = f.split('_')[1]; num = nm[len(nm.rstrip('0123456789')):]
            sp_rows.append((f[:4] + ' ' + nm.rstrip('0123456789').title() + (' ' + num if num else ''), float(np.median(m)), float(np.median(o))))
    from scipy.stats import wilcoxon as _wx
    sp_p = float(_wx([m - o for _, m, o in sp_rows]).pvalue)
    stp = [pr['stance'][i] for i in (2, 1, 0)]
    fill_n = {
        'head_video': one(stp[0]['file'], 'Head height, McGregor against his opponent'), 'hands_video': one(stp[1]['file'], 'Hands, McGregor against his opponent'),
        'blade_video': one(pr['blade']['file'], 'Shoulders, McGregor against his opponent') if 'blade' in pr else '',
        'rate_video': one(pr['vs'][0]['file'], 'His fastest left beside Alvarez\'s fastest right'), 'rear_video': one(pr['vs'][2]['file'], 'Siver'),
        'speed_video': one(pr['vs'][1]['file'], 'The knockdown left beside Alvarez\'s right'), 'counter_video': one(pr['counter_ie']['file'], 'Pull and slip counters'),
        'drops_two': ''.join(f'<figure><video src="clips/{Path(p["file"]).name}#t=0.1" muted loop playsinline preload="metadata" aria-label="{esc(p["opponent"])}"></video><figcaption>{p["date"][:4]} vs {esc(p["opponent"])} · {esc(p["sub"])}</figcaption></figure>' for p in pr.get('drops', [])[2:4]),
        'study_video': one('pairs/left_study.mp4', 'His lefts beside every left he threw, stacked'),
        'era_video': one(pr['left'][0]['file'], 'A typical left, early and late'),
        'ctr_chart': rate_pairs(ev_rates, ' %', 'Fired back within 0.6 s of making the punch miss'),
        'speed_chart': speed_chart(sp_rows), 'sp_m': f"{np.median([m for _, m, _ in sp_rows]):.0f}", 'sp_o': f"{np.median([o for _, _, o in sp_rows]):.0f}",
        'sp_n': f"faster in {sum(m > o for _, m, o in sp_rows)} of {len(sp_rows)} fights · p {sp_p:.2f}",
        'late_steps_video': one(pr['fatigue'][0]['file'], 'Diaz 2, early and late'), 'late_hands_video': one(pr['fatigue'][1]['file'], 'Khabib, early and late'),
        'steps_dumb': dumbbells(fa0['tests']['self_steps_min'], 'Steps a minute: round 1 (open) → after minute 8 (filled)', '', '{:.0f}'),
        'guard_break': break_chart(bp['guard_cm'], 'His hands against the shoulder line, cm (each fight centred)', ' cm'),
        'steps_break': break_chart(bp['steps_min'], 'His steps a minute (each fight centred)', ''),
        'g_tau': f"{bp['guard_cm']['tau_min']:.0f}", 'g_slope': f"{abs(bp['guard_cm']['slope_per_min']) * 5:.0f}", 'g_neg': f"{bp['guard_cm']['share_slope_neg']:.0%}",
        's_tau': f"{bp['steps_min']['tau_min']:.0f}", 's_lo': f"{bp['steps_min']['tau_ci'][0]:.0f}", 's_hi': f"{bp['steps_min']['tau_ci'][1]:.0f}",
    }
    fa = json.loads((RES / 'fatigue' / 'fatigue.json').read_text()); ft_ = fa['tests']; fs_ = fa['slopes']
    d2 = next(r for r in fa['fights'] if r['fight'] == '2016_diaz2')
    fat_pairs = pairtabs('fp', pr.get('fatigue', []), lambda p: vid('clips/' + Path(p['file']).name, 'Same fight, early beside late'))
    fat_dumb = ''.join(f'<figure class="cell">{dumbbells(ft_["self_" + k], lab, u, f)}</figure>' for k, lab, u, f in
                       (('steps_min', 'Steps a minute', '', '{:.0f}'), ('heel_lift', 'Heel lift (0 flat)', '', '{:.2f}'), ('guard_cm', 'Hands above shoulders (cm)', ' cm', '{:.0f}'), ('width_cm', 'Feet apart (cm)', ' cm', '{:.0f}')))
    fv = lambda k, w: ft_['self_' + k][w]
    fill_f = {'fat_pairs': fat_pairs, 'fat_dumb': fat_dumb,
              'd2_steps': minute_curve(d2, 'steps_min', 'Steps a minute, Diaz 2', ''), 'd2_guard': minute_curve(d2, 'guard_cm', 'Hands above the shoulder line (cm), Diaz 2', ' cm'),
              'd2_width': minute_curve(d2, 'width_cm', 'Feet apart (cm), Diaz 2', ' cm'),
              'f_n': str(ft_['self_steps_min']['n']),
              'f_steps_e': f"{fv('steps_min', 'mcg_early'):.0f}", 'f_steps_l': f"{fv('steps_min', 'mcg_late'):.0f}", 'f_steps_oe': f"{fv('steps_min', 'opp_early'):.0f}", 'f_steps_ol': f"{fv('steps_min', 'opp_late'):.0f}",
              'f_steps_down': f"{ft_['self_steps_min']['down_in']} of {ft_['self_steps_min']['n']}", 'f_steps_p': f"{ft_['self_steps_min']['holm_p']:.2f}",
              'f_heel_e': f"{fv('heel_lift', 'mcg_early'):.2f}", 'f_heel_l': f"{fv('heel_lift', 'mcg_late'):.2f}",
              'f_bounce_e': f"{fv('bounce_cm', 'mcg_early'):.1f}", 'f_bounce_l': f"{fv('bounce_cm', 'mcg_late'):.1f}",
              'f_str_e': f"{fv('strikes_min', 'mcg_early'):.0f}", 'f_str_l': f"{fv('strikes_min', 'mcg_late'):.0f}", 'f_str_oe': f"{fv('strikes_min', 'opp_early'):.0f}", 'f_str_ol': f"{fv('strikes_min', 'opp_late'):.0f}",
              'f_w_e': f"{fv('width_cm', 'mcg_early'):.0f}", 'f_w_l': f"{fv('width_cm', 'mcg_late'):.0f}", 'f_w_down': f"{ft_['self_width_cm']['down_in']} of {ft_['self_width_cm']['n']}", 'f_w_p': f"{ft_['self_width_cm']['p']:.2f}", 'f_w_h': f"{ft_['self_width_cm']['holm_p']:.2f}",
              'f_w_oe': f"{fv('width_cm', 'opp_early'):.0f}", 'f_w_ol': f"{fv('width_cm', 'opp_late'):.0f}",
              's_guard': f"{abs(fs_['self_guard_cm']['slope_per_5min']):.0f}", 's_guard_n': f"{fs_['self_guard_cm']['down_in']} of {fs_['self_guard_cm']['n']}", 's_guard_p': f"{fs_['self_guard_cm']['p']:.2f}", 's_guard_h': f"{fs_['self_guard_cm']['holm_p']:.2f}",
              's_width': f"{abs(fs_['self_width_cm']['slope_per_5min']):.0f}", 's_width_n': f"{fs_['self_width_cm']['down_in']} of {fs_['self_width_cm']['n']}", 's_width_p': f"{fs_['self_width_cm']['p']:.2f}",
              's_vguard': f"{abs(fs_['vs_guard_cm']['slope_per_5min']):.0f}", 's_vguard_p': f"{fs_['vs_guard_cm']['p']:.2f}",
              's_vwidth': f"{fs_['vs_width_cm']['slope_per_5min']:.0f}", 's_steps': f"{fs_['self_steps_min']['slope_per_5min']:.0f}", 's_steps_n': f"{fs_['self_steps_min']['down_in']} of {fs_['self_steps_min']['n']}",
              's_n': str(fs_['self_steps_min']['n'])}
    syl_html = (HERE / 'site_syllables.html').read_text() if (HERE / 'site_syllables.html').exists() else '<p class="lede">Fitting.</p>'

    fill = {
        'grid': grid, 'head_chart': head_chart, 'guard_chart': guard_chart, 'nulls': str(nulls_tested),
        'head_m': f"{hs['mcg']:.0f}", 'head_o': f"{hs['opp']:.0f}", 'head_n': f"{hs['his_higher_in']} of {hs['n']}",
        'guard_m': f"{-gs['mcg']:.0f}", 'guard_o': f"{-gs['opp']:.0f}", 'guard_n': f"{gs['n'] - gs['his_higher_in']} of {gs['n']}",
        'reach_m': f"{reach['mcg']:.0f}", 'reach_o': f"{reach['opp']:.0f}",
        'stance_pairs': stance_pairs_html, 'left_pairs': left_pairs_html, 'vs_pairs': vs_html, 'rates': rates,
        'rate_m': f"{lr['mcg']:.1f}", 'rate_o': f"{lr['opp']:.1f}", 'rate_n': f"{lr['his_higher_in']} of {lr['n']}", 'rate_x': f"{lr['mcg'] / lr['opp']:.0f}",
        'rear_m': f"{rs['mcg']:.0%}", 'rear_o': f"{rs['opp']:.0%}", 'rear_n': f"{rs['his_higher_in']} of {rs['n']}",
        'still_m': f"{bh['mcg']:.0%}", 'still_o': f"{bh['opp']:.0%}", 'still_p': f"{bh['holm_p']:.2f}", 'still_base': f"{sl['baseline']['big_move_share']:.0%}",
        'speed_m': f"{sp_m:.1f}", 'speed_o': f"{sp_o:.1f}", 'head_paths': head_paths(sl), 'torso_cm': f"{tc:.0f}",
        'n_lefts': str(sum(e['who'] == 'mcg' for e in speeds)), 'n_opp': str(sum(e['who'] == 'opp' for e in speeds)),
        'syllables': syl_html,
        **fill_f, **fill_b, **fill_n,
        'moved': moved_html, 'drops': drops_html, 'hunt_one': hunt_one, 'late_grid': late_grid,
        'ctr_m': f"{ca_ / len(evd('mcg')):.0%}", 'ctr_o': f"{cb_ / len(evd('opp')):.0%}", 'ctr_note': f"{ca_} of {len(evd('mcg'))} against {cb_} of {len(evd('opp'))} · p {ctr_p:.2f}",
        'ev_pull_m': f"{po_['mcg']['saved_by_pull']:.0%}", 'ev_pull_o': f"{po_['opp']['saved_by_pull']:.0%}", 'ev_pull_n': f"{po_['mcg']['n_pull_would_land']} and {po_['opp']['n_pull_would_land']} pulls · no reliable difference",
        'ev_gap_m': f"{po_['mcg']['saved_margin_median']:.0f}", 'ev_gap_o': f"{po_['opp']['saved_margin_median']:.0f}", 'ev_need': f"{po_['mcg']['saved_needed_median']:.0f}", 'ev_trav': f"{po_['mcg']['saved_travel_median']:.0f}",
        'blade_strip': '<video class="mstrip" src="clips/measure_blade.mp4#t=0.1" muted loop playsinline preload="metadata" aria-label="His shoulders in each era"></video>',
        'map_knn': f"{json.loads((RES / 'syllables' / 'syllables_site.json').read_text()).get('map_knn', 0):.0%}", 'map_chance': f"{json.loads((RES / 'syllables' / 'syllables_site.json').read_text()).get('map_chance', 0):.0%}",
        'rear_chart': rear_chart, 'still_chart': still_chart, 'lead_chart': lead_chart, 'hunt_grid': hunt_grid, 'counters_late': counters_late,
        'mix_chart': mix_chart(mix), 'counter_pairs': counter_html, 'counter_traces': traces_svg,
        'lead_m': f"{lm['mcg'] * tc:.0f}", 'lead_o': f"{lm['opp'] * tc:.0f}", 'lead_n': f"{lm['n'] - lm['his_higher_in']} of {lm['n']}",
        'posted_m': {0: 'never', 1: 'once'}.get(round(lf['_posted']['mcg'] * mix['mcg']['n']), f"{round(lf['_posted']['mcg'] * mix['mcg']['n'])} times"),
        'posted_o': {0: 'never', 1: 'once'}.get(round(lf['_posted']['opp'] * mix['opp']['n']), f"{round(lf['_posted']['opp'] * mix['opp']['n'])} times"),
        'paw_m': f"{mix['mcg']['behind the paw'] / mix['mcg']['n']:.0%}", 'paw_o': f"{mix['opp']['behind the paw'] / mix['opp']['n']:.0%}",
        'jab_m': f"{lf['jab_before']['mcg']:.0%}", 'jab_o': f"{lf['jab_before']['opp']:.0%}", 'jab_p': f"{lf['jab_before']['holm_p']:.2f}",
        'jabr_m': f"{lf['jab_rate']['mcg']:.1f}", 'jabr_o': f"{lf['jab_rate']['opp']:.1f}", 'jabr_p': f"{lf['jab_rate']['holm_p']:.2f}",
        'still_n': f"{bh['n'] - bh['his_higher_in']} of {bh['n']}",
        'n_mcg_lefts': str(mix['mcg']['n']), 'n_opp_lefts': str(mix['opp']['n']),
        'mix_early_c': str(mix['early']['pull counter'] + mix['early']['slip counter']), 'mix_early_n': str(mix['early']['n']),
        'mix_late_c': str(mix['late']['pull counter'] + mix['late']['slip counter']), 'mix_late_n': str(mix['late']['n']),
        'cs_e': f"{100 * cs['early']:.0f}", 'cs_l': f"{100 * cs['late']:.0f}", 'cs_p': f"{cs['p']:.2f}", 'cs_h': f"{cs['holm_p']:.2f}",
        'hook_e': f"{lf['era_left_hook']['early']:.1f}", 'hook_l': f"{lf['era_left_hook']['late']:.1f}", 'hook_p': f"{lf['era_left_hook']['holm_p']:.2f}",
        'pull_back': f"{sets['pull counter']['head_back'] * tc:.0f}", 'pull_ms': f"{lf['counter_timing']['pull_ms']:.0f}",
        'slip_down': f"{sets['slip counter']['head_down'] * tc:.0f}", 'slip_ms': f"{lf['counter_timing']['slip_ms']:.0f}",
        'n_pull': str(n_pull), 'n_slip': str(n_slip), 'ct_p': f"{lf['counter_timing']['p']:.2f}",
        'n_era_tests': str(sum(1 for v in lf.values() if isinstance(v, dict) and v.get('family') == 'era')),
        'n_sig_tests': str(sum(1 for v in lf.values() if isinstance(v, dict) and v.get('family') == 'sig')),
    }
    t = (HERE / 'site_template.html').read_text()
    for k, v in fill.items():
        t = t.replace('{{' + k + '}}', v)
    (SITE / 'index.html').write_text(t)
    print('wrote', SITE / 'index.html', sum(f.stat().st_size for f in SITE.rglob('*') if f.is_file()) / 1e6, 'MB')


if __name__ == '__main__':
    main()
