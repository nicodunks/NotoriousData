"""SVG pieces for the report, drawn from the results (no plotting library: the page ships its own markup).

Every chart is drawn to one scale per axis; colours come from the page's CSS tokens (classes, not literals).
"""
from __future__ import annotations

import html
from datetime import date

import numpy as np

# skeleton bones over the 23 body+feet points (COCO-WholeBody)
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
         (14, 16), (15, 19), (19, 17), (16, 22), (22, 20)]
T0, T1 = date(2011, 6, 1), date(2021, 12, 31)


def esc(s):
    return html.escape(str(s), quote=True)


def skeleton(pts: np.ndarray, cls: str, scale: float, ox: float, oy: float, w=2.2, dot=2.6, head=True) -> str:
    """Pen-style skeleton from canonical points (hips at 0, torso 1, opponent to the right)."""
    P = lambda i: (ox + pts[i, 0] * scale, oy + pts[i, 1] * scale)
    out = []
    for a, b in BONES:
        if np.isfinite(pts[[a, b]]).all():
            (x1, y1), (x2, y2) = P(a), P(b)
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="{w}"/>')
    if head:
        face = [i for i in range(5) if np.isfinite(pts[i]).all()]
        if len(face) >= 2:
            c = pts[face].mean(0); r = max(.16, np.linalg.norm(pts[face] - c, axis=1).max() * 1.45)
            cx, cy = ox + c[0] * scale, oy + c[1] * scale
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * scale:.1f}" class="{cls}" fill="none" stroke-width="{w}"/>')
            if np.isfinite(pts[[5, 6]]).all():
                nb = (pts[5] + pts[6]) / 2
                d = nb - c; dl = np.linalg.norm(d)
                if dl > r:
                    e = c + d / dl * r
                    out.append(f'<line x1="{ox + e[0] * scale:.1f}" y1="{oy + e[1] * scale:.1f}" x2="{ox + nb[0] * scale:.1f}" y2="{oy + nb[1] * scale:.1f}" class="{cls} faint" stroke-width="{w * .6}"/>')
    for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        if np.isfinite(pts[i]).all():
            x, y = P(i); out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot}" class="{cls} dot"/>')
    return ''.join(out)


def xdate(d: str, x0: float, x1: float) -> float:
    dd = date.fromisoformat(d)
    return x0 + (dd - T0).days / (T1 - T0).days * (x1 - x0)


def timeline(fights, key: str, label: str, unit: str, split: str, fmt='{:.2f}', width=560, height=250, invert=False) -> str:
    """One quantity across the career: McGregor (accent) and opponents (grey), dot = fight, whisker = 95 % CI."""
    L, R, T, B = 52, 16, 16, 34
    x0, x1, y0, y1 = L, width - R, T, height - B
    vals = []
    for f in fights:
        for who in ('mcg', 'opp'):
            v = f.get(who, {}).get(key); ci = f.get(f'{who}_ci', {}).get(key, [np.nan, np.nan])
            vals += [v] + list(ci)
    vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    pad = (hi - lo) * .12 or .1; lo, hi = lo - pad, hi + pad
    # nice ticks
    span = hi - lo; step = 10 ** np.floor(np.log10(span / 4));
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= 5: step *= m; break
    ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    Y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(label)} per fight, McGregor and opponents">']
    for tv in ticks:
        y = Y(tv); out.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{x0 - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt.format(tv)}</text>')
    for yr in range(2012, 2022, 2):
        x = xdate(f'{yr}-01-01', x0, x1)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y1}" y2="{y1 + 5}" class="axis"/><text x="{x:.1f}" y="{y1 + 19}" class="tick" text-anchor="middle">{yr}</text>')
    xs = xdate(split, x0, x1)
    out.append(f'<line x1="{xs:.1f}" x2="{xs:.1f}" y1="{y0}" y2="{y1}" class="split"/><text x="{xs + 5:.1f}" y="{y0 + 10}" class="tick note">Diaz 1</text>')
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" class="axis"/>')
    # trend lines (Theil-Sen: robust to one odd fight), drawn under the dots across the span of each fighter's fights
    from scipy.stats import theilslopes
    for who, cls in (('opp', 'opp'), ('mcg', 'mcg')):
        xy = [(xdate(f['date'], x0, x1), f.get(who, {}).get(key)) for f in fights]
        xy = [(x, v) for x, v in xy if v is not None and np.isfinite(v)]
        if len(xy) >= 4:
            xs_, vs_ = np.array(xy).T
            sl, ic, _, _ = theilslopes(vs_, xs_)
            xa, xb = xs_.min(), xs_.max()
            out.append(f'<line x1="{xa:.1f}" x2="{xb:.1f}" y1="{Y(ic + sl * xa):.1f}" y2="{Y(ic + sl * xb):.1f}" class="trend {cls}"/>')
    for who, cls, dx in (('opp', 'opp', 3.5), ('mcg', 'mcg', -3.5)):
        pts = []
        for f in fights:
            v = f.get(who, {}).get(key)
            if v is None or not np.isfinite(v): continue
            x = xdate(f['date'], x0, x1) + dx; y = Y(v)
            ci = f.get(f'{who}_ci', {}).get(key, [np.nan, np.nan])
            if np.isfinite(ci).all():
                out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(max(min(ci[0], hi), lo)):.1f}" y2="{Y(max(min(ci[1], hi), lo)):.1f}" class="{cls} whisker"/>')
            pts.append((x, y))
            who_name = 'McGregor' if who == 'mcg' else f['opponent']
            tip = f"{who_name}, {f['date'][:4]} vs {f['opponent'] if who == 'mcg' else 'McGregor'}: {fmt.format(v)}{unit}"
            if np.isfinite(ci).all(): tip += f" (95% CI {fmt.format(ci[0])} to {fmt.format(ci[1])})"
            r = 4.5 if who == 'mcg' else 3.5
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls} pt"><title>{esc(tip)}</title></circle>')
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" class="hit"><title>{esc(tip)}</title></circle>')
    out.append('</svg>')
    return ''.join(out)
