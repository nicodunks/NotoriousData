"""Ghost Fight, full clips: one round per tracked source clip, covering the whole clip, so the drawing and the
revealed footage are the same length and stay in step. Same identity and keypoint rules as build_game.py
(clothing-role identities only; keypoints with raw score >= 2 inside the frame). Reuses the footage files
build_game.py bundled; a source without bundled footage is left out.

    python3 work/game/build_full.py        (from mma/)
"""
import json
from pathlib import Path

R = Path.cwd()
O = R / 'outputs/game'
B = R / 'outputs/expanded/mcgregor/tracked'
SPECS = [  # source, pose, identities, roles, year, insight
    ('A', R/'outputs/experiment/body_data/A_rtmw_133.json', R/'outputs/expanded/identity/A-identities.json',
     {'fighter_green': 'Alex Pereira', 'fighter_yellow': 'Israel Adesanya'}, None, 'Long-limbed standing exchanges: watch how each manages the space.'),
    ('E', R/'outputs/expanded/body_data/E_rtmw_133.json', R/'outputs/expanded/identity/E-identities.json',
     {'fighter_cyan': 'Alex Pereira', 'fighter_coral': 'Israel Adesanya'}, None, 'A low kick to the lead leg, then a round kick into the raised guard.'),
    ('B', R/'outputs/experiment/body_data/B_rtmw_133.json', R/'outputs/expanded/identity/B-identities.json',
     {'fighter_green': 'Conor McGregor', 'fighter_black': 'Khabib Nurmagomedov'}, 2018, 'A straight punch leads into a kick toward the body.'),
    ('P', B/'body_data/P_rtmw_133.json', B/'identity/P-identities.json',
     {'fighter_cyan': 'Conor McGregor', 'fighter_coral': 'Ivan Buchinger'}, 2012, 'Body kicks, fast retraction, and distance resets.'),
    ('Q', B/'body_data/Q_rtmw_133.json', B/'identity/Q-identities.json',
     {'fighter_cyan': 'Conor McGregor', 'fighter_coral': 'Max Holloway'}, 2013, 'A shuffle, a long lunging punch, then stance recovery.'),
    ('R', B/'body_data/R_rtmw_133.json', B/'identity/R-identities.json',
     {'fighter_cyan': 'Conor McGregor', 'fighter_coral': 'Khabib Nurmagomedov'}, 2018, 'Short repositioning and hand probing before a lifted kick.'),
    ('S', B/'body_data/S_rtmw_133.json', B/'identity/S-identities.json',
     {'fighter_cyan': 'Conor McGregor', 'fighter_coral': 'Dustin Poirier'}, 2021, 'Punch-led pursuit with repeated forward steps toward the fence.'),
]

# Names as the game shows them (short enough for one row of chips).
SHORT = {'Khabib Nurmagomedov': 'Khabib N.'}

# Left out after review: A's close camera crops the legs and the drawing distorts (torsos balloon, shins vanish).
DROP = {'A'}

old = json.loads((O/'manifest.json').read_text())
footage = {}
for r in old['rounds']:
    footage.setdefault(r['source_id'], r['video'])

def pose_dist(a, b):
    """Mean distance between the body joints both poses have; None if they share none."""
    d = [((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2) ** .5 for i in range(17) if a[i] and b[i]]
    return sum(d) / len(d) if d else None


def keep_identity(frames, width):
    """Fighters can't change colour mid-clip. The clothing classifier labels each frame on its own and can swap
    the two during occlusion, so each body keeps the colour of whichever fighter it continues from the previous
    frame. Where continuity can't be judged (a camera cut: nothing close), the clothing label stands. Finally the
    clothing labels decide, by majority, which continuous track is which fighter."""
    original = [[p['role'] for p in f['people']] for f in frames]
    last, swaps, near = {}, 0, .15 * width
    for f in frames:
        people = f['people']
        def cost(roles):
            ds = [pose_dist(p['points'], last[r]) for p, r in zip(people, roles) if r in last]
            ds = [x for x in ds if x is not None]
            return sum(ds) / len(ds) if ds else None
        keep = [p['role'] for p in people]; swap = [1 - r for r in keep]
        ck, cs = cost(keep), cost(swap)
        if ck is not None and cs is not None and cs < near and cs < .6 * ck:
            for p, r in zip(people, swap): p['role'] = r
            swaps += 1
        for p in people: last[p['role']] = p['points']
    agree = sum(p['role'] == r for f, o in zip(frames, original) for p, r in zip(f['people'], o))
    total = sum(len(o) for o in original)
    if total and agree < total / 2:
        for f in frames:
            for p in f['people']: p['role'] = 1 - p['role']
    return swaps


def tracked_span(frames, gap=1.5):
    """The longest stretch in which both fighters keep being tracked: split wherever neither pair is seen together
    for more than `gap` seconds (a takedown into tangled bodies, say), keep the longest piece."""
    both = [f['t'] for f in frames if len({p['role'] for p in f['people']}) == 2]
    if not both: return frames[0]['t'], frames[-1]['t']
    spans, start, prev = [], both[0], both[0]
    for t in both[1:]:
        if t - prev > gap: spans.append((start, prev)); start = t
        prev = t
    spans.append((start, prev))
    return max(spans, key=lambda s: s[1] - s[0])


rounds = []
for cid, posepath, idpath, names, year, insight in SPECS:
    if cid in DROP:
        print('skip', cid, '(dropped after review)'); continue
    if cid not in footage:
        print('skip', cid, '(no bundled footage)'); continue
    p = json.loads(posepath.read_text()); ids = json.loads(idpath.read_text()); keys = list(names)
    frames, allxy, both = [], [], 0
    for j, f in enumerate(p['frames']):
        people = []
        for z in ids['frames'][j]['people']:
            if z['identity'] not in keys: continue
            k = f['people'][z['detection_index']]['keypoints']
            pts = [[round(x, 1), round(y, 1)] if sc >= 2 and 0 <= x < p['width'] and 0 <= y < p['height'] else None for x, y, sc in k]
            allxy.extend(q for q in pts[:23] if q)
            people.append({'role': keys.index(z['identity']), 'points': pts})
        both += len({q['role'] for q in people}) == 2
        frames.append({'t': round(f['time'], 4), 'people': people})
    swaps = keep_identity(frames, p['width'])
    t0, t1 = tracked_span(frames)
    full = p['frames'][-1]['time'] + 1 / p['fps']
    frames = [{**f, 't': round(f['t'] - t0, 4)} for f in frames if t0 <= f['t'] <= t1]
    allxy = [q for f in frames for person in f['people'] for q in person['points'][:23] if q]
    both = sum(len({q['role'] for q in f['people']}) == 2 for f in frames)
    xs, ys = zip(*allxy)
    duration = round(t1 - t0 + 1 / p['fps'], 4)
    motion = {'width': p['width'], 'height': p['height'], 'duration': duration, 'fps': p['fps'],
              'bounds': [max(0, min(xs) - 30), max(0, min(ys) - 30), min(p['width'], max(xs) + 30), min(p['height'], max(ys) + 30)],
              'frames': frames}
    (O/'motion'/f'{cid}.json').write_text(json.dumps(motion, separators=(',', ':')))
    rounds.append({'id': cid, 'motion': f'motion/{cid}.json', 'names': [SHORT.get(names[k], names[k]) for k in keys], 'video': footage[cid],
                   'start': round(t0, 4), 'duration': duration, 'year': year, 'insight': insight, 'source_id': cid,
                   'both_fighters_coverage': round(both / len(frames), 3)})
    print(cid, f'{duration:.1f}s', f'{len(frames)} frames', 'both fighters in', f'{both/len(frames):.0%}', f'{swaps} frames re-coloured for continuity', f'kept {t0:.1f}-{t1 + 1 / p["fps"]:.1f}s of {full:.1f}s')

manifest = {**old, 'rounds': rounds, 'roster': sorted({n for r in rounds for n in r['names']})}
(O/'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
