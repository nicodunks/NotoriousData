# McGregor, measured

Every McGregor MMA fight with a public upload → his body, frame by frame → a pre-registered test of the story
that he went from a wide, bouncing karate stance to a flat-footed boxer. Off-the-shelf models only; no training.
Built after Peter Wang's tennis pipeline (`docs/PIPELINE.md`): the same shape (keep live play, find the people,
name them, measure, compare years), with MMA's problems solved differently.

Read first: [PREREGISTRATION.md](PREREGISTRATION.md) (what was predicted, before measuring, and every change
since). The write-up is `results/report.html`; the numbers are `results/results.json`.

## Steps

| # | Step | Code | Notes |
|---|---|---|---|
| 0 | Get the fights | `sources.json`, `download.sh` | 16 uploads (≤ 720p, video only), official where they exist |
| 1 | Find the fight clock | `board.py`, `board_check.py`, `live.py` | live play = clock on screen; drops walkouts, replays, interviews |
| 2 | Pose | `pose.py`, `run_queue.sh` | UFC-trained fighter detector + RTMW-x (133 points), 10 frames/s, live frames only |
| 3 | Who is who | `identity.py`, `windows.json` | shorts colour, prototypes named by eye once per fight |
| 4 | Keep the clean frames | `frames.py` | two fighters, identity held through occlusion, both upright and apart |
| 5 | Measure | `metrics.py`, `stance.py` | ratios of body lengths, so zoom cancels; opponent measured in the same frames |
| 6 | Test | `study.py` | per-fight medians, block bootstrap, trend vs date, Holm, opponents as control |
| 7 | Check by eye | `qa_feet.py`, `qa_kicks.py` | the pre-registered visual checks |
| 8 | Write up | `report.py`, `report_template.html`, `report_text.json` | one page, every number from the results |

Run from `work/` with `venv/bin/python deep-sports-analysis/mma/study/mcgregor/<step>.py`. Data lives in
`work/mcgregor/` (`raw/`, `board/`, `pose/`, `identity/`), outside the repo.

## What each step does

0. **Get the fights.** One upload per fight; the Cage Warriors fights come from UFC Fight Pass's official
   compilation, cut into fights at the "McGREGOR vs …" title cards (`span` in `windows.json`).
1. **Find the clock.** The fight clock is the overlay whose edges stay at the same pixels from one second to
   the next *while two fighters are on screen* (the detector marks those seconds). Floor logos slide with the
   camera, crowds flicker, name banners come with walkouts, so none of them qualify. A frame is live where at
   least 55 % of the clock's own full-strength edges are present. Two feeds have no clock: Poirier 1 (Spanish
   feed; window set by eye) and Aldo (only a replay package exists; excluded).
2. **Pose.** Every live frame, 10 per second: the UFC-trained fighter detector (fight-judge YOLOv8s) finds up
   to three people; RTMW-x gives 133 points each (body, feet, hands, face). Run on Apple's CoreML: same
   keypoints as the CPU (max 1 px), ~45× faster, ~25 frames/s per process. Several processes at once thrash;
   three lanes is the sweet spot.
3. **Who is who.** Shorts colour (median Lab colour of the upper thigh), clustered per fight and named by eye
   from a strip of sample crops; the named cluster centres are the fight's prototypes. A pair of detections is
   labelled McGregor + opponent only when their colours clearly say so.
4. **Clean frames.** Inside a camera shot, a body keeps the identity of the fighter it continues from (the
   same rule that stopped Ghost Fight's colours swapping during occlusion). Points within 1.5 % of the frame
   edge count as unseen. "Standing": both fighters' hips above their knees, torsos within ~45° of vertical,
   and their hips more than 1.1 torso lengths apart.
5. **Measure.** Stance width and length, which foot leads, heel lift, guard height, lead-hand reach, crouch,
   lean, range, kicks, stance switches, bounce (hip motion at 1.5–4 Hz) and its rhythm, footwork speed and
   direction. Every length is divided by the fighter's torso length in that shot.
6. **Test.** Unit = fight. Trend = Spearman ρ against fight date with a permutation p; the primary series is
   McGregor minus opponent in the same frames (a camera or division change moves both). Secondary: before vs
   after Diaz 1. Holm-corrected across the seven claims. Robustness: 30 s standing minimum, official uploads
   only.

## Known problems

- **Camera angle** changes what a 2-D stance width looks like; ratios fix zoom, not angle. The opponent in the
  same frames is the control, and it matters (see the report).
- **Short fights** give little standing footage (Cerrone 15 s, O'Keeffe 15 s); fights under 60 s are shown but
  not tested.
- **Jahnsen (2011)**: both fighters in black shorts, so identity by colour is impossible; excluded.
- **Stance side** is right about 80–90 % of the time (checked against opponents' known stances); stance
  switch counts therefore have a noise floor of ~1–3 per minute.
- **Kicks**: about three in four counted events are real kicks or knees (checked by eye).
- **Heel lift** mixes in the direction the foot points relative to the camera; read it against the opponent.

## The writeup (`build_site.py`)

`python findings.py && python leftfindings.py && python pairs.py && python syllables.py build && python build_site.py`
writes `results/site/` (page from `site_template.html` + `site_syllables.html`, clips, JSON). Tabs: karate
stance (signatures vs opponents), stance evolution (era tests), the left hand, syllables, methods. Only findings
passing the rule in PREREGISTRATION #17 are stated as findings.
