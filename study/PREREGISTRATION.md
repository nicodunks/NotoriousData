# McGregor, 2011–2021: pre-registration

Written 2026-10-02, before any pose was run on the study footage. Changes after this point are listed at the
bottom with the reason, so what was planned and what was found stay distinguishable.

## The claim under test

The popular story: early McGregor fought from a wide, karate-style stance, light on his toes, bouncing, kicking
and switching stances; later McGregor became a boxer: narrower, flatter-footed, hands higher, fewer kicks.

## Hypotheses (direction stated in advance)

| # | Quantity | Prediction (early → late) |
|---|---|---|
| H1 | Stance width: ankle-to-ankle distance ÷ his own leg length | narrows |
| H2 | On his toes: share of standing frames with a heel clearly above the big toe | falls |
| H3 | Bounce: hip vertical motion in the 1.5–4 Hz band ÷ torso length | falls |
| H4 | Guard: wrist height relative to the shoulder line, ÷ torso length | rises (hands higher) |
| H5 | Kicks per standing minute | fall |
| H6 | Southpaw ↔ orthodox switches per standing minute | fall |
| H7 | Range: distance between the two fighters' hips ÷ summed leg lengths | shortens |

"Early" and "late" are not chosen from the data: the primary test is a trend against fight date over all
fights; the secondary test splits at the first Diaz fight (2016-03-05), the point the story usually names.

## Measurement

- Footage: every McGregor MMA fight with a public upload (sources.json), ≤ 720p.
- Frames: 10 per second. Wide shots only: both fighters detected, each fighter's box 20–80 % of frame height
  (drops close-ups and crowd shots).
- Models (off the shelf, no training): UFC-trained fighter detector (fight-judge YOLOv8s) for boxes; RTMW-x
  (133 whole-body points: body, feet, hands, face) for pose.
- Identity: McGregor vs opponent by shorts colour, one colour prototype per fight set by hand from a contact
  sheet, then held fixed; frames where the shorts don't clearly match are dropped, not guessed.
- Standing only: both fighters' hips above their knees and a gap between them (no clinch, no ground).
- Every quantity is a ratio of body lengths, so camera zoom cancels. Camera angle does not cancel; see controls.

## Inference

- Unit of analysis: the fight (not the frame — frames inside a fight are not independent).
- Per fight: median of the quantity over standing frames; 95 % interval by block bootstrap (10 s blocks).
- Trend: Spearman's ρ between fight date and the per-fight median; permutation p (10 000 shuffles).
- Split: late − early difference of fight medians; 95 % interval by bootstrap over fights.
- Seven hypotheses: results reported with Holm-corrected p alongside raw p.

## Controls (what would fool us)

1. Opponents as a ruler. The same quantities for his opponents, measured in the same frames. Broadcast style,
   camera height and lens changed from Cage Warriors to UFC; if the opponents "change" the same way he does,
   the change is the camera, not McGregor. The headline numbers are McGregor minus opponent, same frames.
2. Weight class. Featherweight (to 2015) vs lightweight / welterweight (2016 on) changes opponents and body
   size; ratios handle size, opponents-as-ruler handles style of the division.
3. Short fights. Aldo (13 s) and Cerrone (40 s) give little standing time; fights under 60 s of usable
   standing frames are shown but excluded from the tests.
4. Fan uploads (Siver, Diaz 1) may be re-encoded or cropped; reported with and without them.
5. Model blind spots. Feet and heels are the least reliable points; H2 uses only frames where both foot
   points of a foot have high scores, and is checked by eye on a sample of frames per fight.

## Changes after registration

All made after running one fight (Brimage, 2013) end to end, before any other fight's numbers existed.

1. **Live play** is the fight clock on screen (detected per broadcast), not hand-set windows: it also drops
   slow-motion replays, which would have faked low bounce. Two feeds have no clock (Poirier 1, Spanish feed:
   window set by eye; Aldo: the only upload is a replay package, so that fight is excluded).
2. **Wide shots**: "box 20-80 % of frame height" threw out most usable UFC frames (fighters fill ~80 % of a
   720p frame). Replaced by: two fighter-sized detections (box >= 25 % of frame height, head in frame), with
   any point within 1.5 % of the frame border treated as unseen, so a cut-off foot never enters a stance
   measurement while the same fighter's guard and range still count. (Requiring the whole body in frame kept
   only 169 of 749 live Brimage frames: broadcast framing often puts the near fighter's feet at the edge.)
3. **H2 (on the toes)**: the yes/no "heel above toe" was 100 % for both fighters, because a camera above the
   cage lifts every heel in the image. Replaced by a continuous heel-lift index (toe minus heel height, per foot
   length), median per fight; the camera's share cancels in the pre-registered McGregor - opponent difference.
4. **H3 (bounce)**: standing stretches inside one camera shot are mostly 2-3 s, so the minimum run is 2 s
   (20 samples at 10 per second) instead of 3 s; band unchanged (1.5-4 Hz).
5. **Pose ran on Apple's CoreML** instead of the CPU: same model, keypoints within 1 px (median 0), ~45x faster.
6. **Two robustness runs**, declared before any cross-fight result existed: the same tests with a 30 s
   standing minimum (the quick knockouts, Brimage, Poirier 1 and Cerrone, have 15-53 s of standing footage
   and fall under the 60 s rule), and with official uploads only (drops Siver and Diaz 1).
7. **H7 (range)** is one distance shared by the two fighters, so McGregor minus opponent is zero by
   construction; its primary test uses McGregor's own value (a design slip, caught on the first full run).
8. **A counting bug, fixed**: the stance-side share divided by all frames instead of frames where the stance
   was read, which made orthodox opponents read ~50 % orthodox. Found by checking the instrument against
   opponents whose stance is public knowledge (see "Does the instrument see what we already know?").
9. **Standing** also needs an upright torso (shoulders above hips by > 0.7 torso, i.e. within ~45 degrees of
   vertical): the kick check below found ground grappling passing the hips-above-knees test.
10. **H5 (kicks)**: the pre-registered by-eye check of counted kicks found about half were steps, pivots or
   grappling. A kick now needs the ankle above the standing leg's knee and the leg extended >= 0.75 of its
   length from the hip, held >= 2 frames (0.2 s).

## The left-hand study (added after the style study was published)

11. **Punch detector calibrated by eye** on 43 Khabib candidates before any other fight was examined: early
    thresholds counted paws and resets; elbow angle proved unreliable under foreshortening, so straight vs hook
    is read from the wrist path (displacement >= 75 % of distance travelled = straight).
12. **Frame-rate bug, fixed**: the Siver upload is 25 fps; the code assumed 30, so every punch window overran
    its time limit and Siver read zero punches. Sampling rate is now read from each file's timestamps.

## Exploratory additions (2026-10-03), not pre-registered

13. **Stance in depth** (`stance_deep.py`): 15 measures, each tested for him, for the opponents, for him minus
    them, and again on side-on shots only. Treated as exploratory: with 15 measures a p < 0.05 is expected by
    chance, so a change is called his only if it holds for him, holds side-on, and isn't shared by opponents.
14. **Feet** (`feet.py`): heel lift, ankle bounce and planted time from heel/toe/ankle points in any view. The
    "planted" stillness threshold was raised from 0.25 to 0.6 torso lengths/s after the first run read ~0 %
    for everyone (pose jitter alone exceeds 0.25).
15. **Before the left** (`slips.py`, `leadhand.py`): head/torso movement relative to the hips in the 0.8 s
    before each left, against a no-punch baseline of 644 random windows and against opponents' rear straights.
16. **Strike selection and who starts exchanges** (`selection.py`, `pressure.py`): per-fight rates and shares,
    opponents as baseline. "Moving forward" depends on the camera pan and is the weaker measure.
17. **Reporting rule for the writeup** (`findings.py`, `leftfindings.py`), after Peter Wang's per-stroke tests: a
    finding is shown only if Holm-corrected p < 0.05 across its family and the effect is at least 0.4 of the
    frame-to-frame spread. Era tests: Welch on per-fight values, 2016-21 vs 2012-15. Signatures: Wilcoxon,
    McGregor vs opponent, paired by fight. Units in cm from his height (175 cm; torso ~47 cm).
18. **Syllables** (`moseq_prep.py`, `moseq_fit.py`, `syllables.py`): keypoint-MoSeq 0.6.6 on 17 keypoints,
    side-on shots only (size ratio .8-1.25, separation > 1.2 torso), mirrored so the opponent is on the right.
    kappa 1e8 (AR) / 1e7 (full); syllables > 1 % of frames kept; per-fight usage, era Welch + Holm.
    Names given by hand after watching clips. A first fit on all views mostly encoded camera angle (discarded).
19. **Left-hand families, set at the rebuild (after seeing results; disclosed on the page):** sig (6 tests: rate,
    rear share, jab rate, jab before, big head move, lead-hand travel), era (9), counter timing (1); Holm within each.
    Under one family of 16 only the rate passes (rear share .07, still head .14, lead hand .21). `lead_motion` now
    measured relative to the lead shoulder (was image space, which mixed in footwork and camera moves).
20. **Shoulder tilt** added to the stance family on request (lead shoulder above rear, cm): no era change, no
    signature (him -1.7 vs -0.3 cm, Holm .52). Measure clips (`pairs.py measures`) use one typical late window, and
    for the shoulder the late window closest to his median tilt.
21. **Late rounds** (`fatigue.py`), rules written in the script docstring before results: fight clock = cumulative
    live play rescaled to official length; fights past 9 min; round 1 vs minute 8 on; >= 10 s standing per window;
    steps from the inter-ankle vector (pan-invariant; QA image results/qa/steps_diaz2.jpg); one-sample t over fights,
    Holm within 'self' and 'vs' families (8 measures each). Null. The whole-fight slope test (30-s chunks, 8 fights)
    was added after the null and is labelled exploratory: guard -3 cm/5 min (6/8, p .03, Holm .23), width -5 cm/5 min
    (7/8, p .04; shared with opponents). Diaz 2 shown minute by minute as a single-fight picture, not a test.
22. **Blade** (shoulders turned side-on, degrees, side-on shots only) added to the stance family at the reader's
    suggestion: no era change for him (37 -> 35, Holm 1.0); his margin over opponents 9 deg before Diaz -> -1 deg
    after (Welch p .038, one test, uncorrected, reader's hypothesis stated before we looked): opponents turned side-on.
23. **Syllable map**: t-SNE of per-occurrence MoSeq latent features (syl_embed.py); 10-NN syllable agreement 45 %
    (chance 20 %): McGregor's syllables overlap. Clips: 10 per syllable placed by the map (4 central + 6 spread).
24. **Spot McGregor** (`game.py`): one round per fight, best-landing clean left; 3 rounds mirrored to balance sides.
25. **Knockout reel**: knockdowns/KOs found by scanning footage at each fight's end and by an opponent-hip-drop
    screen, confirmed by eye (Buchinger 737.9 s, Poirier 2014 115.4 s, Mendes 793.9 s, Alvarez 177.8 s). Speed signs
    only where the punch crosses the picture plane and is tracked.
26. **Distance management** (`evasion.py`, rules in docstring before running): the pull's clearance and economy are
    no different from opponents' (75 vs 73 % of would-land punches evaded; 7 vs 7 cm clearance; ~12 cm pulled for
    ~3 cm needed). After evading, he fired back within 0.6 s 11/75 vs 4/89 (Fisher p .03; fight-bootstrap CI -3 to
    +19 points). Stance tab reorganised around 2012-15 vs the 2021 Poirier fights at the reader's request; the page
    says these four shifts don't pass the bar on their own.
27. **Blade, stricter view** (size ratio .9-1.11, separation > 1.8, hips level): him 36 -> 30 deg (p .04, Holm .58);
    margin over opponents +9.5 before Diaz (6/7) -> -3.3 after (p .09). Supersedes the looser filter in #22.
28. **When does he fade** (`breakpoint.py`): flat-then-linear with a level per fight, break chosen over minutes 2-15,
    fight bootstrap. Steps: best break min 15 (only Diaz 2 reaches it; 80 % range 3.5-15). Hands: decline from ~min 2,
    -0.6 cm/min, negative in 95 % of resamples. No evidence of a minute-8 break.
29. **Fist speed**: his left vs opponents' rear straight, per-fight medians: 13 vs 13 mph, faster in 2/7, p .69.
