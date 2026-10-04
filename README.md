# NotoriousData

Every live second of fifteen Conor McGregor fights, 2012 to 2021, measured frame by frame against the man in front of him.

**Read it:** https://nicodunks.github.io/NotoriousData/

By Nico Christie and Peter Wang. The pipeline follows Peter Wang's [deep-sports-analysis](https://github.com/pwang724/deep-sports-analysis) (keep live play, find the people, track their joints, test per fight).

## What's here

| Path | What it is |
|---|---|
| `docs/` | The published page (GitHub Pages): HTML, data files and clips |
| `study/` | The McGregor study: pipeline, analyses, page builder, `PREREGISTRATION.md` (every rule and change, in order) |
| `study/results/` | The analysis outputs as JSON (findings, per-fight values, tests) |
| `lab/` | Earlier MMA experiments: detector and pose comparisons, the Ghost Fight game |

## Models

- **Fighter detection:** YOLOv8, fine-tuned on UFC footage
- **Pose:** RTMW-x whole-body (133 points), via [rtmlib](https://github.com/Tau-J/rtmlib) on ONNX Runtime
- **Movement syllables:** [keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq) (Weinreb et al., 2024)

## Reproducing

Raw video, pose files and model weights are not in the repo. Set `MMA_WORK` to a working directory, then:

```bash
cd study
sh download.sh                 # fight footage (yt-dlp)
python board.py && python pose.py --live     # fight clock, then pose on live play
python findings.py && python leftfindings.py && python evasion.py && python fatigue.py && python breakpoint.py
python pairs.py && python game.py && python build_site.py
```

`study/README.md` and `study/PREREGISTRATION.md` describe each step and every rule.

## Footage

Clips on the page are short excerpts of UFC and Cage Warriors broadcasts, used for commentary and analysis. All rights in the footage belong to their owners.
