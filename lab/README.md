# MMA evidence laboratory and Ghost Fight

This branch adds the local MMA research, tracked film viewers, reviewed annotations, prediction experiments, McGregor era comparisons, and skeleton recognition game. The original tennis project remains at the repository root.

## Run the viewers

Pull the media with Git LFS, then serve the existing artifacts. No model inference is needed to play the game or review the videos.

```sh
git lfs pull
cd mma
python3 work/serve_review.py
```

Open these pages:

- Ghost Fight: http://127.0.0.1:8767/game/index.html
- Tracked film room: http://127.0.0.1:8767/expanded/index.html
- McGregor across eras: http://127.0.0.1:8767/expanded/mcgregor/index.html
- Experiment review: http://127.0.0.1:8767/experiment/index.html
- Expanded assessment: http://127.0.0.1:8767/expanded/assessment/index.html

If port 8767 is already in use, stop the earlier viewer server or use `python3 -m http.server 8780 --bind 127.0.0.1 --directory outputs` from `mma/` and change the URLs to port 8780.

## Ghost Fight

Black background, Cyan and Coral tracking, dots and vectors before guessing, real footage with tracking after the answer. The curated pool contains 11 committed strike exchanges, an eight-round fight card, an eleven-round full card, and a six-round McGregor era challenge. One window per selected substantive attack or exchange; quiet probing, reset-only windows, and ambiguous attempts are excluded. Both fighter identities must pass the window coverage gate and the strike-frame coverage gate. Selection means a visible attack, not proof of clean contact or damage.

`outputs/game/strike-selection.json` records the reviewed evidence and attack interval for every round. `outputs/game/build-audit.json` records excluded windows and input hashes. The small collection repeats fighters and bouts, so scores measure recognition of these sampled clips, not broad MMA expertise or career-wide style changes.

## What is preserved

`outputs/` contains the original viewers, source excerpts, tracked videos, pose data, rich labels, independent reviews, audit material, experiment results, and reports. `work/` contains the study scripts and run configuration, retaining their original relative layout. Run reproduction scripts from `mma/`; requirements differ by experiment and are recorded in the existing reproduce directories and environment manifest. Rebuilding inference requires the separately installed model/runtime dependencies and, for some scripts, downloaded full source fights. Viewing the committed results does not.

`artifact-inventory.json` records SHA-256 hashes and sizes of the original delivered artifacts and research scripts. Full downloaded fights, model weights, virtual environments, vendor caches, and redundant ZIP bundles are excluded. Images and short video artifacts are stored with Git LFS. Historical audit records retain original local paths; those paths document provenance rather than portable execution locations.

Sources inspected: Peter Wang's deep-sports-analysis and tennis skeleton quiz, plus Maximilianb1/mma-fight-analyzer. No new model was trained. Full methodological limits and findings are included in the assessment pages and existing reports.
