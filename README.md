# Does brightening dark video help a person detector

Code, hand labels and saved results for a test of two ways to brighten dark video before running a person detector. It uses 7 real dark clips from the PiLLVD dataset, each with a normal-light twin, and 60 hand-labeled frames per clip (420 frames, 416 labeled person boxes).

This repository holds no video frames. The frames belong to the PiLLVD dataset (Abu Awwad, Rosin, Karakus, arXiv 2611.16515, Zenodo record 23015904, CC BY 4.0). `tools/fetch_clips.py` downloads the 7 clips from Zenodo, about 9 GB.

## What was tested

Five conditions per frame:

| Condition | What is run |
|---|---|
| `dark` | the dark frame as captured (`ll_final`) |
| `gamma_1.5` | output = 255 x (input / 255)^(1 / 1.5) on the dark frame |
| `gamma_2.5` | the same with 2.5 |
| `clahe` | OpenCV CLAHE (clip limit 2.0, 8x8 tiles) on the L channel of LAB, per frame |
| `normal` | the normal-light twin (`nl_final`), the reference |

Two detectors, both Apache-2.0: `PekingU/rtdetr_r50vd` (run first) and `facebook/detr-resnet-50` (added after the first result). Person class only, confidence threshold 0.5, a match is IoU of 0.5 or more. The settings were fixed before any run and were not tuned on scores.

## Saved results

| File | Content |
|---|---|
| `results/scores.md`, `scores.json` | RT-DETR recall, dropouts, false detections and time per frame, by clip and condition |
| `results/scores_detr.md`, `scores_detr.json` | the same for DETR |
| `results/confidence.md`, `confidence_detr.md` | mean confidence of matched detections (frames matched in all five conditions) |
| `results/detections/`, `results/detections_detr/` | the raw detections, one file per clip and condition |
| `labels/*.json` | the hand labels: one person box per frame, ID 1, `visible` flag, labeled range in `range` |

Short version: RT-DETR found 410 of 416 people in every condition. DETR found 396 of 416 on dark frames and 411 of 416 after gamma 1.5; the difference comes mostly from one clip (`whiteboard`). Seven clips, one scene each, so treat it as exploratory.

## Run it

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 tools/fetch_clips.py                       # downloads the 7 clips into data/ (about 9 GB)
.venv/bin/python tools/run_test.py detect          # RT-DETR on all conditions (a few minutes per clip on CPU)
.venv/bin/python tools/run_test.py score
DETECTOR=detr .venv/bin/python tools/run_test.py detect
DETECTOR=detr .venv/bin/python tools/run_test.py score
.venv/bin/python tools/confidence.py               # also with DETECTOR=detr
.venv/bin/python tools/alignment_check.py          # how well the dark and normal twins line up
```

`detect` skips clip and condition pairs that are already saved, so the saved detections can be rescored without rerunning the detector. `tools/label_server.py` is the local page used to draw the boxes (open http://127.0.0.1:8800). `tools/make_figure_whiteboard.py` redraws the whiteboard figure.

## Licence

Code, labels and result files: MIT (see `LICENSE`). The video frames are not included and keep their own CC BY 4.0 licence.
