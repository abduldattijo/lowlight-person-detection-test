#!/usr/bin/env python3
"""Secondary, post-hoc measure (added after the first recall result was a tie): how confident is the
detector in the person it finds, per arm? Written before looking at any confidence numbers.

Definition (fixed here): for every visible labeled person-frame where the person is matched (IoU >= 0.5)
in ALL five arms, take the score of the matched detection in each arm. Report the mean per arm per clip,
and the mean paired difference to the dark arm (arm minus dark) with its min and max over frames.
No significance test across clips; seven clips are descriptive only.

Run:  DETECTOR=rtdetr|detr .venv/bin/python tools/confidence.py
"""
import json, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_test.py")).read().split("if __name__")[0]
ns = {"__name__": "rt", "__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_test.py")}
exec(src, ns)
ARMS, CLIPS, DET, match, labeled_frames, MODEL, RES, SUFFIX = (ns[k] for k in ("ARMS", "CLIPS", "DET", "match", "labeled_frames", "MODEL", "RES", "SUFFIX"))
iou = ns["iou"]

L = [f"# Confidence of matched person detections ({MODEL})\n",
     "Frames where the person is matched in all five arms. Mean score per arm; paired difference vs dark = mean (min to max) of arm minus dark.\n",
     "| Clip | Paired frames | dark | gamma_1.5 | gamma_2.5 | clahe | normal | gamma_1.5 - dark | gamma_2.5 - dark | clahe - dark | normal - dark |", "|---|---|---|---|---|---|---|---|---|---|---|"]
tot = {a: [] for a in ARMS}
for clip in CLIPS:
    if not all(os.path.exists(f"{DET}/{clip}__{a}.json") for a in ARMS): continue
    names, fr = labeled_frames(clip)
    D = {a: json.load(open(f"{DET}/{clip}__{a}.json"))["frames"] for a in ARMS}
    S = {a: [] for a in ARMS}
    for n in names:
        vis = [(b["id"], [b["x1"], b["y1"], b["x2"], b["y2"]]) for b in fr[n]["boxes"] if b["visible"]]
        for gid, gbox in vis:
            sc = {}
            for a in ARMS:
                # score of the detection matched to this label (same greedy matching as scoring)
                pairs = sorted(((iou(gbox, d["box"]), di) for di, d in enumerate(D[a][n])), reverse=True)
                if pairs and pairs[0][0] >= 0.5: sc[a] = D[a][n][pairs[0][1]]["score"]
            if len(sc) == len(ARMS):
                for a in ARMS: S[a].append(sc[a])
    n = len(S["dark"])
    if not n: continue
    m = {a: sum(S[a]) / n for a in ARMS}
    diffs = {a: [x - y for x, y in zip(S[a], S["dark"])] for a in ARMS}
    cells = [f"{sum(d)/n:+.3f} ({min(d):+.2f} to {max(d):+.2f})" for a, d in ((a, diffs[a]) for a in ARMS[1:])]
    L.append(f"| {clip} | {n} | " + " | ".join(f"{m[a]:.3f}" for a in ARMS) + " | " + " | ".join(cells) + " |")
    for a in ARMS: tot[a] += S[a]
open(f"{RES}/confidence{SUFFIX}.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
