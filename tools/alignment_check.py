#!/usr/bin/env python3
"""How well do the dark (ll_final) and normal-light (nl_final) twins line up?
No detector. Shift between twins by phase correlation on brightness-normalised
gradient images, (a) whole frame and (b) inside the hand-labelled person box,
reported in full-resolution pixels. Also reports how fast the person moves.
Run:  .venv/bin/python tools/alignment_check.py
"""
import json, os
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = 4  # downsample factor for speed


def prep(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (g.shape[1] // DS, g.shape[0] // DS), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)
    # local contrast normalisation so dark and bright twins look alike
    mu = cv2.GaussianBlur(g, (0, 0), 12); sd = np.sqrt(cv2.GaussianBlur((g - mu) ** 2, (0, 0), 12)) + 2.0
    n = (g - mu) / sd
    return n


def shift(a, b, box=None):
    if box:
        x1, y1, x2, y2 = [int(v / DS) for v in box]
        a, b = a[y1:y2, x1:x2], b[y1:y2, x1:x2]
        if a.shape[0] < 24 or a.shape[1] < 24: return None
    w = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, w)
    return dx * DS, dy * DS, resp


clips = sorted(d for d in os.listdir(f"{ROOT}/data"))
for clip in clips:
    names = sorted(os.listdir(f"{ROOT}/data/{clip}/nl_final"))
    lab = f"{ROOT}/labels/{clip}.json"
    fr = json.load(open(lab))["frames"] if os.path.exists(lab) else {}
    step = 1 if fr else 5
    rows = []; prev = None
    for n in names[::step]:
        a = prep(f"{ROOT}/data/{clip}/ll_final/{n}"); b = prep(f"{ROOT}/data/{clip}/nl_final/{n}")
        g = shift(a, b)
        row = {"n": n, "gdx": g[0], "gdy": g[1]}
        if n in fr and fr[n]["boxes"]:
            bx = fr[n]["boxes"][0]; box = (bx["x1"], bx["y1"], bx["x2"], bx["y2"])
            s = shift(a, b, box)
            if s and bx["visible"]: row["pdx"], row["pdy"] = s[0], s[1]
            c = ((bx["x1"] + bx["x2"]) / 2, (bx["y1"] + bx["y2"]) / 2)
            row["speed"] = np.hypot(c[0] - prev[0], c[1] - prev[1]) if prev else 0
            prev = c
        rows.append(row)
    G = np.array([[r["gdx"], r["gdy"]] for r in rows])
    gm = np.hypot(*G.T)
    print(f"\n{clip}: {len(rows)} frames checked (step {step}). Whole-frame shift dark->normal, full-res px: "
          f"median {np.median(gm):.1f}, 90th pct {np.percentile(gm,90):.1f}, max {gm.max():.1f}")
    P = [r for r in rows if "pdx" in r]
    if P:
        pm = np.array([np.hypot(r["pdx"] - r["gdx"], r["pdy"] - r["gdy"]) for r in P])
        sp = np.array([r["speed"] for r in P])
        print(f"  Person region shift beyond the background shift: median {np.median(pm):.1f}, 90th pct {np.percentile(pm,90):.1f}, max {pm.max():.1f} px")
        fast = sp >= np.percentile(sp, 75)
        print(f"  Fast frames (top quarter by person speed, >= {np.percentile(sp,75):.0f} px/frame): median extra shift {np.median(pm[fast]):.1f} px; slow frames: {np.median(pm[~fast]):.1f} px")
        print(f"  correlation(speed, extra shift) = {np.corrcoef(sp, pm)[0,1]:.2f}")
        worst = sorted(zip(pm, [r['n'] for r in P], sp), reverse=True)[:5]
        print("  worst frames:", ", ".join(f"{n[5:9]} ({d:.0f}px, speed {s:.0f})" for d, n, s in worst))
