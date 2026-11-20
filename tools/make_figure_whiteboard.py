#!/usr/bin/env python3
"""Figure: whiteboard, DETR-R50, dark vs gamma 1.5 vs normal-light, on four frames from the missed run.
Frames fixed by rule: evenly spaced through the dark arm's no-detection run (99, 100, 102-110): 100, 104, 108, 112.
Green = hand label, red = detector output (score shown). Run: .venv/bin/python tools/make_figure_whiteboard.py"""
import json, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FR = [100, 104, 108, 112]; ARMS = [("dark", "ll_final", "Dark original"), ("gamma_1.5", "ll_final", "Gamma 1.5"), ("normal", "nl_final", "Normal-light reference")]
lab = json.load(open(f"{ROOT}/labels/whiteboard.json"))["frames"]
det = {a: json.load(open(f"{ROOT}/results/detections_detr/whiteboard__{a}.json"))["frames"] for a, _, _ in ARMS}
TW = 520; S = None; font = ImageFont.load_default()
def gamma(im, g):
    lut = (255.0 * (np.arange(256) / 255.0) ** (1.0 / g)).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(lut[np.asarray(im)])
rows = []
for k in FR:
    n = "frame%04d.jpg" % k; tiles = []
    for a, kind, title in ARMS:
        im = Image.open(f"{ROOT}/data/whiteboard/{kind}/{n}").convert("RGB")
        if a == "gamma_1.5": im = gamma(im, 1.5)
        S = TW / im.width; im = im.resize((TW, int(im.height * S))); g = ImageDraw.Draw(im)
        b = lab[n]["boxes"][0]; g.rectangle([b["x1"]*S, b["y1"]*S, b["x2"]*S, b["y2"]*S], outline=(0, 220, 0), width=3)
        for d in det[a][n]:
            x1, y1, x2, y2 = [v * S for v in d["box"]]; g.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=3)
            g.text((x1 + 4, y1 + 4), f"{d['score']:.2f}", fill=(255, 255, 0), font=font)
        if not det[a][n]: g.text((8, im.height - 18), "no detection", fill=(255, 255, 0), font=font)
        g.text((8, 6), f"frame {k}", fill=(255, 255, 255), font=font); tiles.append(im)
    rows.append(tiles)
th = rows[0][0].height; head = 26
sheet = Image.new("RGB", (TW * 3 + 8, head + (th + 4) * len(FR)), (255, 255, 255)); d = ImageDraw.Draw(sheet)
for j, (_, _, t) in enumerate(ARMS): d.text((j * (TW + 4) + 6, 8), t, fill=(0, 0, 0), font=font)
for i, tiles in enumerate(rows):
    for j, im in enumerate(tiles): sheet.paste(im, (j * (TW + 4), head + i * (th + 4)))
os.makedirs(f"{ROOT}/figures", exist_ok=True); sheet.save(f"{ROOT}/figures/whiteboard_detr.png")
print(sheet.size)
