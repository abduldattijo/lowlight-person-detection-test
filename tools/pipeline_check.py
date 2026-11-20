#!/usr/bin/env python3
"""Pipeline check (sets no parameters): RT-DETR person detections vs hand labels
on the labeled frames of each clip, for the dark and normal-light frames.
Run from the project folder:  .venv/bin/python tools/pipeline_check.py
"""
import json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, RTDetrForObjectDetection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL, THRESH, IOU = "PekingU/rtdetr_r50vd", 0.5, 0.5   # fixed in DESIGN.md
OUT = os.path.join(ROOT, "pipeline_check_out"); os.makedirs(OUT, exist_ok=True)

proc = AutoImageProcessor.from_pretrained(MODEL)
model = RTDetrForObjectDetection.from_pretrained(MODEL).eval()
PERSON = [k for k, v in model.config.id2label.items() if v == "person"][0]
print("person label id:", PERSON, "| device: cpu | threshold", THRESH)


def detect(img):
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_object_detection(out, threshold=THRESH, target_sizes=[(img.height, img.width)])[0]
    return [(b.tolist(), s.item()) for b, s, l in zip(r["boxes"], r["scores"], r["labels"]) if l.item() == PERSON]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0


for clip in ("indoor_person01", "outdoor_person04"):
    fr = json.load(open(f"{ROOT}/labels/{clip}.json"))["frames"]
    names = sorted(fr)
    for arm, kind in (("dark original", "ll_final"), ("normal-light", "nl_final")):
        gt_vis = hit = fdet = frames = 0; ious = []
        for n in names:
            img = Image.open(f"{ROOT}/data/{clip}/{kind}/{n}").convert("RGB")
            dets = detect(img)
            gts = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in fr[n]["boxes"] if b["visible"]]
            hid = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in fr[n]["boxes"] if not b["visible"]]
            used = set()
            for g in gts:
                best, bi = 0, None
                for j, (d, s) in enumerate(dets):
                    if j not in used and iou(g, d) > best: best, bi = iou(g, d), j
                gt_vis += 1
                if best >= IOU: hit += 1; used.add(bi); ious.append(best)
            # unmatched detections that overlap a hidden box are not counted as false
            fdet += sum(1 for j, (d, s) in enumerate(dets) if j not in used and not any(iou(h, d) >= 0.1 for h in hid))
            frames += 1
            if n in ("frame0000.jpg", "frame0030.jpg", "frame0055.jpg"):
                im = img.copy(); g = ImageDraw.Draw(im)
                for b in gts: g.rectangle(b, outline=(0, 255, 0), width=12)
                for d, s in dets: g.rectangle(d, outline=(255, 0, 0), width=8)
                im.resize((im.width // 4, im.height // 4)).save(f"{OUT}/{clip}_{kind}_{n[:-4]}.jpg")
        print(f"{clip:18s} {arm:14s} visible labels {gt_vis:3d} matched {hit:3d} "
              f"recall {hit/gt_vis:.2f} mean IoU {np.mean(ious) if ious else 0:.2f} "
              f"false detections {fdet} ({fdet/frames:.2f}/frame)", flush=True)
