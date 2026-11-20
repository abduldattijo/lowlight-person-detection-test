#!/usr/bin/env python3
"""Full test: five arms x labeled frames of each clip, fixed RT-DETR person detector.

Everything below is fixed in DESIGN.md before any run; nothing is tuned on scores.
  arms:  dark          original dark frame (ll_final)
         gamma_1.5     out = 255*(in/255)^(1/1.5) on the dark frame
         gamma_2.5     out = 255*(in/255)^(1/2.5) on the dark frame
         clahe         OpenCV CLAHE (clipLimit 2.0, 8x8 tiles) on the L channel of LAB, per frame
         normal        normal-light twin (nl_final), the reference
  detector: PekingU/rtdetr_r50vd, person class, confidence threshold 0.5
  match:  IoU >= 0.5 (greedy by IoU), person class only

Step 1 (slow): run detector, cache raw detections to results/detections/<clip>__<arm>.json
Step 2 (fast): score from the cache -> results/scores.json and results/scores.md

Usage (project folder):
  .venv/bin/python tools/run_test.py detect [clip ...]    # skips clip/arm already cached
  .venv/bin/python tools/run_test.py score
  .venv/bin/python tools/run_test.py smoke                # 3 frames, prints only 'ran ok', saves nothing
"""
import json, os, sys, time
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DETECTOR = os.environ.get("DETECTOR", "rtdetr")          # rtdetr (main) or detr (second detector, added after the first result)
MODEL = {"rtdetr": "PekingU/rtdetr_r50vd", "detr": "facebook/detr-resnet-50"}[DETECTOR]   # both Apache-2.0
SUFFIX = "" if DETECTOR == "rtdetr" else "_" + DETECTOR
DET = os.path.join(RES, "detections" + SUFFIX)
THRESH, IOU = 0.5, 0.5
ARMS = ["dark", "gamma_1.5", "gamma_2.5", "clahe", "normal"]
CLIPS = ["indoor_person01", "outdoor_person04", "outdoor_person05", "outdoor_person02",
         "drawing", "hand_gesture02", "whiteboard"]


def labeled_frames(clip):
    d = json.load(open(f"{ROOT}/labels/{clip}.json")); fr = d["frames"]
    a, b = d["range"]
    names = ["frame%04d.jpg" % k for k in range(a, b + 1)]
    assert all(n in fr and fr[n]["done"] for n in names), f"{clip}: not all frames done"
    return names, fr


def make_arm(arm, clip, n):
    kind = "nl_final" if arm == "normal" else "ll_final"
    bgr = cv2.imread(f"{ROOT}/data/{clip}/{kind}/{n}")
    t0 = time.perf_counter()
    if arm.startswith("gamma_"):
        g = float(arm.split("_")[1])
        lut = (255.0 * (np.arange(256) / 255.0) ** (1.0 / g)).round().clip(0, 255).astype(np.uint8)
        bgr = cv2.LUT(bgr, lut)
    elif arm == "clahe":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr, time.perf_counter() - t0


def load_detector():
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForObjectDetection.from_pretrained(MODEL).eval()
    person = [k for k, v in model.config.id2label.items() if v == "person"][0]

    def detect(bgr):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = proc(images=img, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_object_detection(out, threshold=THRESH, target_sizes=[(img.height, img.width)])[0]
        dt = time.perf_counter() - t0
        dets = [{"box": [round(v, 1) for v in b.tolist()], "score": round(s.item(), 4)}
                for b, s, l in zip(r["boxes"], r["scores"], r["labels"]) if l.item() == person]
        return dets, dt
    return detect


def do_detect(clips, smoke=False):
    os.makedirs(DET, exist_ok=True)
    detect = load_detector()
    for clip in clips:
        names, _ = labeled_frames(clip)
        if smoke: names = names[:3]
        for arm in ARMS:
            f = f"{DET}/{clip}__{arm}.json"
            if os.path.exists(f) and not smoke: continue
            out = {}; tp = td = 0.0
            for n in names:
                bgr, p = make_arm(arm, clip, n); dets, d = detect(bgr)
                out[n] = dets; tp += p; td += d
            if smoke:
                print(clip, arm, "ran ok", len(out), "frames"); continue
            json.dump({"clip": clip, "arm": arm, "frames": out,
                       "prep_s_per_frame": tp / len(names), "detect_s_per_frame": td / len(names)}, open(f, "w"))
            print(clip, arm, "saved", flush=True)
        if smoke: break


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def match(gts, dets):
    """greedy by IoU; gts = list of (id, box); returns set of matched gt ids and used det indices"""
    pairs = sorted(((iou(g[1], d["box"]), gi, di) for gi, g in enumerate(gts) for di, d in enumerate(dets)), reverse=True)
    mg, md = {}, set()
    for v, gi, di in pairs:
        if v < IOU: break
        if gi in mg or di in md: continue
        mg[gi] = di; md.add(di)
    return {gts[gi][0] for gi in mg}, md


def score_clip(clip):
    names, fr = labeled_frames(clip)
    res = {}
    for arm in ARMS:
        det = json.load(open(f"{DET}/{clip}__{arm}.json"))
        r = dict(gt=0, hit=0, fdet=0, gt_all=0, hit_all=0, pairs=0, dropouts=0, frames=len(names))
        hit_by_frame = {}
        for n in names:
            dets = det["frames"][n]
            vis = [(b["id"], [b["x1"], b["y1"], b["x2"], b["y2"]]) for b in fr[n]["boxes"] if b["visible"]]
            hid = [(b["id"], [b["x1"], b["y1"], b["x2"], b["y2"]]) for b in fr[n]["boxes"] if not b["visible"]]
            m, used = match(vis, dets)
            r["gt"] += len(vis); r["hit"] += len(m); hit_by_frame[n] = (m, {i for i, _ in vis})
            # sensitivity: hidden boxes counted as visible (exit frames included)
            m2, _ = match(vis + hid, dets); r["gt_all"] += len(vis) + len(hid); r["hit_all"] += len(m2)
            r["fdet"] += sum(1 for j, d in enumerate(dets) if j not in used and not any(iou(h[1], d["box"]) >= 0.1 for h in hid))
        for a, b in zip(names[:-1], names[1:]):
            (ma, va), (mb, vb) = hit_by_frame[a], hit_by_frame[b]
            for i in va & vb:                      # visible in both frames, same track ID
                r["pairs"] += 1
                if i in ma and i not in mb: r["dropouts"] += 1   # matched in first, missed in second
        r["recall"] = r["hit"] / r["gt"] if r["gt"] else None
        r["miss_rate"] = 1 - r["recall"] if r["gt"] else None
        r["recall_with_hidden"] = r["hit_all"] / r["gt_all"] if r["gt_all"] else None
        r["fdet_per_frame"] = r["fdet"] / r["frames"]
        r["dropout_rate"] = r["dropouts"] / r["pairs"] if r["pairs"] else None
        r["ms_per_frame"] = 1000 * (det["prep_s_per_frame"] + det["detect_s_per_frame"])
        res[arm] = r
    return res


def do_score():
    allr = {c: score_clip(c) for c in CLIPS if all(os.path.exists(f"{DET}/{c}__{a}.json") for a in ARMS)}
    json.dump(allr, open(f"{RES}/scores{SUFFIX}.json", "w"), indent=1)
    L = [f"# Scores ({MODEL}, person, threshold 0.5, IoU 0.5)\n",
         "Recall = matched visible person boxes / visible person boxes. Dropouts = person matched in one frame and missed in the next "
         "(eligible = visible in both frames, same track ID). Miss rate is shown next to dropouts because the arms have different denominators.\n",
         "| Clip | Arm | Visible labels | Recall | Miss rate | Eligible pairs | Dropouts | False det/frame | ms/frame |", "|---|---|---|---|---|---|---|---|---|"]
    for c, rr in allr.items():
        for a in ARMS:
            r = rr[a]
            L.append(f"| {c} | {a} | {r['gt']} | {r['recall']:.3f} | {r['miss_rate']:.3f} | {r['pairs']} | {r['dropouts']} | {r['fdet_per_frame']:.2f} | {r['ms_per_frame']:.0f} |")
    open(f"{RES}/scores{SUFFIX}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "detect": do_detect(sys.argv[2:] or CLIPS)
    elif cmd == "smoke": do_detect(["indoor_person01"], smoke=True)
    elif cmd == "score": do_score()
    else: sys.exit(__doc__)
