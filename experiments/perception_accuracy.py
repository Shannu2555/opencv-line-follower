#!/usr/bin/env python3
"""Static perception accuracy: put the robot at known poses along each track (random +-3 cm offset,
+-8 deg heading) and compare the fitted line with the true centreline at 15/25/40/55 cm ahead.
Uses the calibrated intrinsics. Writes results/perception_accuracy.json and a raw csv."""
import csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from line_follower.camera_model import CameraModel
from line_follower.perception import LinePerception
from line_follower.sim import World

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOOK = (0.20, 0.30, 0.45, 0.60)


def truth_lateral(tr, x, y, th, i):
    n = len(tr.pts)
    win = np.arange(i - 40, i + 160) % n  # only the stretch ahead (figure-eight crossing!)
    d = tr.pts[win] - (x, y)
    c, s = np.cos(th), np.sin(th)
    fx = c * d[:, 0] + s * d[:, 1]
    fy = -s * d[:, 0] + c * d[:, 1]
    o = np.argsort(fx)
    return np.interp(LOOK, fx[o], fy[o])


c = json.load(open(os.path.join(ROOT, "results", "calibration.json")))
K, D = np.array(c["K"]), np.array(c["D"])
cam = CameraModel()
rows, out = [], {}
rng = np.random.default_rng(42)
for track in ("oval_chicane", "figure_eight", "technical"):
    for cond in ("clean", "hard"):
        w = World(track, cond, 0, cam)
        tr = w.track
        p = LinePerception(K, D, cam.R, cam.C, (cam.w, cam.h))
        errs, head, miss = [], [], 0
        for i in range(0, len(tr.pts), 37):
            th = tr.tangent[i] + rng.uniform(-0.14, 0.14)
            off = rng.uniform(-0.03, 0.03)
            x, y = tr.pts[i, 0] + np.sin(tr.tangent[i]) * off, tr.pts[i, 1] - np.cos(tr.tangent[i]) * off
            p.reset()
            e = p.process(w.camera_image((x, y, th), rng.uniform(0, 14)))
            if not e.ok:
                miss += 1
                rows.append([track, cond, i, "", "", "", ""])
                continue
            err = [e.y_at(f) - t for f, t in zip(LOOK, truth_lateral(tr, x, y, th, i))]
            errs.append(err)
            rows.append([track, cond, i] + [f"{v*1000:.2f}" for v in err])
        E = np.abs(np.array(errs)) * 1000
        out[f"{track}/{cond}"] = dict(samples=len(errs) + miss, missed=miss,
                                     rms_mm={f"{int(f*100)}cm": float(np.sqrt((E[:, k] ** 2).mean())) for k, f in enumerate(LOOK)},
                                     p95_mm={f"{int(f*100)}cm": float(np.percentile(E[:, k], 95)) for k, f in enumerate(LOOK)})
        print(track, cond, out[f"{track}/{cond}"])
json.dump(out, open(os.path.join(ROOT, "results", "perception_accuracy.json"), "w"), indent=2)
with open(os.path.join(ROOT, "results", "perception_accuracy_raw.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["track", "condition", "idx"] + [f"err_mm_{int(x*100)}cm" for x in LOOK])
    wr.writerows(rows)
