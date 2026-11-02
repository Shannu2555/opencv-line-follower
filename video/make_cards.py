#!/usr/bin/env python3
"""Figures for the video built from real data: pipeline stages on a recorded camera frame,
and the calibration (detected corners + undistortion)."""
import json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from line_follower.camera_model import CameraModel
from line_follower.perception import LinePerception

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
c = json.load(open(os.path.join(ROOT, "results", "calibration.json")))
K, D = np.array(c["K"]), np.array(c["D"])
cam = CameraModel()
d = pickle.load(open(os.path.join(ROOT, "video", "rec", "technical_hard_stanley_ff_v0.19_s0.pkl"), "rb"))
frames = d["frames"]
# pick the frame with the most lighting contrast (shadow edge / glare) where the fit is good
per = LinePerception(K, D, cam.R, cam.C, (cam.w, cam.h))
best, score = None, -1
for i in range(20, len(frames), 5):
    img = cv2.imdecode(frames[i]["jpg"], cv2.IMREAD_COLOR)
    v = img.max(2).astype(np.float32)
    sc = np.percentile(v, 98) - np.percentile(v, 10)
    if frames[i]["ok"] and frames[i]["lost"] == 0 and abs(frames[i]["cte"]) < 0.02 and sc > score:
        best, score = i, sc
img = cv2.imdecode(frames[best]["jpg"], cv2.IMREAD_COLOR)
per.prev = None
for j in range(max(0, best - 8), best + 1):  # warm up the tracker on the frames before it
    e = per.process(cv2.imdecode(frames[j]["jpg"], cv2.IMREAD_COLOR), keep_debug=True)
dbg = e.dbg
fit = cv2.cvtColor(dbg["bev"], cv2.COLOR_BGR2RGB).copy()
fit = (fit * 0.6).astype(np.uint8)
fit[dbg["mask"] > 0] = (34, 197, 94)
fit = cv2.resize(fit, (500, 500), interpolation=cv2.INTER_NEAREST)
k = 4
for x0, r0, x1, r1, good in dbg["wins"]:
    cv2.rectangle(fit, (x0 * k, r0 * k), (x1 * k, r1 * k), (34, 211, 238) if good else (148, 163, 184), 2, cv2.LINE_AA)
rows = np.arange(per.bh)
fw = per.rows_f
cols = (per.p["y_half"] - e.y_at(fw)) / per.p["bev_res"]
cv2.polylines(fit, [np.round(np.column_stack([cols * k, rows * k])).astype(np.int32)], False, (217, 70, 239), 4, cv2.LINE_AA)
panels = [("1. raw camera\n(lens distortion, glare, shadow)", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
          ("2. undistorted\n(calibrated K, D)", cv2.cvtColor(dbg["und"], cv2.COLOR_BGR2RGB)),
          ("3. bird's-eye view\n0.16–0.66 m ahead, 4 mm/px", cv2.cvtColor(dbg["bev"], cv2.COLOR_BGR2RGB)),
          ("4. mask: brightness ÷\nlocal background < 0.66", (dbg["mask"] * 255).astype(np.uint8)),
          ("5. sliding windows +\nRANSAC quadratic", fit)]
fig = plt.figure(figsize=(12, 8.2), dpi=200)
pos = [(0.02, 0.52, 0.46, 0.42), (0.52, 0.52, 0.46, 0.42), (0.02, 0.03, 0.30, 0.40), (0.35, 0.03, 0.30, 0.40), (0.68, 0.03, 0.30, 0.40)]
for (t, im), p in zip(panels, pos):
    ax = fig.add_axes(p)
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest" if im.shape[0] < 200 else "antialiased")
    ax.set_title(t, fontsize=13, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.02, 0.005, f"offset {e.offset*100:+.1f} cm · heading {np.degrees(e.heading):+.1f}° · curvature {e.curvature:+.2f} 1/m"
         f" · true CTE {frames[best]['cte']*100:+.1f} cm", fontsize=9, color="#475569")
os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "plots", "pipeline_stages.png"))
print("pipeline frame", best, "t", frames[best]["t"])

# calibration figure
from line_follower.calibration import make_views
views = make_views(cam, 25)
for im0 in views:
    im0 = im0[0] if isinstance(im0, tuple) else im0
    g = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY) if im0.ndim == 3 else im0
    ok, cor = cv2.findChessboardCorners(g, (9, 6))
    if ok:
        break
vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
if ok:
    cv2.drawChessboardCorners(vis, (9, 6), cv2.cornerSubPix(g, cor, (5, 5), (-1, -1), (3, 30, 1e-3)), ok)
und = cv2.undistort(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), K, D)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), dpi=200)
ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[0].set_title("rendered board + detected 9×6 corners", loc="left")
ax[1].imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB)); ax[1].set_title("undistorted with the estimated K, D", loc="left")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.text(0.01, 0.02, f"fx {K[0,0]:.1f} (true 230) · fy {K[1,1]:.1f} · cx {K[0,2]:.1f} (160) · cy {K[1,2]:.1f} (120) · "
         f"k1 {D[0]:.3f} (-0.280) · k2 {D[1]:.3f} (0.080)", fontsize=11, color="#475569")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(ROOT, "plots", "calibration.png"))
print("calib ok", ok)
