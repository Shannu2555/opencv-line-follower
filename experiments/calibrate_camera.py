#!/usr/bin/env python3
"""Render 25 checkerboard views through the lens, run OpenCV calibration, compare with the true model."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
from line_follower.camera_model import CameraModel
from line_follower.calibration import make_views, calibrate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cam = CameraModel()
imgs = make_views(cam, 25, seed=3)
os.makedirs(os.path.join(ROOT, "results", "calib_views"), exist_ok=True)
for i in (0, 5, 10):
    cv2.imwrite(os.path.join(ROOT, "results", "calib_views", f"view_{i:02d}.png"), imgs[i])
r = calibrate(imgs)
# how far is the calibrated model from the truth, in pixels, over the whole image?
u, v = np.meshgrid(np.arange(0, cam.w, 4.0), np.arange(0, cam.h, 4.0))
pts = np.stack([u, v], -1).reshape(-1, 1, 2)
crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-10)
n_true = cv2.undistortPoints(pts, cam.K, cam.D, criteria=crit).reshape(-1, 2)
n_est = cv2.undistortPoints(pts, r["K"], r["D"], criteria=crit).reshape(-1, 2)
err_px = np.linalg.norm(n_true - n_est, axis=1) * cam.K[0, 0]
out = dict(views_used=r["views_used"], views_total=r["views_total"], reproj_rms_px=r["rms_px"],
           K=r["K"].tolist(), D=r["D"].tolist(), K_true=cam.K.tolist(), D_true=cam.D.tolist(),
           undistort_err_px_mean=float(err_px.mean()), undistort_err_px_max=float(err_px.max()))
with open(os.path.join(ROOT, "results", "calibration.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
