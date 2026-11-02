#!/usr/bin/env python3
"""Re-run the laps used in the demo video (same seeds -> same runs as the logs) and keep every
camera frame + the perception debug output, so the video shows the real pipeline output."""
import json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
from line_follower.camera_model import CameraModel
from line_follower.sim import World, run_lap
from line_follower.perception import LinePerception, CentroidPerception
from line_follower.controllers import CONTROLLERS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "video", "rec")
os.makedirs(OUT, exist_ok=True)
RUNS = [("technical", "hard", 0.19, "stanley_ff", 0), ("technical", "hard", 0.26, "pid", 0),
        ("technical", "hard", 0.26, "stanley_ff", 0), ("figure_eight", "hard", 0.19, "centroid_pid", 0),
        ("figure_eight", "hard", 0.19, "stanley_ff", 0)]
if len(sys.argv) > 1:
    RUNS = [r for r in RUNS if any(a in f"{r[0]}_{r[1]}_{r[3]}_v{r[2]:.2f}" for a in sys.argv[1:])]
c = json.load(open(os.path.join(ROOT, "results", "calibration.json")))
K, D = np.array(c["K"]), np.array(c["D"])
cam = CameraModel()
for tr, cond, v, ctrl, seed in RUNS:
    w = World(tr, cond, seed, cam)
    per = CentroidPerception((cam.w, cam.h)) if ctrl == "centroid_pid" else LinePerception(K, D, cam.R, cam.C, (cam.w, cam.h))
    frames = []

    def cb(t, robot, img, est, cte):
        d = est.dbg
        rec = dict(t=t, pose=robot.pose, cte=cte, ok=est.ok, lost=est.lost_frames, coef=None if est.coef is None else np.array(est.coef),
                   off=est.offset, head=est.heading, curv=est.curvature, lat=est.latency_ms,
                   jpg=cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])
        if ctrl == "centroid_pid":
            rec.update(cx=d.get("cx"), cy=d.get("cy"), contour=d.get("contour"), roi=d.get("roi"))
        else:
            rec.update(mask=np.packbits(d["mask"]), wins=d["wins"], bev=cv2.imencode(".jpg", d["bev"])[1])
        frames.append(rec)

    log, s = run_lap(tr, cond, v, per, CONTROLLERS[ctrl](v), seed, world=w, on_frame=cb)
    name = f"{tr}_{cond}_{ctrl}_v{v:.2f}_s{seed}"
    cv2.imwrite(os.path.join(OUT, f"floor_{tr}_{cond}_s{seed}.png"),
                np.clip(w.floor_lit.astype(np.float32) + w.glare[..., None] * 0.6, 0, 255).astype(np.uint8))
    with open(os.path.join(OUT, name + ".pkl"), "wb") as f:
        pickle.dump(dict(summary=s, frames=frames, bev_shape=(per.bh, per.bw) if hasattr(per, "bh") else None,
                         track=tr, pts=w.track.pts), f)
    print(name, s["status"], f"{s['cte_rms_cm']:.2f} cm", len(frames), "frames", flush=True)
