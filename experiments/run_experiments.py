#!/usr/bin/env python3
"""Full test matrix: tracks x lighting x speeds x controllers x seeds. Writes one raw log per run
(logs/<track>/<cond>/<ctrl>_v<speed>_s<seed>.csv.gz) and results/summary.csv."""
import argparse, csv, gzip, json, os, sys, time, itertools
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from line_follower.camera_model import CameraModel
from line_follower.sim import World, run_lap
from line_follower.perception import LinePerception, CentroidPerception
from line_follower.controllers import CONTROLLERS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKS = ["oval_chicane", "figure_eight", "technical"]
CONDS = ["clean", "hard"]
SPEEDS = [0.12, 0.19, 0.26]
CTRLS = ["centroid_pid", "pid", "pure_pursuit", "stanley_ff"]
SEEDS = [0, 1, 2]
_world = {}


def calib():
    with open(os.path.join(ROOT, "results", "calibration.json")) as f:
        c = json.load(f)
    return np.array(c["K"]), np.array(c["D"])


def job(args):
    track, cond, speed, ctrl, seed = args
    cam = CameraModel()
    key = (track, cond, seed)
    if key not in _world:
        _world.clear()
        _world[key] = World(track, cond, seed, cam)
    K, D = calib()  # perception only gets the calibrated intrinsics, never the true ones
    per = CentroidPerception((cam.w, cam.h)) if ctrl == "centroid_pid" else LinePerception(K, D, cam.R, cam.C, (cam.w, cam.h))
    log, s = run_lap(track, cond, speed, per, CONTROLLERS[ctrl](speed), seed, world=_world[key])
    d = os.path.join(ROOT, "logs", track, cond)
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, f"{ctrl}_v{speed:.2f}_s{seed}.csv.gz"), "wt") as f:
        keys = list(log)
        f.write(",".join(keys) + "\n")
        for row in zip(*[log[k] for k in keys]):
            f.write(",".join(f"{x:.5g}" for x in row) + "\n")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = list(itertools.product(TRACKS, CONDS, SPEEDS, CTRLS, SEEDS))
    jobs.sort(key=lambda j: (j[0], j[1], j[4]))
    t0 = time.time()
    out = []
    with Pool(a.workers) as p:
        for i, s in enumerate(p.imap(job, jobs, chunksize=4)):
            out.append(s)
            print(f"[{i+1}/{len(jobs)}] {s['track']} {s['condition']} {s['controller']} v={s['speed']} s{s['seed']}: "
                  f"{s['status']} rms={s['cte_rms_cm']:.2f}cm max={s['cte_max_cm']:.1f}cm", flush=True)
    with open(os.path.join(ROOT, "results", "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"done {len(out)} runs in {time.time()-t0:.0f} s")
