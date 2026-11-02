# opencv-line-follower

Camera-based line following for a small diff-drive robot, ROS 2 Humble style. I wanted to see how far a proper OpenCV pipeline (calibration, bird's-eye view, sliding windows, RANSAC fit) gets you compared to the usual "threshold + biggest contour + PID" line follower, and whether the steering law matters once the perception is decent.

**Honest note on the setup:** I don't have the robot in hand right now. The robot is a TurtleBot3 Waffle Pi-style model in a small sim I wrote myself (`line_follower/sim.py`). It uses the Waffle Pi's real numbers: 0.287 m wheel base, 33 mm wheels, 0.26 m/s and 1.82 rad/s max. The camera is a simulated pinhole + barrel-distortion lens on a mast, 0.20 m up, tilted down 45°. Everything after the camera image is real OpenCV code that would run the same on a real camera feed. There's no Gazebo here, and no physics beyond diff-drive kinematics with wheel speed limits.

## What's in here

```
line_follower/
  camera_model.py   pinhole + distortion lens, mount pose, floor <-> pixel maps
  calibration.py    checkerboard calibration (cv2.calibrateCamera)
  perception.py     undistort -> bird's-eye (IPM) -> shadow-tolerant mask -> sliding windows -> RANSAC quadratic
  controllers.py    centroid PID (baseline), look-ahead PID, pure pursuit, Stanley + curvature feed-forward
  sim.py            diff-drive robot, renders the camera view from the floor texture
  track.py          3 tracks (oval with chicane, figure eight, technical), dashed stretches, lighting disturbances
  nodes/            camera_sim_node, perception_node, control_node (rclpy)
experiments/        calibrate_camera.py, run_experiments.py, perception_accuracy.py, analyze.py
results/            summary.csv (one row per run), results_table.md, calibration.json, perception accuracy
logs/               per-run CSV logs (pose, cross-track error, latency for every frame)
plots/              all charts
video/              recording + 3D render scripts for the demo video
```

## The pipeline

1. **Calibrate.** 25 checkerboard views, 20 detected, reprojection RMS 0.14 px. Undistorting with my estimate vs the true lens: 0.66 px mean error.
2. **Undistort + bird's-eye.** Warp the floor 0.16 to 0.66 m ahead of the robot into a 125x125 top view (4 mm per pixel).
3. **Mask.** A fixed threshold dies under shadows, so I divide the V channel by a local background (grey closing that removes the thin line but keeps the shadow), then threshold that ratio. Glare doesn't make anything darker than its surroundings, so it mostly drops out, and tiny blobs get removed by an area filter.
4. **Sliding windows + RANSAC.** Windows walk up the image from the bottom, then a RANSAC quadratic in floor coordinates gives offset, heading and curvature.
5. **Control.** Same perception output into 4 controllers. The baseline uses its own centroid perception.

## Experiment

3 tracks x clean/hard lighting x 0.12/0.19/0.26 m/s x 4 controllers x 3 seeds = **216 runs**. "Hard" means soft cast shadows (partial, blurred, the line stays visible through them), glare spots from ceiling lights, and a slow global brightness swing. A run fails if the robot gets 12 cm off the true centreline.

| controller | laps done | hard light | CTE RMS (cm) | CTE max (cm) | heading RMS (°) | perception ms |
|---|---|---|---|---|---|---|
| Centroid PID (baseline) | 27/54 | 3/27 | 0.88 | 11.2 | 1.7 | 0.09 |
| Fit + look-ahead PID | 52/54 | 25/27 | 2.64 | 10.5 | 2.5 | 1.93 |
| Fit + pure pursuit | 51/54 | 24/27 | 0.86 | 12.0 | 1.9 | 1.92 |
| Fit + Stanley + FF | 49/54 | 24/27 | 0.48 | 8.8 | 1.0 | 1.93 |

Errors are vs sim ground truth, completed laps only. On the 8 cases all four finished on every seed, Stanley + FF averaged 0.47 cm RMS vs 2.27 cm for the look-ahead PID. Full tables: `results/results_table.md`.

Static perception check (robot placed on the true centreline, 56 to 100 poses per track): lateral error is about 3 to 5 mm RMS at 20 cm ahead. It gets bad far out on tight curves (47 to 57 mm at 45 cm on the technical track), which is why the look-ahead point sits at 30 cm.

## What broke (and what I still haven't fixed)

- **The baseline only finished 3 of 27 hard-lighting laps.** It finishes 24 of 27 clean ones, so it's not the PID. A fixed threshold picks up shadow edges and the biggest contour isn't always the line.
- **Technical track, hard lighting, seed 1:** the fit-based controllers leave the line at almost the same spot (about 38% of the lap) in 6 of 9 runs. All three fail at the same place, so it's a perception problem, not a steering one. That's where a dashed stretch overlaps the lighting disturbances. Not fixed yet.
- **Stanley at 0.12 m/s on the technical track** finished only 1 of 6, mostly going off around 16% of the lap. Faster was actually better for it (6/6 at 0.19 and 0.26). I haven't pinned down why. My guess is the offset term: it's atan(k*e/(v+v0)), so at low speed the gain goes up and it may be over-correcting, but I haven't proven it.
- **Runs weren't reproducible at first.** I cached the World object per worker to save time, and its camera-noise random generator carried over between runs. So the same seed gave a different lap depending on which runs that worker did before. One Stanley run went from "off track at 88%" to "completed" once I reset the generator every lap. Now the video recordings and the matrix give the same result for the same seed.
- Early on I put the camera mast at the middle of the plate, and the camera would have been looking at its own robot. I moved it to the front edge and shifted the bird's-eye window.
- My first shadow model was hard black polygons. They looked like obstacles and weren't realistic, so I replaced them with soft partial shadows and re-ran all 216 runs (again after the seed fix). The numbers above are from the soft-shadow floor only.

## Run it

```bash
pip install -r requirements.txt
python3 -m pytest -q                       # 13 tests
python3 experiments/calibrate_camera.py
python3 experiments/run_experiments.py     # ~11 min on 6 cores
python3 experiments/perception_accuracy.py
python3 experiments/analyze.py
```

With ROS 2 Humble sourced: `colcon build && ros2 launch line_follower line_follower.launch.py`. The nodes are `/camera/image_raw` -> perception_node -> `/line/estimate` -> control_node -> `/cmd_vel`.

Shanmukh Narayana Chaganti, BTech Robotics & AI, Amrita School of Engineering, Bengaluru
