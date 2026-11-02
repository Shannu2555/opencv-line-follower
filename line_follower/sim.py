"""Small 2D world: diff-drive robot (TurtleBot3 Waffle Pi limits) with motor lag, wheel slip and speed limits, plus the camera
render of the floor. Ground truth (cross-track / heading error, progress) is computed here only."""
import time
import numpy as np
import cv2
from .track import Track, PPM, FLOOR_W, FLOOR_H, disturbance_maps, global_gain
from .camera_model import CameraModel

# TurtleBot3 Waffle Pi numbers (ROBOTIS e-manual): wheel separation 287 mm, wheel radius 33 mm,
# max linear 0.26 m/s, max angular 1.82 rad/s. XM430-W210 at 77 rpm -> ~0.266 m/s at the tyre.
WHEEL_BASE = 0.287
WHEEL_RADIUS = 0.033
V_MAX = 0.26
W_MAX = 1.82
WHEEL_MAX = 0.266   # m/s per wheel
MOTOR_TAU = 0.06    # s
OFF_TRACK = 0.12    # m from the centreline counts as leaving the track


class World:
    def __init__(self, track_name, condition="clean", seed=0, cam=None):
        self.track = Track(track_name)
        self.condition = condition
        self.cam = cam or CameraModel()
        floor = self.track.render_floor(seed=7).astype(np.float32)
        shade, glare = disturbance_maps(floor.shape, condition, seed=seed)
        self.floor_lit = np.clip(floor * shade[..., None], 0, 255).astype(np.uint8)
        self.glare = np.clip(glare, 0, 255).astype(np.uint8)
        self.shade = shade
        self.gx, self.gy = self.cam.ground_lut()
        self.sky = ~np.isfinite(self.gx)
        gx = np.nan_to_num(self.gx)
        gy = np.nan_to_num(self.gy)
        self._gx, self._gy = gx, gy
        h, w = gx.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
        self.vignette = (1 - 0.28 * r ** 2).astype(np.float32)[..., None]
        self.noise_sigma = 6.0 if condition == "hard" else 2.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def camera_image(self, pose, t):
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        wx = x + c * self._gx - s * self._gy
        wy = y + s * self._gx + c * self._gy
        mx = ((wx + FLOOR_W / 2) * PPM).astype(np.float32)
        my = ((FLOOR_H / 2 - wy) * PPM).astype(np.float32)
        mx[self.sky] = -10
        my[self.sky] = -10
        img = cv2.remap(self.floor_lit, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(70, 75, 80)).astype(np.float32)
        gl = cv2.remap(self.glare, mx, my, cv2.INTER_LINEAR).astype(np.float32)
        img = img * global_gain(t, self.condition) * self.vignette + gl[..., None]
        img[self.sky] = (120, 125, 130)
        img += self.rng.normal(0, self.noise_sigma, img.shape).astype(np.float32)
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
        return np.clip(img, 0, 255).astype(np.uint8)


class Robot:
    def __init__(self, pose, seed=0):
        self.x, self.y, self.th = pose
        self.vl = self.vr = 0.0
        self.rng = np.random.default_rng(500 + seed)

    @property
    def pose(self):
        return (self.x, self.y, self.th)

    def step(self, v, w, dt, sub=4):
        v = float(np.clip(v, -V_MAX, V_MAX))
        w = float(np.clip(w, -W_MAX, W_MAX))
        vl_c, vr_c = v - w * WHEEL_BASE / 2, v + w * WHEEL_BASE / 2
        m = max(abs(vl_c), abs(vr_c)) / WHEEL_MAX
        if m > 1:  # wheel saturated: slow both wheels, keep the commanded curvature
            vl_c, vr_c = vl_c / m, vr_c / m
        h = dt / sub
        a = h / (MOTOR_TAU + h)
        slip = 1 + self.rng.normal(0, 0.02, 2)
        for _ in range(sub):
            self.vl += a * (vl_c - self.vl)
            self.vr += a * (vr_c - self.vr)
            vl, vr = self.vl * slip[0], self.vr * slip[1]
            vv = (vl + vr) / 2
            ww = (vr - vl) / WHEEL_BASE
            self.x += vv * np.cos(self.th + ww * h / 2) * h
            self.y += vv * np.sin(self.th + ww * h / 2) * h
            self.th += ww * h
        return (self.vl + self.vr) / 2, (self.vr - self.vl) / WHEEL_BASE


class Progress:
    """Nearest centreline sample in a window around the last one (so the figure-eight crossing
    doesn't make us jump branches). Signed cross-track error, + = robot left of the line."""

    def __init__(self, track, i0=0):
        self.t = track
        self.i = i0
        self.travel = 0.0
        self.n = len(track.pts)

    def update(self, x, y, th):
        win = np.arange(self.i - 60, self.i + 160) % self.n
        d = self.t.pts[win] - (x, y)
        k = int(np.argmin((d ** 2).sum(1)))
        j = int(win[k])
        step = (j - self.i) % self.n
        if step > self.n // 2:
            step -= self.n
        self.travel += step * self.t.length / self.n
        self.i = j
        tang = self.t.tangent[j]
        dx, dy = x - self.t.pts[j, 0], y - self.t.pts[j, 1]
        cte = -np.sin(tang) * dx + np.cos(tang) * dy
        he = np.angle(np.exp(1j * (th - tang)))
        return float(cte), float(he), j


def run_lap(track_name, condition, speed, perception, controller, seed=0, dt=1 / 30,
            latency_frames=1, timeout_factor=3.0, world=None, on_frame=None):
    """One lap. Returns (per-frame log as dict of arrays, summary dict)."""
    world = world or World(track_name, condition, seed)
    tr = world.track
    # reset the camera noise stream every lap. I used to cache World objects between runs and the
    # noise carried over, so the same seed gave a different lap depending on run order.
    world.rng = np.random.default_rng(world.seed)
    rng = np.random.default_rng(seed)
    i0 = int(rng.integers(0, len(tr.pts)))
    off = rng.uniform(-0.02, 0.02)
    p0 = tr.pts[i0]
    t0 = tr.tangent[i0]
    robot = Robot((p0[0] - np.sin(t0) * off, p0[1] + np.cos(t0) * off, t0 + rng.uniform(-0.05, 0.05)), seed)
    prog = Progress(tr, i0)
    perception.reset()
    controller.reset()
    queue = [(0.0, 0.0)] * latency_frames
    log = {k: [] for k in ("t", "x", "y", "th", "cte", "head_err", "v", "w", "v_cmd", "w_cmd",
                           "det_ok", "lost_frames", "est_offset", "est_heading", "est_curv", "lat_ms", "idx")}
    t = 0.0
    t_max = timeout_factor * tr.length / speed
    status = "timeout"
    while t < t_max:
        img = world.camera_image(robot.pose, t)
        est = perception.process(img, keep_debug=on_frame is not None)
        cmd = controller(est)
        queue.append(cmd)
        v_cmd, w_cmd = queue.pop(0)
        v, w = robot.step(v_cmd, w_cmd, dt)
        t += dt
        cte, he, j = prog.update(*robot.pose)
        for k, val in (("t", t), ("x", robot.x), ("y", robot.y), ("th", robot.th), ("cte", cte), ("head_err", he),
                       ("v", v), ("w", w), ("v_cmd", v_cmd), ("w_cmd", w_cmd), ("det_ok", int(est.lost_frames == 0)),
                       ("lost_frames", est.lost_frames), ("est_offset", est.offset), ("est_heading", est.heading),
                       ("est_curv", est.curvature), ("lat_ms", est.latency_ms), ("idx", j)):
            log[k].append(val)
        if on_frame is not None:
            on_frame(t, robot, img, est, cte)
        if abs(cte) > OFF_TRACK:
            status = "off_track"
            break
        if prog.travel >= tr.length:
            status = "completed"
            break
        if prog.travel < -0.5:
            status = "reversed"
            break
    log = {k: np.asarray(v) for k, v in log.items()}
    cte = log["cte"]
    lat = log["lat_ms"]
    summary = dict(track=track_name, condition=condition, speed=speed, controller=controller.name, seed=seed,
                   status=status, completed=status == "completed", lap_time=float(t) if status == "completed" else None,
                   distance=float(prog.travel), frac_done=float(min(1.0, prog.travel / tr.length)),
                   cte_rms_cm=float(np.sqrt(np.mean(cte ** 2)) * 100), cte_max_cm=float(np.abs(cte).max() * 100),
                   head_rms_deg=float(np.degrees(np.sqrt(np.mean(log["head_err"] ** 2)))),
                   det_rate=float(log["det_ok"].mean()), lat_med_ms=float(np.median(lat)),
                   lat_p95_ms=float(np.percentile(lat, 95)), frames=int(len(cte)))
    return log, summary
