"""Steering laws. All return (v, omega); omega > 0 turns left. Line offsets are + when the line is left."""
import numpy as np


class CentroidPID:
    """Baseline: PID on the normalised pixel error of the centroid."""
    name = "centroid_pid"

    def __init__(self, v, kp=2.2, ki=0.0, kd=0.12, dt=1 / 30):
        self.v, self.kp, self.ki, self.kd, self.dt = v, kp, ki, kd, dt
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev = None
        self.last = 0.0

    def __call__(self, est):
        if not est.ok:
            return self.v, self.last  # keep turning the way we were (classic "search")
        e = est.offset
        d = 0.0 if self.prev is None else (e - self.prev) / self.dt
        self.prev = e
        self.i = np.clip(self.i + e * self.dt, -1, 1)
        self.last = self.kp * e + self.ki * self.i + self.kd * d
        return self.v, self.last


class LookaheadPID:
    """PID on the metric lateral error of the fitted line at a fixed look distance."""
    name = "pid"

    def __init__(self, v, look_f=0.30, kp=7.0, ki=0.5, kd=0.35, dt=1 / 30):
        self.v, self.look_f, self.kp, self.ki, self.kd, self.dt = v, look_f, kp, ki, kd, dt
        self.reset()

    def reset(self):
        self.i, self.prev, self.last = 0.0, None, 0.0

    def __call__(self, est):
        if not est.ok:
            return self.v, self.last
        e = est.y_at(self.look_f)
        d = 0.0 if self.prev is None else (e - self.prev) / self.dt
        self.prev = e
        self.i = float(np.clip(self.i + e * self.dt, -0.2, 0.2))
        self.last = self.kp * e + self.ki * self.i + self.kd * d
        return self.v, self.last


class PurePursuit:
    """Arc through a look-ahead point on the fitted line; look-ahead grows with speed."""
    name = "pure_pursuit"

    def __init__(self, v, ld_min=0.22, ld_gain=0.45, ld_max=0.55):
        self.v, self.ld_min, self.ld_gain, self.ld_max = v, ld_min, ld_gain, ld_max
        self.last = 0.0

    def reset(self):
        self.last = 0.0

    def __call__(self, est):
        if not est.ok:
            return self.v, self.last
        ld = float(np.clip(self.ld_min + self.ld_gain * self.v, self.ld_min, self.ld_max))
        f = ld
        for _ in range(3):  # solve for the point at euclidean distance ld
            y = est.y_at(f)
            f = np.sqrt(max(ld * ld - y * y, 0.01))
        y = est.y_at(f)
        self.last = 2.0 * self.v * y / (f * f + y * y)
        return self.v, self.last


class StanleyFF:
    """Stanley-style law for a diff drive with curvature feed-forward:
    omega = v*kappa + k_h*heading + k_e*atan(k*offset/(v+v0))."""
    name = "stanley_ff"

    def __init__(self, v, k_h=3.0, k_e=2.5, k=2.5, v0=0.15, kff=1.0):
        self.v, self.k_h, self.k_e, self.k, self.v0, self.kff = v, k_h, k_e, k, v0, kff
        self.last = 0.0

    def reset(self):
        self.last = 0.0

    def __call__(self, est):
        if not est.ok:
            return self.v, self.last
        self.last = (self.kff * self.v * est.curvature + self.k_h * est.heading
                     + self.k_e * np.arctan(self.k * est.offset / (self.v + self.v0)))
        return self.v, self.last


CONTROLLERS = {c.name: c for c in (CentroidPID, LookaheadPID, PurePursuit, StanleyFF)}
