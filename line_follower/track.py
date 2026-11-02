"""Track centrelines, floor texture and lighting disturbances (glare, shadows, global gain)."""
import numpy as np
import cv2
from scipy.interpolate import splprep, splev

FLOOR_W, FLOOR_H = 5.0, 4.0   # metres
PPM = 250                     # floor texture resolution, px per metre (4 mm/px)
LINE_W = 0.019                # 19 mm tape


def to_px(xy):
    xy = np.asarray(xy)
    return np.column_stack([(xy[:, 0] + FLOOR_W / 2) * PPM, (FLOOR_H / 2 - xy[:, 1]) * PPM])


def _resample(pts, ds=0.005):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(s[-1] / ds)
    si = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])])


def _closed(fx, fy, n=4000):
    t = np.linspace(0, 2 * np.pi, n + 1)
    return _resample(np.column_stack([fx(t), fy(t)]))


def _spline(points, n=6000):
    p = np.asarray(points, float)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=0, per=1)
    x, y = splev(np.linspace(0, 1, n + 1), tck)
    return _resample(np.column_stack([x, y]))


TRACKS = {
    # gentle oval with a chicane on one side, one dashed stretch
    "oval_chicane": dict(
        pts=lambda: _closed(lambda t: 1.9 * np.cos(t), lambda t: 1.25 * np.sin(t) + 0.30 * np.sin(3 * t) * ((1 + np.cos(t)) / 2) ** 2),
        dashed=[(0.55, 0.65)]),
    # figure eight: the line crosses itself at 90 deg in the middle
    "figure_eight": dict(
        pts=lambda: _closed(lambda t: 1.9 * np.sin(t), lambda t: 1.45 * np.sin(t) * np.cos(t)),
        dashed=[]),
    # technical: hairpin, S-bends, two dashed stretches
    "technical": dict(
        pts=lambda: _spline([(-2.0, -1.5), (0.0, -1.6), (1.8, -1.5), (2.1, -0.6), (1.5, -0.1), (0.7, -0.6),
                             (0.0, -0.2), (0.5, 0.6), (1.7, 0.8), (1.9, 1.5), (0.6, 1.65), (-0.6, 1.0),
                             (-1.4, 1.6), (-2.2, 1.1), (-1.6, 0.2), (-2.2, -0.6)]),
        dashed=[(0.30, 0.38), (0.72, 0.80)]),
}


class Track:
    def __init__(self, name):
        self.name = name
        spec = TRACKS[name]
        self.pts = spec["pts"]()
        d = np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0)
        self.ds = np.linalg.norm(d, axis=1)
        self.length = float(self.ds.sum())
        self.s = np.concatenate([[0], np.cumsum(self.ds)[:-1]])
        self.tangent = np.arctan2(d[:, 1], d[:, 0])
        self.dashed = spec["dashed"]
        dth = np.angle(np.exp(1j * np.diff(np.concatenate([self.tangent, self.tangent[:1]]))))
        self.curvature = dth / self.ds

    def min_radius(self):
        k = np.convolve(np.abs(self.curvature), np.ones(9) / 9, mode="same")
        return 1.0 / k.max()

    def is_gap(self, frac, on=0.12, off=0.06):
        for a, b in self.dashed:
            if a <= frac <= b:
                return ((frac - a) * self.length) % (on + off) > on
        return False

    def render_floor(self, seed=0):
        rng = np.random.default_rng(seed)
        H, W = int(FLOOR_H * PPM), int(FLOOR_W * PPM)
        base = np.empty((H, W, 3), np.float32)
        base[:] = (196, 206, 214)  # BGR light concrete
        grain = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 3) * 18
        base += grain[..., None]
        img = np.clip(base, 0, 255).astype(np.uint8)
        px = to_px(self.pts)
        S = 4  # sub-pixel drawing
        thick = max(1, int(round(LINE_W * PPM)))
        n = len(px)
        i = 0
        while i < n:
            if self.is_gap(i / n):
                i += 1
                continue
            j = i
            while j + 1 < n and not self.is_gap((j + 1) / n) and j - i < 400:
                j += 1
            seg = np.vstack([px[i:j + 1], px[(j + 1) % n:(j + 2) % n or None][:1]])
            cv2.polylines(img, [np.round(seg * S).astype(np.int32)], False, (28, 26, 30), thick, cv2.LINE_AA, shift=2)
            i = j + 1
        return img


def disturbance_maps(shape, condition, seed=0):
    """Multiplicative shadow map and additive glare map in floor pixels."""
    H, W = shape[:2]
    rng = np.random.default_rng(1000 + seed)
    shade = np.ones((H, W), np.float32)
    glare = np.zeros((H, W), np.float32)
    if condition == "hard":
        for _ in range(9):  # soft cast shadows (furniture, people): partial, blurred, line stays visible
            c = rng.uniform([0, 0], [W, H])
            r = rng.uniform(60, 220)
            k = rng.integers(3, 7)
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            poly = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]).astype(np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 14.0)
            shade *= 1 - rng.uniform(0.22, 0.38) * m
        for _ in range(7):  # specular glare from ceiling lights
            c = rng.uniform([0, 0], [W, H])
            yy, xx = np.mgrid[0:H, 0:W]
            sx, sy = rng.uniform(25, 60, 2)
            glare += rng.uniform(170, 260) * np.exp(-(((xx - c[0]) / sx) ** 2 + ((yy - c[1]) / sy) ** 2))
    return shade, glare


def global_gain(t, condition):
    if condition != "hard":
        return 1.0
    return 0.9 + 0.35 * np.sin(2 * np.pi * t / 7.0) + 0.1 * np.sin(2 * np.pi * t / 1.3)
