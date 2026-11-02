"""Line detection: undistort -> bird's-eye (IPM) -> illumination-normalised threshold ->
sliding windows gated by the previous fit -> RANSAC quadratic -> offset / heading / curvature.

Also contains the classic "threshold + centroid" detector used as the baseline."""
import time
import numpy as np
import cv2

DEFAULTS = dict(
    f_min=0.16, f_max=0.66, y_half=0.25, bev_res=0.004,
    bg_kernel=15, ratio_thresh=0.66, min_blob=12,
    n_windows=10, win_margin=0.045, min_pix=6,
    ransac_iters=40, ransac_tol=0.008, min_inliers=18,
    ema=0.55, max_hold=15, look_f=0.30,
)


class LineEstimate:
    __slots__ = ("ok", "coef", "offset", "heading", "curvature", "conf", "lost_frames", "latency_ms", "dbg")

    def __init__(self):
        self.ok = False
        self.coef = None
        self.offset = self.heading = self.curvature = 0.0
        self.conf = 0.0
        self.lost_frames = 0
        self.latency_ms = 0.0
        self.dbg = {}

    def y_at(self, f):
        c = self.coef
        return c[0] + c[1] * f + c[2] * f * f


def fit_ransac_quadratic(f, y, iters=40, tol=0.012, rng=None):
    """RANSAC fit of y = c0 + c1 f + c2 f^2. Returns (coef, inlier_mask) or (None, None)."""
    n = len(f)
    if n < 6:
        return None, None
    rng = np.random.default_rng(0) if rng is None else rng
    best, best_cnt = None, -1
    A = np.column_stack([np.ones(n), f, f * f])
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            c = np.linalg.solve(A[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        if abs(c[2]) > 12:  # radius under ~4 cm is not a line we can follow
            continue
        inl = np.abs(A @ c - y) < tol
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best, best_cnt = inl, cnt
    if best is None or best_cnt < 6:
        return None, None
    span = f[best].max() - f[best].min()
    if span < 0.12:  # short segment: a curve term would be pure noise
        c1, c0 = np.polyfit(f[best], y[best], 1)
        return np.array([c0, c1, 0.0]), best
    c2, c1, c0 = np.polyfit(f[best], y[best], 2)
    return np.array([c0, c1, c2]), best


class LinePerception:
    def __init__(self, K, D, cam_R, cam_C, size, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        p = self.p
        w, h = size
        self.K, self.D = K, D
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
        self.bw = int(round(2 * p["y_half"] / p["bev_res"]))
        self.bh = int(round((p["f_max"] - p["f_min"]) / p["bev_res"]))
        # BEV corners (col,row) <-> ground (f, y)
        bev = np.float32([[0, 0], [self.bw, 0], [self.bw, self.bh], [0, self.bh]])
        ground = np.array([self.px_to_ground(c, r) for c, r in bev])
        P = np.column_stack([ground, np.zeros(4)])
        pc = (P - cam_C) @ cam_R
        img = (K @ (pc / pc[:, 2:3]).T).T[:, :2].astype(np.float32)
        self.H = cv2.getPerspectiveTransform(img, bev)
        self.valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), self.H, (self.bw, self.bh),
                                         flags=cv2.INTER_NEAREST)
        self.valid = cv2.erode(self.valid, np.ones((5, 5), np.uint8))
        self.kbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["bg_kernel"], p["bg_kernel"]))
        self.rows_f = p["f_max"] - np.arange(self.bh) * p["bev_res"]
        self.cols_y = p["y_half"] - np.arange(self.bw) * p["bev_res"]
        self.prev = None
        self.lost = 0
        self.rng = np.random.default_rng(1)

    def px_to_ground(self, col, row):
        p = self.p
        return p["f_max"] - row * p["bev_res"], p["y_half"] - col * p["bev_res"]

    def reset(self):
        self.prev, self.lost = None, 0

    def birdseye(self, bgr):
        und = cv2.remap(bgr, self.map1, self.map2, cv2.INTER_LINEAR)
        return und, cv2.warpPerspective(und, self.H, (self.bw, self.bh), flags=cv2.INTER_LINEAR)

    def segment(self, bev_bgr):
        """Dark-line mask that ignores slow lighting changes and soft shadows.
        V channel / local background (grey closing removes the thin dark line, keeps shadows)."""
        p = self.p
        v = bev_bgr.max(axis=2)
        bg = cv2.morphologyEx(v, cv2.MORPH_CLOSE, self.kbg)
        ratio = v.astype(np.float32) / (bg.astype(np.float32) + 4.0)
        m = ((ratio < p["ratio_thresh"]) & (self.valid > 0) & (bg > 25)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros(n, np.uint8)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= p["min_blob"]
        return keep[lab]

    def sliding_window(self, mask):
        p = self.p
        res = p["bev_res"]
        margin = int(p["win_margin"] / res)
        wh = self.bh // p["n_windows"]
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return np.array([]), np.array([]), []
        if self.prev is not None:
            def pred(row):
                f = self.rows_f[min(row, self.bh - 1)]
                return (p["y_half"] - self.prev.y_at(f)) / res
            x_cur = pred(self.bh - wh // 2)
        else:
            low = mask[self.bh * 6 // 10:]
            hist = low.sum(0)
            if hist.max() == 0:
                hist = mask.sum(0)
            x_cur = float(np.argmax(np.convolve(hist, np.ones(7), "same")))
            pred = None
        sel = []
        wins = []
        last = []
        for k in range(p["n_windows"]):
            r1 = self.bh - k * wh
            r0 = r1 - wh
            if pred is not None:
                x_cur = 0.5 * x_cur + 0.5 * pred((r0 + r1) // 2) if k else x_cur
            elif len(last) >= 2:
                x_cur = x_cur + (last[-1] - last[-2])
            inw = (ys >= r0) & (ys < r1) & (np.abs(xs - x_cur) < margin)
            idx = np.nonzero(inw)[0]
            wins.append((int(x_cur - margin), r0, int(x_cur + margin), r1, len(idx) >= p["min_pix"]))
            if len(idx) >= p["min_pix"]:
                cx = xs[idx]
                # a crossing line fills the window sideways: then trust the prediction, take nearest pixels
                if cx.max() - cx.min() > 1.4 * margin:
                    idx = idx[np.abs(cx - x_cur) < margin * 0.35]
                if len(idx) >= p["min_pix"]:
                    sel.append(idx)
                    x_cur = float(np.median(xs[idx]))
                    last.append(x_cur)
        if not sel:
            return np.array([]), np.array([]), wins
        idx = np.concatenate(sel)
        # one point per bird's-eye row: the centre of the line, not its two edges
        cnt = np.bincount(ys[idx], minlength=self.bh)
        sx = np.bincount(ys[idx], weights=xs[idx], minlength=self.bh)
        rows = np.nonzero(cnt >= 2)[0]
        cx = sx[rows] / cnt[rows]
        f = self.rows_f[rows]
        y = p["y_half"] - cx * res
        return f, y, wins

    def process(self, bgr, keep_debug=False):
        t0 = time.perf_counter()
        p = self.p
        und, bev = self.birdseye(bgr)
        mask = self.segment(bev)
        f, y, wins = self.sliding_window(mask)
        coef, inl = fit_ransac_quadratic(f, y, p["ransac_iters"], p["ransac_tol"], self.rng)
        est = LineEstimate()
        if coef is not None and inl.sum() >= p["min_inliers"]:
            if self.prev is not None and self.prev.ok:
                coef = p["ema"] * coef + (1 - p["ema"]) * self.prev.coef
            est.ok = True
            est.coef = coef
            est.conf = float(min(1.0, inl.sum() / 100))
            self.lost = 0
        elif self.prev is not None and self.lost < p["max_hold"]:
            self.lost += 1
            est.ok = True  # coasting on the last fit
            est.coef = self.prev.coef
            est.conf = 0.0
        else:
            self.lost += 1
        est.lost_frames = self.lost
        if est.coef is not None:
            c0, c1, c2 = est.coef
            est.offset = float(c0)
            est.heading = float(np.arctan(c1))
            est.curvature = float(2 * c2 / (1 + c1 * c1) ** 1.5)
            self.prev = est
        else:
            self.prev = None
        est.latency_ms = (time.perf_counter() - t0) * 1000
        if keep_debug:
            est.dbg = dict(und=und, bev=bev, mask=mask, wins=wins, f=f, y=y, inl=inl)
        return est


class CentroidPerception:
    """Classic tutorial follower: grey, fixed threshold, bottom ROI, biggest contour centroid (pixels)."""

    def __init__(self, size, thresh=80, roi=(0.55, 0.95)):
        self.w, self.h = size
        self.thresh = thresh
        self.roi = roi
        self.lost = 0

    def reset(self):
        self.lost = 0

    def process(self, bgr, keep_debug=False):
        t0 = time.perf_counter()
        r0, r1 = int(self.roi[0] * self.h), int(self.roi[1] * self.h)
        g = cv2.cvtColor(bgr[r0:r1], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        _, m = cv2.threshold(g, self.thresh, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        est = LineEstimate()
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 20:
                cx = M["m10"] / M["m00"]
                est.ok = True
                est.offset = (self.w / 2 - cx) / (self.w / 2)  # normalised, + = line on the left
                self.lost = 0
                if keep_debug:
                    est.dbg = dict(cx=cx, cy=M["m01"] / M["m00"] + r0, contour=c + [0, r0], roi=(r0, r1), mask=m)
        if not est.ok:
            self.lost += 1
        est.lost_frames = self.lost
        est.latency_ms = (time.perf_counter() - t0) * 1000
        return est
