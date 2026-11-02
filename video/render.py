#!/usr/bin/env python3
"""Demo video renderer. Draws at 2x (2160 px) and downsamples for anti-aliasing, pipes raw frames
into ffmpeg. Works on Linux and Windows; set ENC=nvenc to use the NVIDIA encoder for the master.
usage: python video/render.py OUT_MASTER.mp4"""
import json, os, pickle, subprocess, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from line_follower.camera_model import CameraModel
from line_follower.track import PPM, FLOOR_W, FLOOR_H
from line_follower.sim import WHEEL_BASE, WHEEL_RADIUS
from scene3d import Renderer, look_at, perspective, ortho

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REC = os.path.join(ROOT, "video", "rec")
S = 2
W = H = 1080
FPS = 30
NAME = "Shanmukh Narayana Chaganti"
HONEST = "TurtleBot3 Waffle Pi-style robot modelled in my own sim  ·  real OpenCV pipeline on every frame  ·  ROS 2 nodes"

# colours (RGB), one per element, same everywhere
BG = (248, 250, 252); TXT = (15, 23, 42); MUTE = (71, 85, 105); PANEL = (255, 255, 255); EDGE = (203, 213, 225)
C_ROBOT = (14, 165, 233); C_ROBOT_EDGE = (8, 47, 73); C_FOV = (250, 204, 21); C_FIT = (217, 70, 239)
C_WIN = (34, 211, 238); C_MASK = (34, 197, 94); C_OFF = (220, 38, 38)
C_AMBER = (245, 158, 11); C_DARK = (15, 23, 42)
C_CTRL = {"centroid_pid": (100, 116, 139), "pid": (249, 115, 22), "pure_pursuit": (22, 163, 74), "stanley_ff": (37, 99, 235)}
LBL = {"centroid_pid": "Baseline: threshold + centroid + PID", "pid": "Bird's-eye fit + PID",
       "pure_pursuit": "Bird's-eye fit + pure pursuit", "stanley_ff": "Bird's-eye fit + Stanley + curvature FF"}
ROBOT_SCALE = 1.6

FONT_DIRS = ["/usr/share/fonts/opentype/inter", os.path.join(ROOT, "video", "fonts"), "C:/Windows/Fonts"]


def font(weight, size):
    names = {"r": ["Inter-Regular.otf", "segoeui.ttf"], "m": ["Inter-Medium.otf", "seguisb.ttf"],
             "b": ["Inter-Bold.otf", "segoeuib.ttf"], "sb": ["Inter-SemiBold.otf", "seguisb.ttf"]}[weight]
    for d in FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size * S)
    return ImageFont.load_default()


F = {k: font(*k.split("_")[0:1], int(k.split("_")[1])) if False else None for k in []}
_fc = {}


def f(w, s):
    if (w, s) not in _fc:
        _fc[(w, s)] = font(w, s)
    return _fc[(w, s)]


def s_(v):
    return int(round(v * S))


class Canvas:
    def __init__(self):
        self.a = np.empty((H * S, W * S, 3), np.uint8)
        self.a[:] = BG

    def rect(self, x0, y0, x1, y1, col, fill=True, w=1, radius=0):
        if radius:
            im = Image.fromarray(self.a)
            ImageDraw.Draw(im).rounded_rectangle([s_(x0), s_(y0), s_(x1), s_(y1)], s_(radius),
                                                 fill=col if fill else None, outline=None if fill else col, width=s_(w))
            self.a = np.asarray(im).copy()
        else:
            cv2.rectangle(self.a, (s_(x0), s_(y0)), (s_(x1), s_(y1)), col, -1 if fill else s_(w), cv2.LINE_AA)

    def paste(self, img, x0, y0, w, h, interp=cv2.INTER_AREA):
        r = cv2.resize(img, (s_(w), s_(h)), interpolation=interp)
        self.a[s_(y0):s_(y0) + r.shape[0], s_(x0):s_(x0) + r.shape[1]] = r

    def text(self, items):
        """items: list of (x, y, str, font, colour, anchor)."""
        im = Image.fromarray(self.a)
        d = ImageDraw.Draw(im)
        for x, y, t, fo, col, anc in items:
            d.text((s_(x), s_(y)), t, font=fo, fill=col, anchor=anc)
        self.a = np.asarray(im).copy()

    def out(self):
        return cv2.resize(self.a, (W, H), interpolation=cv2.INTER_AREA)


def fade(img, k):
    if k >= 1:
        return img
    return (img.astype(np.float32) * k + np.array(BG, np.float32) * (1 - k)).astype(np.uint8)


# ------------------------------------------------------------------ run scenes
CH = (20, 96, 760, 520)    # 3D chase view
TOP = (20, 532, 400, 836)  # top view, 5 x 4 m
TOP_SCALE = 1.8


cam = CameraModel()
calib = json.load(open(os.path.join(ROOT, "results", "calibration.json")))
K_est, D_est = np.array(calib["K"]), np.array(calib["D"])
gx, gy = cam.ground_lut()
# camera footprint on the floor (image border, clipped to 0.9 m ahead)
border = np.concatenate([np.column_stack([gx[-1, :], gy[-1, :]]), np.column_stack([gx[:, -1], gy[:, -1]])[::-1],
                         np.column_stack([gx[0, :], gy[0, :]])[::-1], np.column_stack([gx[:, 0], gy[:, 0]])])
border = border[np.isfinite(border).all(1)]
border = border[np.hypot(border[:, 0], border[:, 1]) < 0.9]
from scipy.spatial import ConvexHull
FOOT = border[ConvexHull(border).vertices]
BEV_F = (0.66, 0.16)  # top / bottom forward distance of bird's-eye image
BEV_Y = 0.25


def poly(c, pts, col, w, closed=False):
    p = np.round(np.asarray(pts) * S * 4).astype(np.int32)
    cv2.polylines(c.a, [p], closed, col, s_(w), cv2.LINE_AA, shift=2)


def fill_poly(c, pts, col, alpha):
    p = np.round(np.asarray(pts) * S * 4).astype(np.int32)
    ov = c.a.copy()
    cv2.fillPoly(ov, [p], col, cv2.LINE_AA, shift=2)
    c.a = cv2.addWeighted(ov, alpha, c.a, 1 - alpha, 0)


def to_world(pose, pts):
    x, y, th = pose
    cth, sth = np.cos(th), np.sin(th)
    return np.column_stack([x + cth * pts[:, 0] - sth * pts[:, 1], y + sth * pts[:, 0] + cth * pts[:, 1]])


class RunScene:
    def __init__(self, name, title, sub, plan, freeze_note=None):
        d = pickle.load(open(os.path.join(REC, name + ".pkl"), "rb"))
        self.d = d
        self.fr = d["frames"]
        self.s = d["summary"]
        self.ctrl = self.s["controller"]
        self.title, self.sub = title, sub
        self.freeze_note = freeze_note
        floor = cv2.imread(os.path.join(REC, f"floor_{d['track']}_{self.s['condition']}_s{self.s['seed']}.png"))
        floor = cv2.cvtColor(floor, cv2.COLOR_BGR2RGB)
        g = floor.reshape(-1, 3).astype(np.float32); lum = g.mean(1)
        base = np.median(g[(lum > np.percentile(lum, 20)) & (lum < np.percentile(lum, 70))], 0)
        self.sw_floor = tuple(int(v) for v in base)
        self.sw_line = tuple(int(v) for v in np.median(g[lum <= np.percentile(lum, 1.0)], 0))
        self.sw_shadow = tuple(int(v) for v in base * 0.70)
        self.sw_glare = tuple(int(v) for v in np.median(g[lum >= np.percentile(lum, 99.7)], 0))
        from line_follower.track import Track as _T
        self.has_dash = bool(_T(d["track"]).dashed)
        self.gl = Renderer(floor, FLOOR_W, FLOOR_H, (cam.mount_x, cam.mount_height, cam.pitch_deg))
        P = np.array([fr["pose"] for fr in self.fr])
        self.xy = P[:, :2]
        ds = np.r_[0, np.hypot(*np.diff(P[:, :2], axis=0).T)]
        dth = np.r_[0, np.diff(np.unwrap(P[:, 2]))]
        self.wang = np.column_stack([np.cumsum(ds - dth * WHEEL_BASE / 2), np.cumsum(ds + dth * WHEEL_BASE / 2)]) / WHEEL_RADIUS
        th = np.unwrap(P[:, 2]); hs = np.empty_like(th); hs[0] = th[0]
        a = (1 / FPS) / (0.8 + 1 / FPS)
        for j in range(1, len(th)):
            hs[j] = hs[j - 1] + a * (th[j] - hs[j - 1])
        self.hs = hs
        self.t_end = self.fr[-1]["t"]
        # plan: list of (speed, n_sim_frames or None=rest) -> list of (sim index, speed label)
        seq, i = [], 0
        for sp, n in plan:
            end = len(self.fr) if n is None else min(len(self.fr), i + n)
            while i < end:
                seq.append((i, sp))
                i += sp
        if seq[-1][0] != len(self.fr) - 1:
            seq.append((len(self.fr) - 1, seq[-1][1]))
        self.seq = seq
        self.cte_hist = np.array([fr["cte"] for fr in self.fr])
        self.ts = np.array([fr["t"] for fr in self.fr])

    def n_frames(self):
        return len(self.seq) + (int(2.5 * FPS) if self.freeze_note else int(0.8 * FPS))

    def frame(self, k):
        if k < len(self.seq):
            i, sp = self.seq[k]
            frozen = False
        else:
            i, sp = len(self.fr) - 1, None
            frozen = True
        return self.draw(i, sp, frozen, k - len(self.seq))

    def draw(self, i, sp, frozen, kf):
        fr = self.fr[i]
        c = Canvas()
        col = C_CTRL[self.ctrl]
        # header
        c.text([(20, 22, self.title, f("b", 30), TXT, "la"), (20, 62, self.sub, f("r", 17), MUTE, "la")])
        if sp is not None:
            lab = "1× real time" if sp == 1 else f"{sp}× time-lapse"
            bc = (16, 185, 129) if sp == 1 else (124, 58, 237)
            c.rect(880, 26, 1060, 66, bc, radius=20)
            c.text([(970, 46, lab, f("sb", 19), (255, 255, 255), "mm")])
        # 3D chase view
        pose = fr["pose"]
        x, y = pose[:2]
        hd = self.hs[i]
        dv, pv = np.array([np.cos(hd), np.sin(hd)]), np.array([-np.sin(hd), np.cos(hd)])
        eye = np.r_[np.array([x, y]) - 0.62 * dv - 0.30 * pv, 0.42]
        tgt = np.r_[np.array([x, y]) + 0.30 * dv, 0.02]
        foot = np.column_stack([to_world(pose, FOOT), np.zeros(len(FOOT))])
        apex = np.r_[to_world(pose, np.array([[cam.mount_x, 0.0]]))[0], cam.mount_height]
        fitw = None
        if fr["coef"] is not None and fr.get("mask") is not None:
            ff = np.linspace(0.16, 0.66, 30); cc = fr["coef"]
            fitw = to_world(pose, np.column_stack([ff, cc[0] + cc[1] * ff + cc[2] * ff * ff]))
        trail = self.xy[max(0, i - 6000):i + 1] if i > 1 else None
        tc = tuple(v / 255 for v in col)
        cw, ch = CH[2] - CH[0], CH[3] - CH[1]
        img = self.gl.render((s_(cw), s_(ch)), look_at(eye, tgt), perspective(42, cw / ch, 0.03, 40), eye, pose, self.wang[i],
                             trail=trail, trail_col=tc, fit=fitw, frustum=(apex, foot))
        c.a[s_(CH[1]):s_(CH[1]) + img.shape[0], s_(CH[0]):s_(CH[0]) + img.shape[1]] = img
        c.rect(*CH, EDGE, fill=False, w=1.5)
        tw, th_ = TOP[2] - TOP[0], TOP[3] - TOP[1]
        eye2 = np.array([0, 0, 6.0])
        img = self.gl.render((s_(tw), s_(th_)), look_at(eye2, [0, 0, 0], (0, 1, 0)), ortho(-FLOOR_W / 2, FLOOR_W / 2, -FLOOR_H / 2, FLOOR_H / 2, 0.1, 20),
                             eye2, pose, self.wang[i], trail=trail, trail_col=tc, fit=None, frustum=(apex, foot),
                             robot_scale=TOP_SCALE, trail_w=0.03)
        c.a[s_(TOP[1]):s_(TOP[1]) + img.shape[0], s_(TOP[0]):s_(TOP[0]) + img.shape[1]] = img
        c.rect(*TOP, EDGE, fill=False, w=1.5)
        c.rect(CH[0] + 6, CH[1] + 6, CH[0] + 196, CH[1] + 30, (255, 255, 255), radius=6)
        c.rect(TOP[0] + 4, TOP[1] + 4, TOP[0] + 300, TOP[1] + 26, (255, 255, 255), radius=6)
        c.text([(CH[0] + 14, CH[1] + 11, "3D chase view (to scale)", f("sb", 14), TXT, "la"),
                (TOP[0] + 8, TOP[1] + 6, f"Whole track, top view (robot drawn {TOP_SCALE:g}x)", f("sb", 13), TXT, "la")])
        if frozen and self.freeze_note:
            pm = TOP[0] + (x + FLOOR_W / 2) / FLOOR_W * tw, TOP[1] + (FLOOR_H / 2 - y) / FLOOR_H * th_
            cv2.circle(c.a, (s_(pm[0]), s_(pm[1])), s_(18), C_AMBER, s_(4), cv2.LINE_AA)
        # right column: camera + bird's-eye
        X0, CW = 776, 284
        c.text([(X0, 100, "Robot camera (raw, lens distortion)", f("sb", 14), TXT, "la")])
        cimg = cv2.cvtColor(cv2.imdecode(fr["jpg"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        big = cv2.resize(cimg, (s_(CW), s_(CW * 0.75)), interpolation=cv2.INTER_CUBIC)
        sc = s_(CW) / 320
        if self.ctrl == "centroid_pid":
            if fr.get("roi"):
                r0, r1 = fr["roi"]
                cv2.rectangle(big, (0, int(r0 * sc)), (big.shape[1] - 1, int(r1 * sc)), C_WIN, s_(1.5), cv2.LINE_AA)
            if fr.get("contour") is not None:
                cv2.drawContours(big, [(fr["contour"] * sc).astype(np.int32)], -1, C_MASK, s_(2), cv2.LINE_AA)
            if fr.get("cx") is not None:
                cv2.circle(big, (int(fr["cx"] * sc), int(fr["cy"] * sc)), s_(5), C_FIT, -1, cv2.LINE_AA)
                cv2.line(big, (big.shape[1] // 2, int(fr["cy"] * sc)), (int(fr["cx"] * sc), int(fr["cy"] * sc)), C_FIT, s_(2), cv2.LINE_AA)
        elif fr["coef"] is not None:
            ff = np.linspace(0.17, 0.66, 40)
            cc = fr["coef"]
            px = cam.project_ground(np.column_stack([ff, cc[0] + cc[1] * ff + cc[2] * ff * ff]), K=K_est, D=D_est)
            ok = (px[:, 0] > -50) & (px[:, 0] < 370) & (px[:, 1] > -50) & (px[:, 1] < 290)
            if ok.sum() > 2:
                cv2.polylines(big, [np.round(px[ok] * sc * 4).astype(np.int32)], False, C_FIT, s_(2.2), cv2.LINE_AA, shift=2)
        c.a[s_(122):s_(122) + big.shape[0], s_(X0):s_(X0) + big.shape[1]] = big
        c.rect(X0, 122, X0 + CW, 122 + CW * 0.75, EDGE, fill=False, w=1)
        y2 = 122 + CW * 0.75 + 14
        if self.ctrl != "centroid_pid":
            c.text([(X0, y2, "Bird's-eye view + mask + fit", f("sb", 14), TXT, "la")])
            bh, bw = self.d["bev_shape"]
            bev = cv2.cvtColor(cv2.imdecode(fr["bev"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            m = np.unpackbits(fr["mask"])[:bh * bw].reshape(bh, bw).astype(bool)
            bev = (bev * 0.75).astype(np.uint8)
            bev[m] = C_MASK
            side = 210
            k = s_(side) / bw
            bb = cv2.resize(bev, (s_(side), s_(side)), interpolation=cv2.INTER_NEAREST)
            for (x0, r0, x1, r1, good) in fr["wins"]:
                cv2.rectangle(bb, (int(x0 * k), int(r0 * k)), (int(x1 * k), int(r1 * k)), C_WIN if good else (148, 163, 184),
                              s_(1.3), cv2.LINE_AA)
            if fr["coef"] is not None:
                rows = np.arange(bh)
                fwd = 0.66 - rows * 0.004
                cc = fr["coef"]
                cols = (BEV_Y - (cc[0] + cc[1] * fwd + cc[2] * fwd ** 2)) / 0.004
                cv2.polylines(bb, [np.round(np.column_stack([cols * k, rows * k]) * 4).astype(np.int32)], False, C_FIT,
                              s_(2.4), cv2.LINE_AA, shift=2)
            bx = X0 + (CW - side) / 2
            c.a[s_(y2 + 22):s_(y2 + 22) + bb.shape[0], s_(bx):s_(bx) + bb.shape[1]] = bb
            c.rect(bx, y2 + 22, bx + side, y2 + 22 + side, EDGE, fill=False, w=1)
            y3 = y2 + 22 + side + 12
        else:
            c.text([(X0, y2, "Fixed threshold, biggest contour,", f("r", 14), MUTE, "la"),
                    (X0, y2 + 20, "centroid x -> steering", f("r", 14), MUTE, "la")])
            y3 = y2 + 60
        # live numbers
        stat = [("time", f"{fr['t']:5.1f} s"), ("speed", f"{self.s['speed']:.2f} m/s"),
                ("cross-track", f"{fr['cte']*100:+5.1f} cm"), ("perception", f"{fr['lat']:.1f} ms")]
        if self.ctrl != "centroid_pid":
            stat.insert(3, ("line lost", f"{fr['lost']} frames" if fr["lost"] else "no"))
        items = []
        for j, (a, b) in enumerate(stat):
            items += [(X0, y3 + j * 21, a, f("r", 14), MUTE, "la"), (X0 + CW, y3 + j * 21, b, f("sb", 14), TXT, "ra")]
        c.text(items)
        # legend
        leg = [("black tape line (solid)", self.sw_line, "line")]
        if self.has_dash:
            leg += [("dashed stretch: gaps in the tape", self.sw_line, "dash")]
        if self.s["condition"] == "hard":
            leg += [("shadow (lighting disturbance)", self.sw_shadow, "soft"), ("glare spot (ceiling light)", self.sw_glare, "soft")]
        leg += [("floor", self.sw_floor, "box"), ("TurtleBot3 Waffle Pi-style robot", (51, 51, 54), "box"),
               ("line camera field of view", C_FOV, "box"), (f"driven path: {LBL[self.ctrl].split(': ')[-1]}", col, "line"),
               ("detected line fit (RANSAC)", C_FIT, "line")]
        if self.ctrl != "centroid_pid":
            leg += [("detected line pixels", C_MASK, "box"), ("sliding search windows", C_WIN, "rect")]
        else:
            leg = [x for x in leg if not x[0].startswith("detected line fit")]
            leg += [("centroid + steering offset", C_FIT, "line"), ("biggest contour", C_MASK, "line"), ("region of interest", C_WIN, "rect")]
        items = []
        LX, LY = 414, 548
        for j, (t, cl, kind) in enumerate(leg):
            yy = LY + 10 + j * min(30, 278 / max(len(leg), 1))
            if kind == "dash":
                for xs in (LX, LX + 12, LX + 24):
                    cv2.line(c.a, (s_(xs), s_(yy)), (s_(xs + 6), s_(yy)), cl, s_(4), cv2.LINE_AA)
            elif kind == "soft":
                c.rect(LX + 2, yy - 9, LX + 28, yy + 9, self.sw_floor)
                cv2.ellipse(c.a, (s_(LX + 15), s_(yy)), (s_(11), s_(7)), 0, 0, 360, cl, -1, cv2.LINE_AA)
                c.rect(LX + 2, yy - 9, LX + 28, yy + 9, EDGE, fill=False, w=1)
            elif kind == "line":
                cv2.line(c.a, (s_(LX), s_(yy)), (s_(LX + 30), s_(yy)), cl, s_(4), cv2.LINE_AA)
            elif kind == "box":
                c.rect(LX + 2, yy - 8, LX + 28, yy + 8, cl)
            else:
                c.rect(LX + 2, yy - 8, LX + 28, yy + 8, cl, fill=False, w=2)
            items.append((LX + 40, yy, t, f("r", 14), TXT, "lm"))
        c.text(items)
        # chart
        X1, Y1, X2, Y2 = 20, 848, 1060, 1008
        c.rect(X1, Y1, X2, Y2, PANEL)
        c.rect(X1, Y1, X2, Y2, EDGE, fill=False, w=1)
        T = max(self.t_end, 1.0)
        ymap = lambda v: (Y1 + Y2) / 2 + 8 - v / 0.15 * (Y2 - Y1) / 2 * 0.72
        xmap = lambda t: X1 + 50 + (X2 - X1 - 56) * t / T
        for lv in (-0.12, 0.12):
            for xs in np.arange(xmap(0), X2 - 10, 14):
                cv2.line(c.a, (s_(xs), s_(ymap(lv))), (s_(xs + 7), s_(ymap(lv))), C_OFF, s_(1.2), cv2.LINE_AA)
        cv2.line(c.a, (s_(xmap(0)), s_(ymap(0))), (s_(X2 - 10), s_(ymap(0))), EDGE, s_(1), cv2.LINE_AA)
        if i > 1:
            poly(c, np.column_stack([xmap(self.ts[:i + 1]), ymap(self.cte_hist[:i + 1])]), col, 2.0)
        cv2.circle(c.a, (s_(xmap(self.ts[i])), s_(ymap(self.cte_hist[i]))), s_(4.5), col, -1, cv2.LINE_AA)
        c.text([(X1 + 10, Y1 + 6, "Cross-track error vs ground truth (cm)", f("sb", 16), TXT, "la"),
                (X1 + 40, ymap(0.12), "+12", f("sb", 17), C_OFF, "rm"), (X1 + 40, ymap(-0.12), "-12", f("sb", 17), C_OFF, "rm"),
                (X1 + 40, ymap(0), "0", f("sb", 17), MUTE, "rm"), (X2 - 10, Y2 - 6, f"time (s) · run length {T:.0f} s", f("r", 15), MUTE, "rd"),
                (X1 + 52, Y2 - 6, "red dashes = off-track limit", f("r", 15), C_OFF, "ld")])
        # progress bar
        c.rect(20, 1016, 1060, 1023, (226, 232, 240), radius=3)
        c.rect(20, 1016, 20 + 1040 * (i + 1) / len(self.fr), 1023, col, radius=3)
        c.text([(20, 1040, HONEST, f("r", 13), MUTE, "lm"), (1060, 1040, NAME, f("sb", 14), TXT, "rm")])
        if frozen and self.freeze_note:
            k = min(1.0, kf / 8)
            bx0, by0, bx1, by1 = CH[0] + 40, CH[3] - 118, CH[2] - 40, CH[3] - 22
            base = c.a.copy()
            c.rect(bx0, by0, bx1, by1, C_AMBER, radius=16)
            c.rect(bx0 + 3, by0 + 3, bx1 - 3, by1 - 3, C_DARK, radius=13)
            c.rect(bx0 + 22, by0 + 24, bx0 + 28, by1 - 24, C_AMBER, radius=3)
            c.text([(bx0 + 46, by0 + 22, self.freeze_note[0], f("b", 26), C_AMBER, "la"),
                    (bx0 + 46, by0 + 62, self.freeze_note[1], f("r", 18), (226, 232, 240), "la")])
            c.a = (c.a.astype(np.float32) * k + base.astype(np.float32) * (1 - k)).astype(np.uint8)
        return c.out()


# ------------------------------------------------------------------ cards
def card_title(k, n):
    c = Canvas()
    c.rect(0, 0, W, 10, C_CTRL["stanley_ff"])
    c.text([(90, 300, "Camera-based line following", f("b", 54), TXT, "la"),
            (90, 372, "for a small diff-drive robot", f("m", 34), MUTE, "la"),
            (90, 470, "Lens model + calibration  ·  bird's-eye view  ·  RANSAC line fit", f("r", 22), TXT, "la"),
            (90, 506, "PID vs pure pursuit vs Stanley with curvature feed-forward", f("r", 22), TXT, "la"),
            (90, 542, "216 laps: 3 tracks · glare, shadows, light changes · 3 speeds · 3 seeds", f("r", 22), TXT, "la"),
            (90, 700, NAME, f("b", 30), TXT, "la"),
            (90, 940, HONEST, f("r", 16), MUTE, "la")])
    for j, cl in enumerate(C_CTRL.values()):
        c.rect(90 + j * 70, 610, 146 + j * 70, 618, cl, radius=4)
    return fade(c.out(), min(1, k / 12, (n - k) / 10))


def card_end(k, n):
    c = Canvas()
    c.rect(0, 0, W, 10, C_CTRL["stanley_ff"])
    c.text([(90, 330, NAME, f("b", 50), TXT, "la"),
            (90, 410, "Camera-based line following for a small diff-drive robot", f("m", 28), MUTE, "la"),
            (90, 452, "OpenCV · bird's-eye RANSAC fit · PID vs pure pursuit vs Stanley + FF", f("r", 22), TXT, "la")])
    for j, cl in enumerate(C_CTRL.values()):
        c.rect(90 + j * 70, 520, 146 + j * 70, 528, cl, radius=4)
    c.rect(90, 580, 990, 660, PANEL, radius=14)
    c.rect(90, 580, 990, 660, EDGE, fill=False, w=1.5, radius=14)
    c.text([(120, 620, "Code:", f("sb", 26), MUTE, "lm"), (210, 620, "github.com/Shannu2555/opencv-line-follower", f("b", 28), TXT, "lm"),
            (90, 940, HONEST, f("r", 16), MUTE, "la")])
    img = fade(c.out(), min(1, k / 10))
    nb = 21  # last 0.7 s fades to black
    if k >= n - nb:
        a = (n - 1 - k) / nb
        img = (img.astype(np.float32) * max(a, 0)).astype(np.uint8)
    return img


def card_image(title, sub, path, k, n, note=None):
    c = Canvas()
    c.text([(50, 40, title, f("b", 34), TXT, "la"), (50, 92, sub, f("r", 19), MUTE, "la")])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    maxw, maxh = 980, 820 if note else 880
    sc = min(maxw / img.shape[1], maxh / img.shape[0])
    w, h = img.shape[1] * sc, img.shape[0] * sc
    x0, y0 = (W - w) / 2, 140 + (maxh - h) / 2
    c.paste(img, x0, y0, w, h)
    if note:
        c.text([(50, 990, note, f("r", 18), TXT, "la")])
    c.text([(50, 1048, HONEST, f("r", 13), MUTE, "lm"), (1030, 1048, NAME, f("sb", 14), TXT, "rm")])
    return fade(c.out(), min(1, k / 8, (n - k) / 8))


def card_table(rows, k, n, title, sub, header, colw, notes):
    c = Canvas()
    c.text([(50, 40, title, f("b", 34), TXT, "la"), (50, 92, sub, f("r", 19), MUTE, "la")])
    y = 170
    x = 50
    items = []
    xs = np.cumsum([0] + colw) + x
    c.rect(40, y - 14, 1040, y + 40, (241, 245, 249), radius=8)
    for j, hd in enumerate(header):
        for li, part in enumerate(hd.split("\n")):
            items.append((xs[j] + (0 if j == 0 else colw[j] - 10), y + li * 19, part, f("sb", 16), TXT, "la" if j == 0 else "ra"))
    y += 66
    for r in rows:
        clr = r[-1]
        c.rect(46, y - 4, 54, y + 30, clr, radius=3)
        for j, v in enumerate(r[:-1]):
            items.append((xs[j] + (16 if j == 0 else colw[j] - 10), y + 13, v, f("m" if j else "sb", 19), TXT, "lm" if j == 0 else "rm"))
        y += 58
    for li, nline in enumerate(notes):
        items.append((50, y + 20 + li * 30, nline, f("r", 18), MUTE, "la"))
    items += [(50, 1048, HONEST, f("r", 13), MUTE, "lm"), (1030, 1048, NAME, f("sb", 14), TXT, "rm")]
    c.text(items)
    return fade(c.out(), min(1, k / 8, (n - k) / 8))


# ------------------------------------------------------------------ assemble
def scenes():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "results", "summary.csv"))
    out = []
    out.append(("title", int(4.5 * FPS), lambda k, n: card_title(k, n)))
    out.append(("pipe", int(8 * FPS), lambda k, n: card_image(
        "What the robot's camera pipeline does", "One real frame from the hard-lighting run: glare, shadows, lens distortion",
        os.path.join(ROOT, "plots", "pipeline_stages.png"), k, n,
        "raw  →  undistort (calibrated)  →  bird's-eye (IPM)  →  shadow-proof mask  →  sliding windows + RANSAC fit")))
    out.append(("calib", int(7 * FPS), lambda k, n: card_image(
        "Step 1: calibrate the lens", f"OpenCV checkerboard calibration, {calib['views_used']}/{calib['views_total']} views detected",
        os.path.join(ROOT, "plots", "calibration.png"), k, n,
        f"Reprojection RMS {calib['reproj_rms_px']:.2f} px · undistortion vs true lens: {calib['undistort_err_px_mean']:.2f} px mean")))
    runs = [
        ("technical_hard_stanley_ff_v0.19_s0", "Technical track, hard lighting", "Stanley + curvature feed-forward · 0.19 m/s",
         [(1, 150), (8, None)], None),
        ("technical_hard_pid_v0.26_s0", "Same track at 0.26 m/s: PID", "Look-ahead PID on the fitted line · no feed-forward",
         [(1, None)], None),
        ("technical_hard_stanley_ff_v0.26_s0", "0.26 m/s: Stanley + feed-forward", "Same perception, different steering law",
         [(1, 150), (8, None)], None),
        ("figure_eight_hard_centroid_pid_v0.19_s0", "Figure eight, hard lighting: the classic baseline",
         "Fixed threshold + biggest contour centroid + PID · 0.19 m/s", [(1, None)], None),
        ("figure_eight_hard_stanley_ff_v0.19_s0", "Figure eight: full pipeline", "Crossing at 90° in the middle · Stanley + FF · 0.19 m/s",
         [(1, 120), (8, None)], None),
    ]
    for name, t, sub, plan, _ in runs:
        sc = RunScene(name, t, sub, plan)
        s = sc.s
        if s["status"] != "completed":
            sc.freeze_note = (f"Lost the line after {s['distance']:.1f} m",
                              f"{100*s['frac_done']:.0f}% of the lap · cross-track error hit {abs(sc.cte_hist[-1])*100:.0f} cm · lap failed")
        else:
            sc.freeze_note = None
        out.append((name, sc.n_frames(), (lambda sc: (lambda k, n: fade(sc.frame(k), min(1, (k + 1) / 6, (n - k) / 6))))(sc)))
    # results tables
    rows = []
    for cn in C_CTRL:
        d = df[df.controller == cn]
        dc = d[d.completed]
        rows.append([LBL[cn].replace("Bird's-eye fit + ", "Fit + ").replace("Baseline: ", ""),
                     f"{int(d.completed.sum())}/{len(d)}", f"{int(d[d.condition=='hard'].completed.sum())}/{len(d[d.condition=='hard'])}",
                     f"{dc.cte_rms_cm.mean():.2f}" if len(dc) else "-", f"{dc.cte_max_cm.max():.1f}" if len(dc) else "-",
                     f"{dc.head_rms_deg.mean():.1f}" if len(dc) else "-", f"{d.lat_med_ms.median():.2f}", C_CTRL[cn]])
    out.append(("table", 8 * FPS, lambda k, n: card_table(
        rows, k, n, "Results: 216 laps", "3 tracks × clean/hard lighting × 0.12/0.19/0.26 m/s × 3 seeds, per controller",
        ["controller", "laps\ndone", "hard light\ndone", "CTE RMS\n(cm)", "CTE max\n(cm)", "heading\nRMS (°)", "percept.\nms"],
        [330, 105, 125, 110, 110, 110, 100],
        ["Errors are against simulator ground truth, completed laps only (so a controller that", "finishes more of the hard laps is graded on harder laps).",
         "Latency: median per frame, 320×240, one CPU core."])))
    charts = [("completion_rate", "Track completion by controller", "Completion rate by track and speed"),
              ("cte_trace_technical_hard", "Cross-track error, same lap", "Technical track, hard lighting, 0.19 m/s"),
              ("paths_technical_fast", "Driven paths at 0.26 m/s", "Technical track, hard lighting"),
              ("cte_rms_vs_speed", "Tracking error vs speed", "Completed laps, mean ± std"),
              ("perception_accuracy", "How accurate is the line fit?", "Static poses vs the true centreline"),
              ("perception_latency", "Perception time per frame", "Undistort + bird's-eye + mask + windows + RANSAC")]
    for fn, t, sub in charts:
        out.append((fn, 8 * FPS, (lambda fn, t, sub: (lambda k, n: card_image(t, sub, os.path.join(ROOT, "plots", fn + ".png"), k, n)))(fn, t, sub)))
    out.append(("end", 4 * FPS, lambda k, n: card_end(k, n)))
    return out


def main():
    master = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video", "master.mp4")
    only = sys.argv[2] if len(sys.argv) > 2 else None
    sc = scenes()
    if only and only.startswith("--scene="):
        sc = [x for x in sc if x[0] == only.split("=", 1)[1]]
    total = sum(n for _, n, _ in sc)
    print(f"{len(sc)} scenes, {total} frames, {total/FPS:.1f} s", flush=True)
    if only == "--stills":
        os.makedirs(os.path.join(ROOT, "video", "stills"), exist_ok=True)
        for name, n, fn in sc:
            for k in (n // 2,):
                cv2.imwrite(os.path.join(ROOT, "video", "stills", f"{name}.png"), cv2.cvtColor(fn(k, n), cv2.COLOR_RGB2BGR))
        return
    enc = os.environ.get("ENC", "x264")
    vargs = (["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "14", "-b:v", "40M", "-maxrate", "60M",
              "-profile:v", "high"] if enc == "nvenc" else
             ["-c:v", "libx264", "-preset", "slow", "-crf", "12", "-profile:v", "high", "-tune", "animation"])
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"] + vargs + ["-pix_fmt", "yuv420p", "-r", str(FPS), master]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    done = 0
    for name, n, fn in sc:
        for k in range(n):
            p.stdin.write(fn(k, n).tobytes())
            done += 1
        print(f"{name}: {done}/{total}", flush=True)
    p.stdin.close()
    p.wait()
    print("master written", master)


if __name__ == "__main__":
    main()
