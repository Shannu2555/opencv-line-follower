"""Checkerboard intrinsic calibration run on images rendered through the (unknown to perception) lens."""
import numpy as np
import cv2
from .camera_model import render_plane


def checker_texture(cols, rows, square):
    def fn(u, v):
        i = np.floor(u / square)
        j = np.floor(v / square)
        inside = (u >= 0) & (u < (cols + 1) * square) & (v >= 0) & (v < (rows + 1) * square)
        val = np.where(((i + j) % 2) == 0, 0.08, 0.92)
        return np.where(inside, val, 0.97)
    return fn


def rot(rx, ry, rz):
    return cv2.Rodrigues(np.array([rx, ry, rz], float))[0]


def make_views(cam, n_views=25, pattern=(9, 6), square=0.025, seed=0, noise=2.0):
    rng = np.random.default_rng(seed)
    cols, rows = pattern
    tex = checker_texture(cols, rows, square)
    W, H = (cols + 1) * square, (rows + 1) * square
    imgs = []
    while len(imgs) < n_views:
        R_b = rot(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3))
        z = rng.uniform(0.22, 0.40)
        centre = np.array([rng.uniform(-0.12, 0.12) * z / 0.3, rng.uniform(-0.08, 0.08) * z / 0.3, z])
        t_b = centre - R_b @ np.array([W / 2, H / 2, 0])
        g = render_plane(cam, R_b, t_b, tex)
        img = np.clip(g * 255 + rng.normal(0, noise, g.shape), 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        imgs.append(img)
    return imgs


def calibrate(imgs, pattern=(9, 6), square=0.025):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    obj, imgp, used = [], [], 0
    for im in imgs:
        ok, corners = cv2.findChessboardCorners(im, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not ok:
            continue
        corners = cv2.cornerSubPix(im, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
        obj.append(objp)
        imgp.append(corners)
        used += 1
    h, w = imgs[0].shape
    rms, K, D, _, _ = cv2.calibrateCamera(obj, imgp, (w, h), None, None,
                                          flags=cv2.CALIB_FIX_K3 * 0)
    return {"rms_px": float(rms), "K": K, "D": D.ravel()[:5], "views_used": used, "views_total": len(imgs)}
