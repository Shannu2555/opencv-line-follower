"""Pinhole camera with Brown-Conrady lens distortion, mounted on the robot.

Frames: robot X forward, Y left, Z up. Camera optical frame: x right, y down, z along the optical axis.
"""
import numpy as np
import cv2


def mount_rotation(pitch_deg):
    """Rotation camera->robot for a camera looking forward and pitched down."""
    th = np.radians(pitch_deg)
    s, c = np.sin(th), np.cos(th)
    # columns are the camera x, y, z axes written in the robot frame
    return np.array([[0.0, -s, c], [-1.0, 0.0, 0.0], [0.0, -c, -s]])


class CameraModel:
    def __init__(self, width=320, height=240, fx=230.0, fy=230.0, cx=160.0, cy=120.0,
                 dist=(-0.28, 0.08, 0.001, -0.0005, -0.01), mount_height=0.20, pitch_deg=45.0, mount_x=0.09):
        self.w, self.h = int(width), int(height)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.D = np.array(dist, dtype=float)
        self.mount_height = float(mount_height)
        self.mount_x = float(mount_x)
        self.pitch_deg = float(pitch_deg)
        self.R = mount_rotation(pitch_deg)
        self.C = np.array([self.mount_x, 0.0, self.mount_height])
        self._rays = None

    # ---- geometry -------------------------------------------------------
    def pixel_rays(self):
        """Unit-z rays (camera frame) for every distorted pixel centre, shape (h, w, 3)."""
        if self._rays is None:
            u, v = np.meshgrid(np.arange(self.w, dtype=np.float64), np.arange(self.h, dtype=np.float64))
            pts = np.stack([u, v], -1).reshape(-1, 1, 2)
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-10)
            n = cv2.undistortPoints(pts, self.K, self.D, R=None, P=None, criteria=crit).reshape(self.h, self.w, 2)
            self._rays = np.concatenate([n, np.ones((self.h, self.w, 1))], -1)
        return self._rays

    def ground_lut(self, max_range=3.0):
        """Ground point (robot frame, metres) hit by each pixel; NaN above the horizon."""
        d = self.pixel_rays() @ self.R.T
        dz = d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dz < -1e-6, self.mount_height / -dz, np.nan)
        gx = self.mount_x + t * d[..., 0]
        gy = t * d[..., 1]
        bad = ~np.isfinite(t) | (np.hypot(gx, gy) > max_range)
        gx[bad] = np.nan
        gy[bad] = np.nan
        return gx.astype(np.float32), gy.astype(np.float32)

    def project_ground(self, pts_robot, K=None, D=None):
        """Project ground points (N,2) in the robot frame to pixels. D=None means ideal pinhole."""
        K = self.K if K is None else K
        P = np.column_stack([pts_robot, np.zeros(len(pts_robot))])
        pc = (P - self.C) @ self.R  # R^T (P - C) for each row
        rvec = np.zeros(3)
        img, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), rvec, np.zeros(3), K,
                                   np.zeros(5) if D is None else D)
        return img.reshape(-1, 2)


def render_plane(cam, R_b, t_b, texture_fn, supersample=2):
    """Render a textured plane (pose R_b, t_b in the camera frame) through the distorted lens.
    texture_fn(u, v) -> intensity in [0, 1] for plane coords in metres. Used for calibration images."""
    rays = cam.pixel_rays()
    n = R_b[:, 2]
    acc = np.zeros((cam.h, cam.w))
    offs = [(i + 0.5) / supersample - 0.5 for i in range(supersample)]
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    for dx in offs:
        for dy in offs:
            # small sub-pixel jitter in normalised coords (approximation, good enough for AA)
            r = rays.copy()
            r[..., 0] += dx / fx
            r[..., 1] += dy / fy
            denom = r @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (t_b @ n) / denom
            p = r * t[..., None] - t_b
            uv = p @ R_b
            val = texture_fn(uv[..., 0], uv[..., 1])
            val[~np.isfinite(t) | (t <= 0)] = 0.5
            acc += val
    return acc / supersample ** 2
