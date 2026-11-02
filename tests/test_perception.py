import numpy as np
import cv2
from line_follower.camera_model import CameraModel
from line_follower.perception import LinePerception, CentroidPerception, fit_ransac_quadratic
from line_follower.sim import World


def make_pipe():
    cam = CameraModel()
    return cam, LinePerception(cam.K, cam.D, cam.R, cam.C, (cam.w, cam.h))


def test_ransac_ignores_outliers():
    rng = np.random.default_rng(0)
    f = np.linspace(0.12, 0.6, 200)
    y = 0.03 + 0.2 * f + 0.5 * f * f + rng.normal(0, 0.002, f.size)
    y[::5] += rng.uniform(0.05, 0.2, y[::5].size)  # 20% junk (a crossing line, a shadow edge)
    c, inl = fit_ransac_quadratic(f, y)
    assert np.allclose(c, [0.03, 0.2, 0.5], atol=0.02)
    assert inl.sum() > 140


def test_offset_sign_and_size_on_straight_line():
    cam, pipe = make_pipe()
    w = World("oval_chicane", "clean", 0, cam)
    tr = w.track
    i = 1603  # a straight stretch of the oval
    for off in (0.03, -0.03):
        th = tr.tangent[i]
        # robot shifted to the right of the line by `off` -> line appears on the left (+)
        x = tr.pts[i, 0] + np.sin(th) * off
        y = tr.pts[i, 1] - np.cos(th) * off
        pipe.reset()
        for _ in range(3):
            est = pipe.process(w.camera_image((x, y, th), 0.0))
        assert est.ok
        assert abs(est.offset - off) < 0.012, (est.offset, off)
        assert abs(est.heading) < np.radians(6)


def test_lost_line_holds_then_gives_up():
    cam, pipe = make_pipe()
    blank = np.full((cam.h, cam.w, 3), 200, np.uint8)
    w = World("oval_chicane", "clean", 0, cam)
    tr = w.track
    pipe.process(w.camera_image((tr.pts[0, 0], tr.pts[0, 1], tr.tangent[0]), 0))
    states = [pipe.process(blank).ok for _ in range(pipe.p["max_hold"] + 2)]
    assert all(states[:pipe.p["max_hold"]]) and not states[-1]


def test_centroid_baseline_sign():
    cam = CameraModel()
    img = np.full((cam.h, cam.w, 3), 210, np.uint8)
    cv2.line(img, (100, 0), (100, cam.h), (20, 20, 20), 8)
    est = CentroidPerception((cam.w, cam.h)).process(img)
    assert est.ok and est.offset > 0  # line left of centre
