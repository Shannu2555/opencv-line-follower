import numpy as np
from line_follower.track import Track, TRACKS
from line_follower.sim import Robot, Progress, WHEEL_MAX, WHEEL_BASE, V_MAX, W_MAX


def test_tracks_closed_and_followable():
    for n in TRACKS:
        t = Track(n)
        assert np.linalg.norm(t.pts[0] - t.pts[-1]) < 0.02
        assert t.min_radius() > 0.15


def test_robot_straight_and_wheel_limit():
    r = Robot((0, 0, 0))
    for _ in range(60):
        r.step(0.5, 0.0, 1 / 30)
    assert abs(r.y) < 0.03 and 0.45 < r.x < V_MAX * 2 + 0.02
    r = Robot((0, 0, 0))
    for _ in range(60):
        v, w = r.step(0.0, 50.0, 1 / 30)
    assert abs(w) <= min(W_MAX, 2 * WHEEL_MAX / WHEEL_BASE) * 1.06


def test_cross_track_sign():
    t = Track("oval_chicane")
    p = Progress(t, 0)
    th = t.tangent[0]
    x = t.pts[0, 0] - np.sin(th) * 0.05
    y = t.pts[0, 1] + np.cos(th) * 0.05
    cte, he, _ = p.update(x, y, th)
    assert np.isclose(cte, 0.05, atol=0.005) and abs(he) < 1e-6
