import numpy as np
from line_follower.controllers import CONTROLLERS
from line_follower.perception import LineEstimate


def est(offset=0.0, heading=0.0, curv=0.0):
    e = LineEstimate()
    e.ok = True
    c1 = np.tan(heading)
    e.coef = np.array([offset, c1, curv * (1 + c1 * c1) ** 1.5 / 2])
    e.offset, e.heading, e.curvature = offset, heading, curv
    return e


def test_all_steer_towards_the_line():
    for name in ("pid", "pure_pursuit", "stanley_ff"):
        c = CONTROLLERS[name](0.4)
        assert c(est(offset=0.05))[1] > 0, name
        c.reset()
        assert c(est(offset=-0.05))[1] < 0, name


def test_stanley_feed_forward_matches_curvature_on_the_line():
    c = CONTROLLERS["stanley_ff"](0.4)
    v, w = c(est(curv=2.0))
    assert np.isclose(w, 0.4 * 2.0)


def test_controllers_keep_last_command_when_line_missing():
    c = CONTROLLERS["pure_pursuit"](0.4)
    _, w = c(est(offset=0.04))
    lost = LineEstimate()
    assert c(lost)[1] == w
