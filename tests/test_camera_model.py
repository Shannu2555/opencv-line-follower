import numpy as np
import cv2
from line_follower.camera_model import CameraModel, mount_rotation


def test_rotation_is_proper():
    R = mount_rotation(45)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rays_reproject_to_their_pixels():
    cam = CameraModel()
    rays = cam.pixel_rays().reshape(-1, 3)[::97]
    px, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
    u, v = np.meshgrid(np.arange(cam.w, dtype=float), np.arange(cam.h, dtype=float))
    ref = np.stack([u, v], -1).reshape(-1, 2)[::97]
    assert np.abs(px.reshape(-1, 2) - ref).max() < 1e-3


def test_ground_lut_matches_projection():
    cam = CameraModel()
    gx, gy = cam.ground_lut()
    r, c = 200, 100
    p = cam.project_ground(np.array([[gx[r, c], gy[r, c]]]), D=cam.D)[0]
    assert np.allclose(p, [c, r], atol=1e-2)
    assert np.isnan(gx[0, 0]) or gx[0, 0] > gx[-1, cam.w // 2]  # top of image looks further
