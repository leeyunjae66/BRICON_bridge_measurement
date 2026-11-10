"""
정합 모듈 시험: 정답을 아는 가상 자료로 각 단계를 따로 확인한다.

    .venv\\Scripts\\python -m unittest test_registration -v

    Rigid      : 강체 변환 계산 (틀린 짝이 섞여도 정답을 찾는지)
    Metadata   : 1단계 형식의 homography.json 읽기 (빠진 값·짝이 안 맞는 그림을 오류로 알리는지)
    EndToEnd   : 가상 바닥 그림에서 잘라 만든 두 정사영 그림을 정합해 정답 변환과 비교
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import metadata as md
import registration as reg
import rigid_estimator as rg
from config import Config

LOG = logging.getLogger("test")
LOG.addHandler(logging.NullHandler())


def make_floor(width: int = 1800, height: int = 1300, seed: int = 0) -> np.ndarray:
    """콘크리트 바닥을 흉내 낸 그림 (1 픽셀 = 1 mm): 여러 크기의 얼룩 + 점 + 긁힌 선 + 줄눈."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 150.0)
    for sigma, amp in ((40, 25), (12, 14), (4, 10), (1.2, 8)):
        img += cv2.GaussianBlur(rng.normal(0, 1, (height, width)), (0, 0), sigma) * amp * sigma ** 0.5
    for _ in range(900):
        c = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.circle(img, c, int(rng.integers(1, 5)), float(rng.uniform(60, 230)), -1, cv2.LINE_AA)
    for _ in range(120):
        p = rng.integers(0, [width, height])
        d = rng.normal(0, 60, 2)
        cv2.line(img, (int(p[0]), int(p[1])), (int(p[0] + d[0]), int(p[1] + d[1])), float(rng.uniform(70, 220)), 1, cv2.LINE_AA)
    for x in range(300, width, 600):
        cv2.line(img, (x, 0), (x, height), 95.0, 2)
    for y in range(250, height, 600):
        cv2.line(img, (0, y), (width, y), 95.0, 2)
    return cv2.cvtColor(np.clip(img, 8, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def cut_ortho(floor: np.ndarray, marker_xy: tuple[float, float], angle_deg: float, origin_mm: tuple[float, float],
              size: tuple[int, int], mm_per_pixel: float = 1.0) -> tuple[np.ndarray, md.OrthoMeta]:
    """바닥 그림에서 '마커 좌표계에 맞춘 정사영 그림' 한 장을 잘라 만든다.

    마커 평면 좌표 p 와 바닥 좌표 F 의 관계:  F = R(angle) · p + marker_xy.
    정사영 그림의 픽셀 (u,v) 는 마커 평면의 origin_mm + (u,v) · mm_per_pixel 이다. 바닥 밖은 검은색(0)이 된다.
    """
    a = math.radians(angle_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    m = np.zeros((2, 3))
    m[:, :2] = r * mm_per_pixel                       # 픽셀 → 바닥 좌표 (바닥은 1 픽셀 = 1 mm)
    m[:, 2] = r @ np.array(origin_mm) + np.array(marker_xy)
    img = cv2.warpAffine(floor, m, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(0, 0, 0))
    return img, md.OrthoMeta(mm_per_pixel, origin_mm, size[0], size[1])


def true_transform(marker_fixed, angle_fixed, marker_moving, angle_moving) -> np.ndarray:
    """정답 T_fixed_from_moving: 움직일 그림의 마커 평면 mm → 고정 그림의 마커 평면 mm."""
    f_from_fixed = rg.make_rigid(math.radians(angle_fixed), *marker_fixed)
    f_from_moving = rg.make_rigid(math.radians(angle_moving), *marker_moving)
    return rg.invert(f_from_fixed) @ f_from_moving


class Rigid(unittest.TestCase):
    def test_fit_rigid_exact(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(-500, 500, (30, 2))
        t = rg.make_rigid(math.radians(27.0), 120.0, -45.0)
        got = rg.fit_rigid(p, rg.apply(t, p))
        np.testing.assert_allclose(got, t, atol=1e-9)

    def test_invert_and_decompose(self):
        t = rg.make_rigid(math.radians(-63.0), 10.0, 20.0)
        np.testing.assert_allclose(rg.invert(t) @ t, np.eye(3), atol=1e-12)
        rot, tx, ty = rg.decompose(t)
        self.assertAlmostEqual(rot, -63.0, places=9)
        self.assertAlmostEqual(tx, 10.0)
        self.assertAlmostEqual(ty, 20.0)

    def test_ransac_ignores_wrong_matches(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1500, (300, 2))
        t = rg.make_rigid(math.radians(12.0), -300.0, 80.0)
        q = rg.apply(t, p) + rng.normal(0, 0.8, p.shape)
        wrong = rng.random(300) < 0.5                       # 절반은 틀린 짝
        q[wrong] = rng.uniform(0, 1500, (int(wrong.sum()), 2))
        fit = rg.ransac_rigid(p, q, threshold_mm=4.0, iterations=2000, seed=0)
        self.assertIsNotNone(fit)
        rot, tx, ty = rg.decompose(fit.t)
        self.assertAlmostEqual(rot, 12.0, delta=0.02)
        self.assertAlmostEqual(tx, -300.0, delta=0.5)
        self.assertAlmostEqual(ty, 80.0, delta=0.5)
        self.assertGreater(fit.n_inliers, 0.9 * int((~wrong).sum()))
        self.assertLess(fit.stats()["rmse_mm"], 2.0)

    def test_ransac_rejects_different_scale(self):
        # 축척이 5 % 다르면 강체로는 넓은 범위를 맞출 수 없다 → 맞는 짝이 일부뿐이어야 한다
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1500, (200, 2))
        fit = rg.ransac_rigid(p, 1.05 * p, threshold_mm=3.0, iterations=2000, seed=0)
        self.assertTrue(fit is None or fit.n_inliers < 100)
        scale, _ = rg.similarity_scale(p, 1.05 * p, 3.0)
        self.assertAlmostEqual(scale, 1.05, delta=0.002)


class Metadata(unittest.TestCase):
    def write(self, folder: Path, data: dict) -> Path:
        path = folder / "homography.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def stage1(self) -> dict:
        return {"source_image": {"file": "a.jpg", "lens_undistorted": False},
                "marker": {"dictionary": "DICT_4X4_50", "id": 0, "size_mm": 150.0,
                           "corners_output_px": [[100, 100], [250, 100], [250, 250], [100, 250]]},
                "output": {"file": "orthorectified.png", "width_px": 800, "height_px": 600, "mm_per_pixel": 1.0,
                           "origin_mm": [-100.0, -100.0], "cropped": False, "horizon_cut": False}}

    def test_reads_stage1_format(self):
        with tempfile.TemporaryDirectory() as d:
            meta = md.load_metadata(self.write(Path(d), self.stage1()), (800, 600))
        self.assertTrue(meta.has_marker)
        np.testing.assert_allclose(rg.apply(meta.plane_from_pixel, np.array([[100.0, 100.0]])), [[0.0, 0.0]])   # 마커 좌상 = 원점
        np.testing.assert_allclose(meta.pixel_from_plane @ meta.plane_from_pixel, np.eye(3), atol=1e-12)

    def test_missing_scale_is_an_error(self):
        data = self.stage1()
        del data["output"]["mm_per_pixel"]
        with tempfile.TemporaryDirectory() as d, self.assertRaises(md.MetadataError):
            md.load_metadata(self.write(Path(d), data), (800, 600))

    def test_wrong_image_size_is_an_error(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(md.MetadataError):
            md.load_metadata(self.write(Path(d), self.stage1()), (801, 600))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.floor = make_floor()

    def register(self, moving_angle: float, detector: str = "SIFT", mpp_moving: float = 1.0):
        cfg = Config(detector=detector, save_debug=False)
        marker_a, marker_b = (500.0, 450.0), (1150.0, 700.0)
        img_a, meta_a = cut_ortho(self.floor, marker_a, 0.0, (-450.0, -400.0), (1000, 900))
        size_b = (int(1100 / mpp_moving), int(900 / mpp_moving))
        img_b, meta_b = cut_ortho(self.floor, marker_b, moving_angle, (-600.0, -450.0), size_b, mpp_moving)
        a = reg.ImageItem("a", Path("a.png"), img_a, meta_a)
        b = reg.ImageItem("b", Path("b.png"), img_b, meta_b)
        for it in (a, b):
            reg.prepare(it, cfg, LOG)
        res = reg.register_pair(a, b, cfg, LOG, "feature")
        return res, true_transform(marker_a, 0.0, marker_b, moving_angle)

    def check(self, res, truth, rot_tol=0.05, shift_tol=1.0):
        self.assertTrue(res.success, res.reason)
        rot, tx, ty = rg.decompose(res.t_fixed_from_moving)
        rot0, tx0, ty0 = rg.decompose(truth)
        self.assertAlmostEqual(rot, rot0, delta=rot_tol)
        # 겹치는 곳 한가운데의 점이 정답 자리에서 얼마나 벗어나는지 (mm)
        centre = np.array([[-200.0, -100.0]])
        err = float(np.linalg.norm(rg.apply(res.t_fixed_from_moving, centre) - rg.apply(truth, centre)))
        self.assertLess(err, shift_tol)
        self.assertAlmostEqual(res.stats["similarity_scale"], 1.0, delta=0.005)

    def test_sift_rotated_pair(self):
        res, truth = self.register(35.0)
        self.check(res, truth)

    def test_sift_different_pixel_size(self):
        # 두 정사영의 mm/픽셀이 달라도 (1 과 2) mm 로 맞추므로 정합돼야 한다
        res, truth = self.register(-20.0, mpp_moving=2.0)
        self.check(res, truth, rot_tol=0.1, shift_tol=2.0)

    def test_orb_pair(self):
        res, truth = self.register(10.0, detector="ORB")
        self.check(res, truth, rot_tol=0.1, shift_tol=2.0)

    def test_no_overlap_fails_cleanly(self):
        cfg = Config(save_debug=False)
        other = make_floor(seed=99)
        img_a, meta_a = cut_ortho(self.floor, (500.0, 450.0), 0.0, (-450.0, -400.0), (900, 800))
        img_b, meta_b = cut_ortho(other, (500.0, 450.0), 0.0, (-450.0, -400.0), (900, 800))
        a = reg.ImageItem("a", Path("a.png"), img_a, meta_a)
        b = reg.ImageItem("b", Path("b.png"), img_b, meta_b)
        for it in (a, b):
            reg.prepare(it, cfg, LOG)
        res = reg.register_pair(a, b, cfg, LOG, "feature")
        self.assertFalse(res.success)
        self.assertTrue(res.reason)

    def test_sequence_skips_failed_image(self):
        cfg = Config(save_debug=False)
        m = [(500.0, 450.0), (900.0, 600.0), (1250.0, 750.0)]
        cuts = [cut_ortho(self.floor, m[0], 0.0, (-450.0, -400.0), (1000, 900)),
                cut_ortho(make_floor(seed=7), m[1], 0.0, (-450.0, -400.0), (900, 800)),      # 다른 바닥: 붙으면 안 된다
                cut_ortho(self.floor, m[2], 15.0, (-650.0, -500.0), (1100, 900))]
        items = [reg.ImageItem(n, Path(n + ".png"), img, meta) for n, (img, meta) in zip("abc", cuts)]
        for it in items:
            reg.prepare(it, cfg, LOG)
        poses, partner, _ = reg.register_sequence(items, cfg, LOG, "feature")
        self.assertIsNone(poses[1])
        self.assertIsNotNone(poses[2])
        self.assertEqual(partner[2], 0)             # 앞 그림이 실패했으므로 기준 그림에 직접 붙는다
        truth = true_transform(m[0], 0.0, m[2], 15.0)
        centre = np.array([[-300.0, -200.0]])
        self.assertLess(float(np.linalg.norm(rg.apply(poses[2], centre) - rg.apply(truth, centre))), 1.0)


if __name__ == "__main__":
    unittest.main()
