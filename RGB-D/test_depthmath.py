"""
깊이 계산 시험 (카메라 없이 돈다): 정답을 아는 가상 자료로 계산이 맞는지 확인한다.

    .venv\\Scripts\\python -m unittest test_depthmath -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import depthmath as dm

INTR = dm.Intrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0)


def tilted_floor(noise_mm: float = 0.0, seed: int = 0) -> np.ndarray:
    """카메라에서 약 800 mm 떨어진, 조금 기울어진 평평한 바닥의 깊이 지도."""
    u, v = np.meshgrid(np.arange(INTR.width, dtype=np.float64), np.arange(INTR.height, dtype=np.float64))
    x, y = (u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy
    # 평면: 0.05·X + 0.02·Y + Z = 800  →  z = 800 / (1 + 0.05x + 0.02y)
    z = 800.0 / (1.0 + 0.05 * x + 0.02 * y)
    if noise_mm:
        z = z + np.random.default_rng(seed).normal(0, noise_mm, z.shape)
    return z.astype(np.float32)


class Average(unittest.TestCase):
    def test_mean_noise_and_missing(self):
        rng = np.random.default_rng(1)
        stack = (1000.0 + rng.normal(0, 2.0, (40, 6, 8))).astype(np.float32)
        stack[:30, 0, 0] = np.nan            # 40장 중 10장만 잡힘 → 값을 주지 않는다
        stack[:10, 0, 1] = np.nan            # 30장은 잡힘 → 값을 준다
        mean, std, count = dm.average_depth(stack, 0.5)
        self.assertTrue(np.isnan(mean[0, 0]))
        self.assertEqual(count[0, 1], 30)
        self.assertAlmostEqual(float(mean[3, 3]), 1000.0, delta=1.5)
        self.assertAlmostEqual(float(np.nanmedian(std)), 2.0, delta=0.4)


class Deproject(unittest.TestCase):
    def test_pinhole(self):
        p = dm.deproject_pixel(INTR, 640 + 90, 360 - 45, 1000.0)
        np.testing.assert_allclose(p, [100.0, -50.0, 1000.0], atol=1e-9)
        xyz = dm.deproject(INTR, np.full((720, 1280), 1000.0, np.float32))
        np.testing.assert_allclose(xyz[315, 730], [100.0, -50.0, 1000.0], atol=1e-6)

    def test_missing_depth_gives_nan(self):
        d = np.full((720, 1280), 500.0, np.float32)
        d[10, 10] = np.nan
        self.assertTrue(np.all(np.isnan(dm.deproject(INTR, d)[10, 10])))

    def test_inverse_brown_conrady_formula(self):
        intr = dm.Intrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0, "inverse_brown_conrady", (-0.05, 0.06, 0.001, -0.002, -0.02))
        x, y = (1000 - 640) / 900.0, (600 - 360) / 900.0
        r2 = x * x + y * y
        f = 1 - 0.05 * r2 + 0.06 * r2 * r2 - 0.02 * r2 ** 3
        want = np.array([x * f + 2 * 0.001 * x * y - 0.002 * (r2 + 2 * x * x), y * f + 2 * -0.002 * x * y + 0.001 * (r2 + 2 * y * y), 1.0]) * 700.0
        np.testing.assert_allclose(dm.deproject_pixel(intr, 1000, 600, 700.0), want, atol=1e-9)

    def test_brown_conrady_inverts_distortion(self):
        # 왜곡을 넣어 만든 픽셀을 다시 풀면 원래 방향이 나와야 한다
        k1, k2, p1, p2, k3 = 0.08, -0.03, 0.001, 0.0005, 0.01
        intr = dm.Intrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0, "brown_conrady", (k1, k2, p1, p2, k3))
        x, y = 0.3, -0.2
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        p = dm.deproject_pixel(intr, xd * 900 + 640, yd * 900 + 360, 1000.0)
        np.testing.assert_allclose(p, [300.0, -200.0, 1000.0], atol=0.01)


class PlaneAndRegion(unittest.TestCase):
    def test_flat_floor_has_zero_deviation(self):
        xyz = dm.deproject(INTR, tilted_floor())
        st = dm.region_stats(xyz, (200, 150, 1000, 600))
        self.assertLess(st["plane_rms_mm"], 0.01)
        self.assertAlmostEqual(st["valid_percent"], 100.0)
        self.assertGreater(st["tilt_deg"], 2.0)             # 기울어진 면이라는 것도 알아낸다

    def test_noise_shows_up_as_deviation(self):
        xyz = dm.deproject(INTR, tilted_floor(noise_mm=1.5))
        st = dm.region_stats(xyz, (200, 150, 1000, 600))
        self.assertAlmostEqual(st["plane_rms_mm"], 1.5, delta=0.15)

    def test_step_height(self):
        # 바닥 위에 높이 50 mm 인 블록을 올려놓은 장면: 블록 윗면은 바닥보다 카메라에 50 mm 가깝다
        depth = tilted_floor()
        xyz = dm.deproject(INTR, depth)
        floor = dm.fit_plane(dm.region_points(xyz, (100, 100, 500, 600))[0])
        block = depth.copy()
        u, v = np.meshgrid(np.arange(INTR.width, dtype=np.float64), np.arange(INTR.height, dtype=np.float64))
        x, y = (u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy
        n = floor.normal                                     # 바닥에서 카메라 쪽으로 50 mm 옮긴 평면: n·(P − c) = 50
        top = (50.0 + n @ floor.point) / (n[0] * x + n[1] * y + n[2])
        block[250:450, 800:1100] = top[250:450, 800:1100]
        xyz2 = dm.deproject(INTR, block)
        hgt = dm.height_above_plane(floor, dm.region_points(xyz2, (820, 270, 1080, 430))[0])
        self.assertAlmostEqual(hgt["mean_mm"], 50.0, delta=0.01)
        self.assertLess(hgt["std_mm"], 0.01)

    def test_outliers_are_ignored_in_plane_fit(self):
        xyz = dm.deproject(INTR, tilted_floor(noise_mm=0.5))
        pts, _ = dm.region_points(xyz, (300, 200, 900, 500))
        pts = pts.copy()
        pts[::200, 2] += 80.0                                 # 엉뚱하게 튄 점
        self.assertLess(dm.fit_plane(pts).rms_mm, 0.7)

    def test_empty_region(self):
        xyz = dm.deproject(INTR, np.full((720, 1280), np.nan, np.float32))
        self.assertIsNone(dm.region_stats(xyz, (0, 0, 100, 100)))


class Ply(unittest.TestCase):
    def test_writes_only_valid_points(self):
        depth = np.full((4, 5), 1000.0, np.float32)
        depth[0, 0] = np.nan
        xyz = dm.deproject(dm.Intrinsics(5, 4, 10.0, 10.0, 2.0, 2.0), depth)
        with tempfile.TemporaryDirectory() as d:
            n = dm.save_ply(Path(d) / "a.ply", xyz, np.zeros((4, 5, 3), np.uint8))
            data = (Path(d) / "a.ply").read_bytes()
        self.assertEqual(n, 19)
        self.assertIn(b"element vertex 19", data)
        self.assertTrue(data.endswith(b"\x00" * 3))          # 마지막 점의 색 (검정)


if __name__ == "__main__":
    unittest.main()
