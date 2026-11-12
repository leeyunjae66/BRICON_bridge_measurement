"""
측정 모듈 시험: 정답을 아는 가상 자료로 좌표 변환·확대 계산·화면 동작을 확인한다.

    .venv\\Scripts\\python -m unittest test_measurement -v

    Coords    : 픽셀 → mm 변환 (정사영 그림, 원본 사진, 합친 그림), 그림 종류 가리기, 잘못된 입력 알리기
    View      : 확대·이동 계산
    Gui       : 창을 실제로 띄워 열기 → 확대 → 점 찍기 → 지우기 (화면을 쓸 수 없는 환경이면 건너뛴다)
"""

from __future__ import annotations

import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import coords
from viewport import Viewport

S = 150.0                                              # 마커 한 변 (mm)
# 원본 사진 픽셀 → 마커 평면 mm (비스듬히 찍은 사진을 흉내 낸 평면 변환)
H_PLANE_FROM_PHOTO = np.array([[0.82, 0.11, -900.0], [-0.09, 0.86, -700.0], [-0.00001, 0.00016, 0.80]])
MPP, ORIGIN = 1.0, (-600.0, -500.0)                    # 정사영 그림의 축척과 원점
PHOTO_SIZE, ORTHO_SIZE = (1600, 1200), (1500, 1300)


def stage1_folder(folder: Path) -> None:
    """1단계가 저장하는 형식의 폴더를 만든다 (그림은 내용 없는 회색)."""
    plane_corners = np.array([[0.0, 0.0], [S, 0.0], [S, S], [0.0, S]])
    h = coords.normalise(H_PLANE_FROM_PHOTO, (1000.0, 800.0))
    corners_photo = coords.apply_h(coords.normalise(np.linalg.inv(h), (S / 2, S / 2)), plane_corners)
    h = coords.normalise(h, corners_photo.mean(axis=0))
    shift = np.array([[1 / MPP, 0, -ORIGIN[0] / MPP], [0, 1 / MPP, -ORIGIN[1] / MPP], [0, 0, 1.0]])
    data = {"source_image": {"file": "a.jpg", "width_px": PHOTO_SIZE[0], "height_px": PHOTO_SIZE[1], "lens_undistorted": False},
            "marker": {"dictionary": "DICT_4X4_50", "id": 0, "size_mm": S, "corners_image_px": corners_photo.tolist(),
                       "corners_output_px": ((plane_corners - ORIGIN) / MPP).tolist()},
            "homography_image_px_to_plane_mm": h.tolist(), "homography_image_px_to_output_px": (shift @ h).tolist(),
            "output": {"file": "orthorectified.png", "width_px": ORTHO_SIZE[0], "height_px": ORTHO_SIZE[1], "mm_per_pixel": MPP,
                       "origin_mm": list(ORIGIN), "cropped": False}}
    (folder / "homography.json").write_text(json.dumps(data), encoding="utf-8")
    Image.new("RGB", ORTHO_SIZE, (120, 120, 120)).save(folder / "orthorectified.png")
    Image.new("RGB", PHOTO_SIZE, (120, 120, 120)).save(folder / "aruco_detected.png")
    Image.new("RGB", (800, 400), (120, 120, 120)).save(folder / "comparison.png")


def stage2_folder(folder: Path, with_mosaic: bool = True) -> None:
    data = {"module": "02_registration", "reference_image": "photo_1",
            "mosaic": {"file": "registered_mosaic.png", "width_px": 900, "height_px": 700, "mm_per_pixel": 2.0,
                       "origin_mm": [-400.0, -300.0], "images": ["photo_1", "photo_2"], "coarsened_for_size": True} if with_mosaic else None}
    (folder / "registration.json").write_text(json.dumps(data), encoding="utf-8")
    Image.new("RGB", (900, 700), (120, 120, 120)).save(folder / "registered_mosaic.png")
    Image.new("RGB", (900, 700), (120, 120, 120)).save(folder / "registration_overlay.png")


def photo_px(plane_mm) -> np.ndarray:
    """마커 평면의 점이 원본 사진의 어느 픽셀에 찍히는지 (정답)."""
    h = coords.normalise(H_PLANE_FROM_PHOTO, (1000.0, 800.0))
    return coords.apply_h(coords.normalise(np.linalg.inv(h), (S / 2, S / 2)), plane_mm)


class Coords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        stage1_folder(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_homography_from_four_points(self):
        src = np.array([[10.0, 20.0], [300.0, 40.0], [280.0, 310.0], [5.0, 290.0]])
        dst = np.array([[0.0, 0.0], [S, 0.0], [S, S], [0.0, S]])
        h = coords.homography_from_points(src, dst)
        np.testing.assert_allclose(coords.apply_h(coords.normalise(h, src.mean(axis=0)), src), dst, atol=1e-9)

    def test_ortho_pixel_to_mm(self):
        m = coords.find_mapper(self.dir / "orthorectified.png", ORTHO_SIZE)
        self.assertEqual(m.kind, "ortho")
        # 마커 좌상 모서리(평면의 원점)는 정사영 그림의 (600, 500) 픽셀
        np.testing.assert_allclose(m.to_mm([[600.0, 500.0]]), [[0.0, 0.0]], atol=1e-9)
        r = m.measure((600.0, 500.0), (900.0, 900.0))
        self.assertAlmostEqual(r["length_mm"], 500.0, places=6)          # 3-4-5 삼각형
        self.assertAlmostEqual(r["dx_mm"], 300.0, places=6)
        self.assertAlmostEqual(r["dy_mm"], 400.0, places=6)

    def test_photo_and_ortho_give_the_same_length(self):
        # 같은 실제 두 점을 원본 사진에서 찍든 정사영 그림에서 찍든 길이가 같아야 한다
        a, b = np.array([-250.0, 120.0]), np.array([480.0, 610.0])
        ortho = coords.find_mapper(self.dir / "orthorectified.png", ORTHO_SIZE)
        photo = coords.find_mapper(self.dir / "aruco_detected.png", PHOTO_SIZE)
        self.assertEqual(photo.kind, "photo")
        p = photo_px([a, b])
        r_photo = photo.measure(p[0], p[1])
        r_ortho = ortho.measure((a - ORIGIN) / MPP, (b - ORIGIN) / MPP)
        true = float(np.linalg.norm(b - a))
        self.assertAlmostEqual(r_photo["length_mm"], true, places=5)
        self.assertAlmostEqual(r_ortho["length_mm"], true, places=5)
        self.assertAlmostEqual(r_photo["sigma_mm"], r_ortho["sigma_mm"], places=5)      # 예상 오차도 같아야 한다

    def test_expected_error_grows_away_from_marker(self):
        m = coords.find_mapper(self.dir / "orthorectified.png", ORTHO_SIZE)
        def sigma(x0):
            a, b = np.array([x0, 60.0]), np.array([x0 + 200.0, 60.0])       # 길이 200 mm 인 같은 선을 마커에서 점점 멀리
            return m.length_sigma((a - ORIGIN) / MPP, (b - ORIGIN) / MPP)
        near, mid, far = sigma(-30.0), sigma(250.0), sigma(600.0)
        self.assertLess(near, mid)
        self.assertLess(mid, far)
        self.assertLess(near, 1.0)

    def test_local_scale_of_photo_varies(self):
        photo = coords.find_mapper(self.dir / "aruco_detected.png", PHOTO_SIZE)
        near, far = photo.local_mm_per_pixel((800.0, 1100.0)), photo.local_mm_per_pixel((800.0, 100.0))
        self.assertNotAlmostEqual(near, far, places=2)

    def test_comparison_image_is_rejected(self):
        with self.assertRaises(coords.CoordError):
            coords.find_mapper(self.dir / "comparison.png", (800, 400))

    def test_wrong_size_is_rejected(self):
        with self.assertRaises(coords.CoordError):
            coords.find_mapper(self.dir / "orthorectified.png", (1501, 1300))

    def test_missing_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            Image.new("RGB", (50, 50)).save(Path(d) / "x.png")
            with self.assertRaises(coords.CoordError):
                coords.find_mapper(Path(d) / "x.png", (50, 50))

    def test_missing_scale_is_rejected(self):
        d = json.loads((self.dir / "homography.json").read_text(encoding="utf-8"))
        del d["output"]["mm_per_pixel"]
        (self.dir / "homography.json").write_text(json.dumps(d), encoding="utf-8")
        with self.assertRaises(coords.CoordError):
            coords.find_mapper(self.dir / "orthorectified.png", ORTHO_SIZE)

    def test_mosaic(self):
        with tempfile.TemporaryDirectory() as d:
            stage2_folder(Path(d))
            m = coords.find_mapper(Path(d) / "registered_mosaic.png", (900, 700))
            self.assertEqual(m.kind, "mosaic")
            r = m.measure((0.0, 0.0), (300.0, 400.0))
            self.assertAlmostEqual(r["length_mm"], 1000.0, places=6)       # 500 픽셀 × 2 mm
            self.assertEqual(r["a_mm"], (-400.0, -300.0))
            self.assertIsNone(r["sigma_mm"])                               # 합친 그림은 예상 오차를 계산할 정보가 없다
            with self.assertRaises(coords.CoordError):
                coords.find_mapper(Path(d) / "registration_overlay.png", (900, 700))

    def test_failed_registration_has_no_mosaic(self):
        with tempfile.TemporaryDirectory() as d:
            stage2_folder(Path(d), with_mosaic=False)
            with self.assertRaises(coords.CoordError):
                coords.find_mapper(Path(d) / "registered_mosaic.png", (900, 700))


class View(unittest.TestCase):
    def test_screen_and_image_are_inverse(self):
        v = Viewport(2.5, 37.0, -12.0)
        u, w = v.to_image(*v.to_screen(123.4, 56.7))
        self.assertAlmostEqual(u, 123.4)
        self.assertAlmostEqual(w, 56.7)

    def test_zoom_keeps_point_under_cursor(self):
        v = Viewport(1.0, 10.0, 20.0)
        before = v.to_image(400.0, 300.0)
        v.zoom_at(400.0, 300.0, 1.25 ** 5, 0.1, 64.0)
        after = v.to_image(400.0, 300.0)
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])
        self.assertAlmostEqual(v.scale, 1.25 ** 5)

    def test_zoom_is_limited(self):
        v = Viewport(1.0)
        v.zoom_at(0, 0, 1000.0, 0.1, 64.0)
        self.assertEqual(v.scale, 64.0)
        v.zoom_at(0, 0, 1e-6, 0.1, 64.0)
        self.assertEqual(v.scale, 0.1)

    def test_fit_shows_whole_image(self):
        v = Viewport()
        v.fit(4000, 1000, 800, 600, margin=10)
        x0, y0 = v.to_screen(-0.5, -0.5)
        x1, y1 = v.to_screen(3999.5, 999.5)
        self.assertGreaterEqual(x0, 9.999)
        self.assertLessEqual(x1, 790.001)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(y1, 600)

    def test_visible_box(self):
        v = Viewport(4.0, -100.0, -40.0)                   # 4배 확대, 그림이 왼쪽 위로 밀려 있음
        src, dest, size = v.visible_box(1000, 800, 400, 300)
        self.assertEqual(src, (25, 10, 125, 85))
        self.assertEqual(dest, (0, 0))
        self.assertEqual(size, (400, 300))
        self.assertIsNone(Viewport(1.0, 5000.0, 0.0).visible_box(1000, 800, 400, 300))


class Gui(unittest.TestCase):
    def setUp(self):
        try:
            from viewer import MeasureApp
            self.app = MeasureApp()
        except tk.TclError as e:
            self.skipTest(f"화면을 쓸 수 없는 환경: {e}")
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        stage1_folder(self.dir)
        self.app.geometry("1200x800+40+40")
        self.app.update()

    def tearDown(self):
        self.app.destroy()
        self.tmp.cleanup()

    def click(self, u: float, v: float) -> None:
        """그림의 (u, v) 픽셀 자리를 왼쪽 단추로 누른다."""
        sx, sy = self.app.view.to_screen(u, v)
        self.app.canvas.event_generate("<ButtonPress-1>", x=round(sx), y=round(sy))
        self.app.update()

    def test_open_zoom_click_delete(self):
        app = self.app
        self.assertTrue(app.open_image(str(self.dir / "orthorectified.png"), ask_on_error=False))
        app.update()
        self.assertEqual(app.mapper.kind, "ortho")
        fit = app.view.scale
        self.assertLess(fit, 1.0)                                          # 처음에는 그림 전체가 보이게 줄어 있다
        sx, sy = app.view.to_screen(600.0, 500.0)
        for _ in range(8):                                                  # 휠을 8칸 올려 확대
            app.canvas.event_generate("<MouseWheel>", x=round(sx), y=round(sy), delta=120)
        app.update()
        self.assertAlmostEqual(app.view.scale, fit * 1.25 ** 8, places=6)
        # 확대한 상태에서 두 점을 찍는다. 화면 픽셀로 반올림되므로 그만큼의 오차는 허용한다.
        self.click(600.0, 500.0)
        self.assertIsNotNone(app.pending)
        self.click(660.0, 580.0)
        self.assertIsNone(app.pending)
        self.assertEqual(len(app.measurements), 1)
        self.assertAlmostEqual(app.measurements[0].result["length_mm"], 100.0, delta=2.0 / app.view.scale)
        self.assertEqual(len(app.tree.get_children()), 1)
        # 하나 더 재고, 고른 것을 지운다
        app.add_measurement((600.0, 500.0), (900.0, 900.0))
        self.assertEqual([m.no for m in app.measurements], [1, 2])
        app.tree.selection_set("1")
        app.delete_selected()
        self.assertEqual([m.no for m in app.measurements], [2])
        app.clear_all()
        self.assertEqual(len(app.measurements), 0)

    def test_escape_cancels_start_point(self):
        app = self.app
        app.open_image(str(self.dir / "orthorectified.png"), ask_on_error=False)
        app.update()
        self.click(700.0, 600.0)
        self.assertIsNotNone(app.pending)
        app.cancel_pending()
        self.assertIsNone(app.pending)
        self.assertEqual(len(app.measurements), 0)

    def test_unsupported_image_is_not_opened(self):
        self.assertFalse(self.app.open_image(str(self.dir / "comparison.png"), ask_on_error=False))
        self.assertIsNone(self.app.mapper)

    def test_photo_measures_same_as_ortho(self):
        app = self.app
        a, b = np.array([-200.0, 100.0]), np.array([400.0, 500.0])
        app.open_image(str(self.dir / "aruco_detected.png"), ask_on_error=False)
        p = photo_px([a, b])
        m = app.add_measurement(p[0], p[1])
        self.assertAlmostEqual(m.result["length_mm"], float(np.linalg.norm(b - a)), places=4)


if __name__ == "__main__":
    unittest.main()
