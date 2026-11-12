"""
화면의 정사영·결합 단추 시험.

    .venv\\Scripts\\python -m unittest test_workflow -v

    Flow     : 앞 단계 프로그램을 가짜로 바꿔 놓고 화면의 흐름만 확인 (사진 추가 → 정사영 → 결합, 실패 처리)
    RealRun  : 앞 단계 프로그램을 실제로 실행 (그 단계의 가상환경이나 예제 사진이 없으면 건너뛴다)
"""

from __future__ import annotations

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import coords
import runner
from config import Config
from test_measurement import stage1_folder, stage2_folder

SAMPLES = [runner.STAGE1_DIR / "input" / n for n in ("KakaoTalk_20261110_173043289.jpg", "KakaoTalk_20261110_173043289_01.jpg")]
STAGE1_READY = (runner.STAGE1_DIR / ".venv" / "Scripts" / "python.exe").is_file()
STAGE2_READY = (runner.STAGE2_DIR / ".venv" / "Scripts" / "python.exe").is_file()


class Flow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        try:
            from viewer import MeasureApp
            self.app = MeasureApp(Config(settings_file=self.dir / "settings.json"))
        except tk.TclError as e:
            self.tmp.cleanup()
            self.skipTest(f"화면을 쓸 수 없는 환경: {e}")
        self.app.quiet = True
        self.app.geometry("1300x800+40+40")
        self.photos = []
        for name in ("a.jpg", "b.jpg"):
            Image.new("RGB", (400, 300), (140, 140, 140)).save(self.dir / name)
            self.photos.append(str(self.dir / name))
        self.app.update()

    def tearDown(self):
        self.app.destroy()
        self.tmp.cleanup()

    def fake_ortho(self, photo, marker_size_mm, *a, **k):
        """1단계 대신: 그 사진의 결과 폴더를 1단계 형식으로 만들어 준다."""
        out = self.dir / "out1" / Path(photo).stem
        out.mkdir(parents=True, exist_ok=True)
        stage1_folder(out)
        self.calls.append(("ortho", Path(photo).name, marker_size_mm))
        return runner.RunResult(True, 0, "가짜 실행 기록", "마커 검출: 성공", out, out / "orthorectified.png")

    def fake_register(self, images, mode="marker", *a, **k):
        out = self.dir / "out2"
        out.mkdir(parents=True, exist_ok=True)
        stage2_folder(out)
        self.calls.append(("register", [Path(p).parent.name for p in images], mode))
        return runner.RunResult(True, 0, "가짜 실행 기록", "[결과] 성공", out, out / "registered_mosaic.png")

    def statuses(self):
        return [it.status for it in self.app.photos]

    def test_add_orthorectify_register(self):
        app, self.calls = self.app, []
        app.add_photos(self.photos)
        app.add_photos(self.photos[:1])                              # 같은 사진을 또 넣어도 한 번만 들어간다
        app.update()
        self.assertEqual(self.statuses(), ["원본", "원본"])
        self.assertIsNotNone(app.image)                              # 넣은 사진이 가운데에 보인다
        self.assertIsNone(app.mapper)                                # 정사영 전이라 치수는 잴 수 없다
        self.assertIsNone(app.add_measurement((10, 10), (50, 50)))

        app.photo_tree.selection_set(())                             # 고른 것이 없으면 아직 안 한 사진 전부
        app.marker_var.set("149.5")
        with mock.patch.object(runner, "run_orthorectification", self.fake_ortho):
            app.orthorectify(sync=True)
        self.assertEqual(self.calls, [("ortho", "a.jpg", 149.5), ("ortho", "b.jpg", 149.5)])
        self.assertEqual(self.statuses(), ["정사영 완료", "정사영 완료"])
        self.assertEqual(app.mapper.kind, "ortho")                   # 펴진 그림이 떠 있고 잴 수 있다
        self.assertIsNotNone(app.add_measurement((600.0, 500.0), (900.0, 900.0)))

        app.mode_var.set(runner.MODES["feature"])
        with mock.patch.object(runner, "run_registration", self.fake_register):
            app.register(sync=True)
        self.assertEqual(self.calls[-1], ("register", ["a", "b"], "feature"))
        self.assertEqual(app.mapper.kind, "mosaic")                  # 합친 그림이 떠 있다
        self.assertEqual([it.kind for it in app.photos], ["photo", "photo", "mosaic"])
        with mock.patch.object(runner, "run_registration", self.fake_register):
            app.register(sync=True)                                  # 다시 결합해도 결합 결과는 하나만 남는다
        self.assertEqual([it.kind for it in app.photos], ["photo", "photo", "mosaic"])

    def test_settings_are_remembered(self):
        app, self.calls = self.app, []
        app.add_photos(self.photos)
        app.marker_var.set("200")
        app.mode_var.set(runner.MODES["feature"])
        with mock.patch.object(runner, "run_orthorectification", self.fake_ortho):
            app.orthorectify(sync=True)
        from viewer import MeasureApp
        again = MeasureApp(Config(settings_file=self.dir / "settings.json"))
        try:
            self.assertEqual(again.marker_size(), 200.0)
            self.assertEqual(again.mode(), "feature")
        finally:
            again.destroy()

    def test_failed_orthorectification_is_reported(self):
        app = self.app
        app.add_photos(self.photos[:1])
        fail = runner.RunResult(False, 1, "기록", "ID 0 마커(DICT_4X4_50)를 찾지 못했습니다.", self.dir, None)
        with mock.patch.object(runner, "run_orthorectification", return_value=fail):
            app.orthorectify(sync=True)
        self.assertEqual(self.statuses(), ["실패"])
        self.assertIn("찾지 못했습니다", app.last_message)
        self.assertIsNone(app.mapper)

    def test_bad_marker_size_does_not_run(self):
        app = self.app
        app.add_photos(self.photos[:1])
        for bad in ("", "abc", "0", "-5"):
            app.marker_var.set(bad)
            with mock.patch.object(runner, "run_orthorectification") as run:
                app.orthorectify(sync=True)
                run.assert_not_called()
            self.assertIn("마커", app.last_message)

    def test_register_needs_two(self):
        app, self.calls = self.app, []
        app.add_photos(self.photos)
        app.photo_tree.selection_set(app.photos[0].iid)
        with mock.patch.object(runner, "run_orthorectification", self.fake_ortho):
            app.orthorectify(sync=True)                              # 한 장만 정사영
        with mock.patch.object(runner, "run_registration") as run:
            app.register(sync=True)
            run.assert_not_called()
        self.assertIn("2장 이상", app.last_message)


class RealRun(unittest.TestCase):
    def test_missing_stage_folder(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(runner.RunnerError):
            runner.run_orthorectification(Path(d) / "x.jpg", 150, stage_dir=Path(d) / "no_such_stage")

    def test_registration_needs_two_images(self):
        with self.assertRaises(runner.RunnerError):
            runner.run_registration([Path("only_one.png")], "marker")

    @unittest.skipUnless(STAGE1_READY, "1단계 가상환경이 없음")
    def test_photo_without_marker_fails_with_reason(self):
        with tempfile.TemporaryDirectory() as d:
            photo = Path(d) / "blank.jpg"
            Image.new("RGB", (800, 600), (128, 128, 128)).save(photo)
            res = runner.run_orthorectification(photo, 150, out_dir=Path(d) / "out")
        self.assertFalse(res.ok)
        self.assertIn("찾지 못했습니다", res.summary)

    @unittest.skipUnless(STAGE1_READY and STAGE2_READY and all(p.is_file() for p in SAMPLES), "앞 단계 가상환경이나 예제 사진이 없음")
    def test_real_orthorectification_and_registration(self):
        with tempfile.TemporaryDirectory() as d:
            results = [runner.run_orthorectification(p, 150, out_dir=Path(d) / f"s{k}") for k, p in enumerate(SAMPLES)]
            for res in results:
                self.assertTrue(res.ok, res.summary)
                with Image.open(res.result_image) as im:
                    size = im.size
                m = coords.find_mapper(res.result_image, size)        # 화면이 그대로 읽을 수 있는 결과여야 한다
                self.assertEqual(m.kind, "ortho")
                self.assertEqual(m.marker_size_mm, 150)
            reg = runner.run_registration([r.result_image for r in results], "marker", out_root=Path(d) / "reg")
            self.assertTrue(reg.ok, reg.summary)
            with Image.open(reg.result_image) as im:
                size = im.size
            self.assertEqual(coords.find_mapper(reg.result_image, size).kind, "mosaic")


if __name__ == "__main__":
    unittest.main()
