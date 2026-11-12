"""
화면의 목록(탭)과 정사영·결합 단추 시험.

    .venv\\Scripts\\python -m unittest test_workflow -v

    Flow     : 앞 단계 프로그램을 가짜로 바꿔 놓고 화면의 흐름만 확인 (사진 추가 → 정사영 → 결합, 실패 처리)
    Lists    : 목록을 여러 개 두었을 때 작업이 목록마다 따로 되는지, 창을 다시 켜면 되살아나는지
    RealRun  : 앞 단계 프로그램을 실제로 실행 (그 단계의 가상환경이나 예제 사진이 없으면 건너뛴다)
"""

from __future__ import annotations

import shutil
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


class GuiCase(unittest.TestCase):
    """화면을 띄우고, 앞 단계 프로그램을 가짜로 바꿔 쓰는 시험의 바탕."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.calls = []
        try:
            self.app = self.make_app()
        except tk.TclError as e:
            self.tmp.cleanup()
            self.skipTest(f"화면을 쓸 수 없는 환경: {e}")
        self.photos = []
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            Image.new("RGB", (400, 300), (140, 140, 140)).save(self.dir / name)
            self.photos.append(str(self.dir / name))
        self.app.update()

    def tearDown(self):
        self.app.destroy()
        self.tmp.cleanup()

    def make_app(self):
        from viewer import MeasureApp
        app = MeasureApp(Config(session_file=self.dir / "session.json"))
        app.quiet = True
        app.geometry("1300x800+40+40")
        return app

    def fake_ortho(self, photo, marker_size_mm, out_dir=None, **k):
        """1단계 대신: 화면이 정해 준 결과 폴더를 1단계 형식으로 채워 준다."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stage1_folder(out)
        self.calls.append(("ortho", Path(photo).name, marker_size_mm, out.parent.name))
        return runner.RunResult(True, 0, "가짜 실행 기록", "마커 검출: 성공", out, out / "orthorectified.png")

    def fake_register(self, images, mode="marker", name=None, **k):
        out = self.dir / "out2" / name
        out.mkdir(parents=True, exist_ok=True)
        stage2_folder(out)
        self.calls.append(("register", [Path(p).parent.name for p in images], mode, name))
        return runner.RunResult(True, 0, "가짜 실행 기록", "[결과] 성공", out, out / "registered_mosaic.png")

    def run_ortho(self):
        with mock.patch.object(runner, "STAGE1_DIR", self.dir / "stage1"), mock.patch.object(runner, "run_orthorectification", self.fake_ortho):
            self.app.orthorectify(sync=True)

    def run_register(self):
        with mock.patch.object(runner, "run_registration", self.fake_register):
            self.app.register(sync=True)

    def statuses(self, k=None):
        items = self.app.photos if k is None else self.app.lists[k].items
        return [it.status for it in items]

    def go(self, k: int):
        """k 번째 탭으로 간다."""
        self.app.book.select(self.app.lists[k].frame)
        self.app.update()


class Flow(GuiCase):
    def test_add_orthorectify_register(self):
        app = self.app
        app.add_photos(self.photos[:2])
        app.add_photos(self.photos[:1])                              # 같은 사진을 또 넣어도 한 번만 들어간다
        app.update()
        self.assertEqual(self.statuses(), ["원본", "원본"])
        self.assertIsNotNone(app.image)                              # 넣은 사진이 가운데에 보인다
        self.assertIsNone(app.mapper)                                # 정사영 전이라 치수는 잴 수 없다
        self.assertIsNone(app.add_measurement((10, 10), (50, 50)))

        app.photo_tree.selection_set(())                             # 고른 것이 없으면 아직 안 한 사진 전부
        app.marker_var.set("149.5")
        self.run_ortho()
        self.assertEqual([c[:3] for c in self.calls], [("ortho", "a.jpg", 149.5), ("ortho", "b.jpg", 149.5)])
        self.assertEqual(self.statuses(), ["정사영 완료", "정사영 완료"])
        self.assertEqual(app.mapper.kind, "ortho")                   # 펴진 그림이 떠 있고 잴 수 있다
        self.assertIsNotNone(app.add_measurement((600.0, 500.0), (900.0, 900.0)))

        app.mode_var.set(runner.MODES["feature"])
        self.run_register()
        self.assertEqual(self.calls[-1][:3], ("register", ["a", "b"], "feature"))
        self.assertEqual(app.mapper.kind, "mosaic")                  # 합친 그림이 떠 있다
        self.assertEqual([it.kind for it in app.photos], ["photo", "photo", "mosaic"])
        self.run_register()                                          # 다시 결합해도 결합 결과는 하나만 남는다
        self.assertEqual([it.kind for it in app.photos], ["photo", "photo", "mosaic"])

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
        app = self.app
        app.add_photos(self.photos[:2])
        app.photo_tree.selection_set(app.photos[0].iid)
        self.run_ortho()                                             # 한 장만 정사영
        with mock.patch.object(runner, "run_registration") as run:
            app.register(sync=True)
            run.assert_not_called()
        self.assertIn("2장 이상", app.last_message)


class Lists(GuiCase):
    def test_starts_with_one_list(self):
        self.assertEqual([lst.name for lst in self.app.lists], ["목록 1"])
        self.assertEqual(self.app.marker_size(), 150.0)
        self.assertEqual(self.app.mode(), "marker")

    def test_work_stays_inside_the_current_list(self):
        app = self.app
        app.add_photos(self.photos[:2])                              # 목록 1: a, b
        app.new_list()                                               # 목록 2 가 앞으로 나온다
        self.assertEqual(app.current.name, "목록 2")
        app.add_photos(self.photos[2:])                              # 목록 2: c
        self.assertEqual([len(lst.items) for lst in app.lists], [2, 1])

        app.photo_tree.selection_set(())
        self.run_ortho()                                             # 목록 2 에서 정사영 → c 만
        self.assertEqual([c[1] for c in self.calls], ["c.jpg"])
        self.assertEqual(self.calls[-1][3], "목록_2")                 # 결과도 목록 이름의 폴더에
        self.assertEqual(self.statuses(0), ["원본", "원본"])           # 목록 1 은 그대로
        self.assertEqual(self.statuses(1), ["정사영 완료"])

        with mock.patch.object(runner, "run_registration") as run:   # 목록 2 는 한 장뿐이라 결합할 수 없다
            app.register(sync=True)                                  # (목록 1 의 사진을 끌어다 쓰지 않는다)
            run.assert_not_called()
        self.assertIn("목록 2", app.last_message)

        self.go(0)
        app.photo_tree.selection_set(())
        self.run_ortho()                                             # 목록 1 에서 정사영 → a, b 만
        self.assertEqual([c[1] for c in self.calls], ["c.jpg", "a.jpg", "b.jpg"])
        self.assertEqual(self.calls[-1][3], "목록_1")
        self.run_register()                                          # 목록 1 끼리만 결합
        self.assertEqual(self.calls[-1][1], ["a", "b"])
        self.assertIn("목록_1", self.calls[-1][3])
        self.assertEqual([it.kind for it in app.lists[0].items], ["photo", "photo", "mosaic"])
        self.assertEqual([it.kind for it in app.lists[1].items], ["photo"])

    def test_marker_size_and_mode_are_per_list(self):
        app = self.app
        app.marker_var.set("150")
        app.new_list()                                               # 새 목록은 보고 있던 목록의 값을 물려받는다
        self.assertEqual(app.marker_entry.get(), "150")
        app.marker_var.set("200")
        app.mode_var.set(runner.MODES["feature"])
        self.go(0)
        self.assertEqual(app.marker_entry.get(), "150")              # 입력 칸이 그 목록의 값으로 바뀐다
        self.assertEqual(app.mode(), "marker")
        self.go(1)
        self.assertEqual(app.marker_entry.get(), "200")
        self.assertEqual(app.mode(), "feature")
        app.add_photos(self.photos[:1])
        app.photo_tree.selection_set(())
        self.run_ortho()
        self.assertEqual(self.calls[-1][2], 200.0)                   # 그 목록의 마커 크기로 계산한다

    def test_switching_list_shows_its_picture(self):
        app = self.app
        app.add_photos(self.photos[:1])
        app.photo_tree.selection_set(())
        self.run_ortho()
        self.assertEqual(app.mapper.kind, "ortho")
        app.new_list()
        app.add_photos(self.photos[1:2])
        app.update()
        self.assertIsNone(app.mapper)                                # 목록 2 의 사진은 아직 정사영 전
        self.go(0)
        self.assertEqual(app.mapper.kind, "ortho")                   # 목록 1 로 돌아오면 그 목록의 그림이 뜬다

    def test_rename_and_delete(self):
        app = self.app
        app.new_list()
        self.assertFalse(app.rename_list("목록 1"))                   # 이미 있는 이름으로는 못 바꾼다
        self.assertTrue(app.rename_list("3번 슬래브"))
        self.assertEqual([lst.name for lst in app.lists], ["목록 1", "3번 슬래브"])
        self.assertEqual(app.book.tab(app.lists[1].frame, "text"), "3번 슬래브")
        app.delete_list()
        self.assertEqual([lst.name for lst in app.lists], ["목록 1"])
        app.delete_list()                                            # 마지막 목록을 지워도 빈 목록 하나는 남는다
        self.assertEqual(len(app.lists), 1)
        self.assertEqual(app.lists[0].items, [])

    def test_session_is_restored(self):
        app = self.app
        app.add_photos(self.photos[:2])
        app.photo_tree.selection_set(())
        self.run_ortho()
        self.run_register()
        app.new_list("2번 구간")
        app.marker_var.set("200")
        app.mode_var.set(runner.MODES["feature"])
        app.add_photos(self.photos[2:])
        app._on_close()                                              # 창을 닫는다
        self.app = again = self.make_app()                           # 다시 켠다
        again.update()
        self.assertEqual([lst.name for lst in again.lists], ["목록 1", "2번 구간"])
        self.assertEqual(again.current.name, "2번 구간")              # 보고 있던 탭
        self.assertEqual(again.marker_size(), 200.0)
        self.assertEqual(again.mode(), "feature")
        self.assertEqual(self.statuses(0), ["정사영 완료", "정사영 완료", "결합 완료"])
        self.assertEqual(self.statuses(1), ["원본"])
        self.go(0)
        self.assertEqual(again.marker_size(), 150.0)
        self.assertEqual(again.mapper.kind, "mosaic")                # 다시 계산하지 않아도 바로 잴 수 있다

    def test_missing_results_fall_back_to_original(self):
        app = self.app
        app.add_photos(self.photos[:2])
        app.photo_tree.selection_set(())
        self.run_ortho()
        app._on_close()
        shutil.rmtree(self.dir / "stage1")                           # 그사이 결과 폴더가 지워졌다면
        self.app = again = self.make_app()
        again.update()
        self.assertEqual(self.statuses(0), ["원본", "원본"])


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
