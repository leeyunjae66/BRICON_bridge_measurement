"""
촬영한 결과에서 z 를 확인한다 (깊이가 잘 잡혔는지 보는 도구).

    python check_z.py                          가장 최근 촬영을 연다
    python check_z.py output\\20261112_153000   그 촬영을 연다
    python check_z.py --roi 500,300,700,450     창 없이 그 영역의 통계만 출력 (x0,y0,x1,y1 픽셀)
    python check_z.py --ref 100,100,400,300 --roi 500,300,600,400    기준 면(ref)에서 roi 가 얼마나 솟았는지

창에서 쓰는 법
    마우스를 올리면     그 픽셀의 z 와 3차원 좌표가 위에 나온다
    왼쪽 단추로 끌기    영역을 잡는다 → 평균 거리, 평평한 정도(평면에서 벗어난 양), 프레임 사이 흔들림
    A                  방금 잡은 영역을 '기준 면' 으로 삼는다 → 이후 잡는 영역이 그 면에서 얼마나 솟았는지 나온다
    왼쪽 단추로 두 번 찍기   두 점 사이의 실제 거리 (3차원)
    D                  깊이 색 지도 켜고 끄기
    C                  지우기          Q 또는 Esc = 끝

확인 방법 예
    평평한 바닥을 찍고 넓게 영역을 잡는다 → '평면에서 벗어난 양' 이 깊이 오차의 크기다.
    높이를 아는 물체(예: 50 mm 블록)를 바닥에 놓고 찍는다 → 바닥을 기준 면(A)으로 삼고 물체 윗면을 잡아 높이를 비교한다.
모든 결과는 그 촬영 폴더의 check_log.txt 에도 남는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import depthmath as dm
import ui
from config import Config

TITLE = "check z (drag = region, A = reference plane, D = depth view, C = clear, Q = quit)"


class Capture:
    """촬영 폴더 하나를 읽어 둔 것."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        for name in ("color.png", "depth_mm.npy", "meta.json"):
            if not (self.folder / name).is_file():
                raise FileNotFoundError(f"{self.folder} 에 {name} 이 없습니다. capture.py 로 찍은 폴더인지 확인하세요.")
        self.color = ui.read_image(self.folder / "color.png")
        self.depth = np.load(self.folder / "depth_mm.npy")
        noise = self.folder / "depth_noise_mm.npy"
        self.noise = np.load(noise) if noise.is_file() else None
        with open(self.folder / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.intr = dm.Intrinsics.from_dict(self.meta["intrinsics"])
        if self.depth.shape != self.color.shape[:2]:
            raise ValueError("컬러 사진과 깊이 지도의 크기가 다릅니다.")
        self.xyz = dm.deproject(self.intr, self.depth)

    def log(self, text: str) -> None:
        print(text)
        with open(self.folder / "check_log.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")


def latest_capture(root: Path) -> Path | None:
    folders = [p for p in root.iterdir() if (p / "depth_mm.npy").is_file()] if root.is_dir() else []
    return max(folders, key=lambda p: p.stat().st_mtime) if folders else None


def parse_box(text: str, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in text.split(","))
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("영역이 너무 작습니다")
    return x0, y0, x1, y1


def describe_region(cap: Capture, box, plane: dm.Plane | None, name: str = "영역") -> tuple[list[str], dm.Plane | None]:
    """영역의 통계를 글로 만든다 → (글 줄들, 그 영역에 맞춘 평면)."""
    st = dm.region_stats(cap.xyz, box, cap.noise)
    if st is None:
        return [f"{name} {box}: 깊이가 잡힌 점이 거의 없습니다."], None
    pts, _ = dm.region_points(cap.xyz, box)
    lines = [f"{name} (픽셀 x {box[0]}~{box[2]}, y {box[1]}~{box[3]}): 깊이가 잡힌 점 {st['valid']:,}개 ({st['valid_percent']:.0f} %)",
             f"  z 평균 {st['z_mean_mm']:,.1f} mm, 중간값 {st['z_median_mm']:,.1f} mm  (범위 {st['z_min_mm']:,.1f} ~ {st['z_max_mm']:,.1f})",
             f"  평면에서 벗어난 양: 보통 {st['plane_rms_mm']:.2f} mm, 최대 {st['plane_max_mm']:.2f} mm   (면이 카메라와 {st['tilt_deg']:.1f}° 비스듬함)"]
    if st.get("temporal_noise_mm") is not None:
        lines.append(f"  프레임 사이 흔들림(중간값): {st['temporal_noise_mm']:.2f} mm")
    if plane is not None:
        hgt = dm.height_above_plane(plane, pts)
        lines.append(f"  기준 면에서의 높이: 평균 {hgt['mean_mm']:+.2f} mm, 중간값 {hgt['median_mm']:+.2f} mm, 흩어짐 {hgt['std_mm']:.2f} mm   (+ = 카메라 쪽으로 솟음)")
    return lines, dm.fit_plane(pts)


def run_window(cap: Capture, cfg: Config) -> None:
    h, w = cap.depth.shape
    scale = ui.fit_scale(w, h, cfg.display_max_width, cfg.display_max_height - 130)
    depth_view = ui.colorize_depth(cap.depth, cfg.depth_view_min_mm, cfg.depth_view_max_mm)
    s = {"pos": None, "down": None, "box": None, "box_plane": None, "ref_box": None, "ref_plane": None,
         "points": [], "show_depth": False, "lines": ["영역을 끌어 잡거나, 두 점을 차례로 찍으세요."], "head_h": 0}

    def to_pixel(x, y):
        return int(np.clip(x / scale, 0, w - 1)), int(np.clip((y - s["head_h"]) / scale, 0, h - 1))

    def on_mouse(event, x, y, flags, param):
        s["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            s["down"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and s["down"] is not None:
            (ax, ay), (bx, by) = to_pixel(*s["down"]), to_pixel(x, y)
            moved = abs(x - s["down"][0]) + abs(y - s["down"][1])
            s["down"] = None
            if moved > 8:                                   # 끌었으면 영역
                try:
                    box = parse_box(f"{ax},{ay},{bx},{by}", w, h)
                except ValueError:
                    return
                s["box"] = box
                s["lines"], s["box_plane"] = describe_region(cap, box, s["ref_plane"])
                for ln in s["lines"]:
                    cap.log(ln)
            else:                                           # 그냥 눌렀으면 점
                z = cap.depth[by, bx]
                if not np.isfinite(z):
                    s["lines"] = [f"픽셀 ({bx}, {by}): 깊이가 없어 고를 수 없습니다."]
                    return
                s["points"] = (s["points"] + [(bx, by)])[-2:]
                p = cap.xyz[by, bx]
                s["lines"] = [f"점 ({bx}, {by}): z = {z:,.1f} mm,  3차원 좌표 ({p[0]:,.1f}, {p[1]:,.1f}, {p[2]:,.1f}) mm"]
                if len(s["points"]) == 2:
                    (u0, v0), (u1, v1) = s["points"]
                    d = float(np.linalg.norm(cap.xyz[v1, u1] - cap.xyz[v0, u0]))
                    s["lines"].append(f"두 점 ({u0}, {v0}) – ({u1}, {v1}) 사이의 실제 거리: {d:,.1f} mm")
                for ln in s["lines"]:
                    cap.log(ln)

    cv2.namedWindow(TITLE)
    cv2.setMouseCallback(TITLE, on_mouse)
    while True:
        base = depth_view if s["show_depth"] else cap.color
        view = base.copy()
        if s["ref_box"]:
            cv2.rectangle(view, s["ref_box"][:2], s["ref_box"][2:], (255, 255, 0), 2)
            cv2.putText(view, "A", (s["ref_box"][0] + 4, s["ref_box"][1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if s["box"]:
            cv2.rectangle(view, s["box"][:2], s["box"][2:], (0, 220, 255), 2)
        for k, (u, v) in enumerate(s["points"]):
            cv2.drawMarker(view, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        if len(s["points"]) == 2:
            cv2.line(view, s["points"][0], s["points"][1], (0, 0, 255), 1, cv2.LINE_AA)
        if scale < 1.0:
            view = cv2.resize(view, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        hover = "마우스를 그림 위에 올리면 그 픽셀의 z 가 나옵니다."
        if s["pos"] is not None and s["head_h"]:
            u, v = to_pixel(*s["pos"])
            z = cap.depth[v, u]
            if np.isfinite(z):
                p = cap.xyz[v, u]
                hover = f"픽셀 ({u}, {v}):  z = {z:,.1f} mm   3차원 좌표 ({p[0]:,.1f}, {p[1]:,.1f}, {p[2]:,.1f}) mm"
                if cap.noise is not None and np.isfinite(cap.noise[v, u]):
                    hover += f"   흔들림 ±{cap.noise[v, u]:.2f} mm"
            else:
                hover = f"픽셀 ({u}, {v}):  깊이 없음"
        guide = "끌기 = 영역   A = 방금 영역을 기준 면으로   두 번 찍기 = 거리   D = 깊이 색 지도   C = 지우기   Q = 끝"
        lines = [guide, hover] + (s["lines"] + [""] * 5)[:5]
        head = ui.banner(view.shape[1], lines, size=16)
        s["head_h"] = head.shape[0]
        if s["down"] is not None and s["pos"] is not None:
            cv2.rectangle(view, (s["down"][0], s["down"][1] - s["head_h"]), (s["pos"][0], s["pos"][1] - s["head_h"]), (0, 220, 255), 1)
        cv2.imshow(TITLE, np.vstack([head, view]))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or ui.window_closed(TITLE):
            break
        if key in (ord("d"), ord("D")):
            s["show_depth"] = not s["show_depth"]
        elif key in (ord("c"), ord("C")):
            s.update(box=None, box_plane=None, ref_box=None, ref_plane=None, points=[], lines=["지웠습니다."])
        elif key in (ord("a"), ord("A")):
            if s["box_plane"] is None:
                s["lines"] = ["먼저 기준으로 삼을 평평한 영역을 끌어서 잡으세요."]
            else:
                s["ref_box"], s["ref_plane"] = s["box"], s["box_plane"]
                s["lines"] = [f"기준 면으로 정했습니다 (x {s['box'][0]}~{s['box'][2]}, y {s['box'][1]}~{s['box'][3]}). 이제 높이를 볼 영역을 끌어 잡으세요."]
                cap.log(s["lines"][0])
    cv2.destroyAllWindows()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    cfg = Config()
    ap = argparse.ArgumentParser(description="촬영한 결과에서 z 를 확인한다.")
    ap.add_argument("folder", nargs="?", help="촬영 폴더 (없으면 가장 최근 것)")
    ap.add_argument("--roi", action="append", default=[], metavar="x0,y0,x1,y1", help="창 없이 이 영역의 통계를 출력 (여러 번 가능)")
    ap.add_argument("--ref", metavar="x0,y0,x1,y1", help="기준 면으로 삼을 영역 (--roi 의 높이를 이 면에서 잰다)")
    args = ap.parse_args(argv)
    folder = Path(args.folder) if args.folder else latest_capture(cfg.output_dir)
    if folder is None:
        print(f"[오류] {cfg.output_dir} 에 촬영 결과가 없습니다. 먼저 python capture.py 로 찍으세요.")
        return 1
    try:
        cap = Capture(folder)
    except (OSError, ValueError, KeyError) as e:
        print(f"[오류] {e}")
        return 1
    cam = cap.meta.get("camera", {})
    h, w = cap.depth.shape
    print(f"촬영: {cap.folder}")
    print(f"카메라: {cam.get('name', '?')}, 사진 {w} × {h} 픽셀, 평균한 프레임 {cap.meta.get('frames_averaged', '?')}장, 깊이가 잡힌 픽셀 {100 * np.isfinite(cap.depth).mean():.1f} %")
    if args.roi or args.ref:
        try:
            plane = None
            if args.ref:
                lines, plane = describe_region(cap, parse_box(args.ref, w, h), None, "기준 면")
                for ln in lines:
                    cap.log(ln)
            for text in args.roi:
                for ln in describe_region(cap, parse_box(text, w, h), plane)[0]:
                    cap.log(ln)
        except ValueError as e:
            print(f"[오류] 영역은 x0,y0,x1,y1 처럼 숫자 네 개로 적습니다 ({e}).")
            return 1
        return 0
    run_window(cap, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
