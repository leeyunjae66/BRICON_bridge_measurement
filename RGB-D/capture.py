"""
RealSense 로 한 장을 찍어, 컬러 사진의 픽셀마다 z (카메라에서의 거리, mm) 를 저장한다.

    python capture.py                  미리보기 창. 스페이스 = 촬영, Q 또는 Esc = 끝
    python capture.py --shot           창 없이 바로 한 장 찍고 끝
    python capture.py --frames 60      촬영 한 번에 평균할 프레임 수 (기본 30)
    python capture.py --csv            픽셀별 z 를 CSV 로도 저장 (큰 파일)
    python capture.py --preset "High Accuracy"     카메라의 깊이 설정 묶음 고르기
    python capture.py --depth-size 1280x720        깊이 영상 크기 직접 정하기

저장 (output\\<날짜_시각>):
    color.png            컬러 사진
    depth_mm.npy         픽셀별 z (mm, 소수 포함). color.png 와 크기가 같고 같은 (u, v) 가 같은 곳이다. 깊이가 없는 곳은 NaN
    depth_noise_mm.npy   그 픽셀의 z 가 프레임마다 흔들린 정도 (표준편차, mm)
    depth_mm.png         z 를 mm 정수로 담은 16비트 그림 (0 = 깊이 없음). 다른 프로그램에서 열기 쉬운 형식
    depth_view.png       깊이를 색으로 칠한 그림 (눈으로 보기용)
    pointcloud.ply       색이 든 3차원 점군 (mm)
    meta.json            카메라 정보, 내부 값, 촬영 조건
z 는 '카메라에서 그 점까지의 직선 거리' 가 아니라 카메라가 바라보는 방향으로 잰 거리다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import depthmath as dm
import ui
from camera import Camera, CameraError
from config import Config

TITLE = "RealSense capture (SPACE = capture, Q = quit)"


def parse_args(argv=None) -> argparse.Namespace:
    d = Config()
    ap = argparse.ArgumentParser(description="RealSense 로 한 장을 찍어 픽셀마다 z (mm) 를 저장한다.")
    ap.add_argument("--shot", action="store_true", help="창 없이 바로 한 장 찍고 끝")
    ap.add_argument("--frames", type=int, default=d.average_frames, help=f"평균할 프레임 수 (기본 {d.average_frames})")
    ap.add_argument("--csv", action="store_true", help="픽셀별 z 를 CSV 로도 저장")
    ap.add_argument("--preset", default=d.visual_preset, help='깊이 설정 묶음 (예: "High Accuracy", "High Density")')
    ap.add_argument("--depth-size", help="깊이 영상 크기 (예: 1280x720). 없으면 카메라에 맞춰 고른다")
    ap.add_argument("--no-emitter", action="store_true", help="적외선 무늬 투사를 끈다")
    ap.add_argument("--output", default=str(d.output_dir), help="저장할 폴더")
    return ap.parse_args(argv)


def check_against_sdk(cam: Camera, frame) -> float:
    """우리가 계산한 3차원 점이 SDK 의 계산과 같은지 몇 픽셀에서 확인한다 → 가장 큰 차이 (mm)."""
    h, w = frame.depth_mm.shape
    worst = 0.0
    for u, v in ((w // 2, h // 2), (20, 20), (w - 21, 20), (20, h - 21), (w - 21, h - 21)):
        ours = dm.deproject_pixel(cam.intrinsics, u, v, 1000.0)
        worst = max(worst, float(np.abs(ours - cam.sdk_point(u, v, 1000.0)).max()))
    return worst


def grab_average(cam: Camera, n: int, cfg: Config):
    """n 프레임을 받아 깊이를 픽셀별로 평균한다 → (마지막 컬러, 평균 z, 프레임 사이 흔들림, 유효 프레임 수)."""
    depths, color = [], None
    while len(depths) < n:
        f = cam.read()
        if f is None:
            continue
        depths.append(f.depth_mm)
        color = f.color
    mean, std, count = dm.average_depth(np.stack(depths), cfg.min_valid_ratio)
    return color, mean, std, count


def save_capture(folder: Path, color, depth, noise, count, cam: Camera, cfg: Config, n_frames: int, write_csv: bool) -> dict:
    """촬영 결과를 파일로 저장하고 요약을 돌려준다."""
    folder.mkdir(parents=True, exist_ok=True)
    ui.save_image(folder / "color.png", color)
    np.save(folder / "depth_mm.npy", depth)
    np.save(folder / "depth_noise_mm.npy", noise)
    ui.save_image(folder / "depth_mm.png", np.where(np.isfinite(depth), np.clip(np.round(depth), 0, 65535), 0).astype(np.uint16))
    ui.save_image(folder / "depth_view.png", ui.colorize_depth(depth, cfg.depth_view_min_mm, cfg.depth_view_max_mm))
    xyz = dm.deproject(cam.intrinsics, depth)
    n_points = dm.save_ply(folder / "pointcloud.ply", xyz, color)
    ok = np.isfinite(depth)
    h, w = depth.shape
    centre = depth[h // 2 - 5:h // 2 + 6, w // 2 - 5:w // 2 + 6]
    summary = {"valid_percent": float(100.0 * ok.mean()),
               "z_min_mm": float(np.nanmin(depth)) if ok.any() else None, "z_max_mm": float(np.nanmax(depth)) if ok.any() else None,
               "z_centre_mm": float(np.nanmedian(centre)) if np.isfinite(centre).any() else None,
               "noise_median_mm": float(np.nanmedian(noise)) if np.isfinite(noise).any() else None, "points": n_points}
    if write_csv:
        v, u = np.nonzero(ok)
        table = np.column_stack([u, v, depth[ok], xyz[ok][:, 0], xyz[ok][:, 1]])
        np.savetxt(folder / "z.csv", table, fmt=["%d", "%d", "%.2f", "%.2f", "%.2f"], delimiter=",", header="u,v,z_mm,X_mm,Y_mm", comments="")
    meta = {"captured_at": datetime.now().isoformat(timespec="seconds"), "camera": cam.info, "intrinsics": cam.intrinsics.to_dict(),
            "frames_averaged": n_frames, "min_valid_ratio": cfg.min_valid_ratio, "summary": summary,
            "coordinates": {"unit": "mm", "pixel": "(u, v) = color.png 의 (가로, 세로) 픽셀. 깊이는 컬러에 맞춰 정렬돼 있다",
                            "z": "카메라가 바라보는 방향으로 잰 거리 (카메라 면에서 수직으로 떨어진 거리). 직선 거리가 아니다",
                            "camera_frame": "원점 = 컬러 카메라 중심, X = 사진 오른쪽, Y = 사진 아래쪽, Z = 바라보는 방향",
                            "pixel_to_point": "x = (u - cx) / fx, y = (v - cy) / fy (왜곡 보정 뒤) → (X, Y, Z) = (x·z, y·z, z)"},
            "files": {"color.png": "컬러 사진", "depth_mm.npy": "픽셀별 z (mm, float32, 없는 곳 NaN)",
                      "depth_noise_mm.npy": "픽셀별 z 의 프레임 사이 표준편차 (mm)", "depth_mm.png": "z 를 mm 정수로 담은 16비트 그림 (0 = 없음)",
                      "depth_view.png": "깊이 색 지도", "pointcloud.ply": "색이 든 점군 (mm)", "z.csv": "u, v, z, X, Y (--csv 를 줬을 때)"}}
    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return summary


def report(folder: Path, s: dict) -> None:
    print(f"\n[촬영] 저장: {folder}")
    print(f"  깊이가 잡힌 픽셀: {s['valid_percent']:.1f} %")
    if s["z_centre_mm"] is not None:
        print(f"  화면 가운데의 z: {s['z_centre_mm']:.1f} mm   (전체 범위 {s['z_min_mm']:.0f} ~ {s['z_max_mm']:.0f} mm)")
    if s["noise_median_mm"] is not None:
        print(f"  프레임 사이 흔들림(중간값): {s['noise_median_mm']:.2f} mm")
    print("  확인하려면:  python check_z.py")


def run(args: argparse.Namespace) -> int:
    cfg = replace(Config(), average_frames=max(1, args.frames), visual_preset=args.preset, emitter=not args.no_emitter, output_dir=Path(args.output))
    if args.depth_size:
        try:
            w, h = (int(v) for v in args.depth_size.lower().split("x"))
            cfg = replace(cfg, depth_size=(w, h))
        except ValueError:
            print(f"[오류] --depth-size 는 1280x720 처럼 적습니다 (받은 값: {args.depth_size})")
            return 1
    cam = Camera(cfg)
    cam.start()
    try:
        i = cam.info
        print(f"카메라: {i['name']}  (일련번호 {i['serial']}, 펌웨어 {i['firmware']}, USB {i['usb']})")
        print(f"영상: 컬러 {i['color_size'][0]}×{i['color_size'][1]}, 깊이 {i['depth_size'][0]}×{i['depth_size'][1]}, {i['fps']} fps"
              f"   깊이 단위 {i['depth_unit_mm']:g} mm, 무늬 투사 {'켬' if i['emitter'] else '끔'}" + (f", 설정 {i['visual_preset']}" if i["visual_preset"] else ""))
        if i["fallback_note"]:
            print("  [주의] 원하던 영상 크기를 쓸 수 없어 낮췄습니다. USB 3 포트와 케이블인지 확인하세요.")
        if i["usb"].startswith("2"):
            print("  [주의] USB 2 로 연결돼 있습니다. 깊이 품질과 영상 크기가 제한됩니다.")
        frame = None
        for _ in range(cfg.warmup_frames):                 # 노출이 안정될 때까지 버린다
            frame = cam.read() or frame
        if frame is None:
            raise CameraError("카메라에서 쓸 수 있는 영상을 받지 못했습니다.")
        diff = check_against_sdk(cam, frame)
        print(f"좌표 계산 확인: SDK 와의 차이 {diff:.4f} mm" + ("" if diff < 0.01 else "  [주의] 차이가 큽니다. 3차원 좌표(X, Y)는 믿지 마세요."))

        def shoot():
            folder = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
            color, depth, noise, count = grab_average(cam, cfg.average_frames, cfg)
            summary = save_capture(folder, color, depth, noise, count, cam, cfg, cfg.average_frames, args.csv)
            report(folder, summary)
            return folder

        if args.shot:
            shoot()
            return 0

        mouse = {"pos": None}
        cv2.namedWindow(TITLE)
        cv2.setMouseCallback(TITLE, lambda e, x, y, flags, p: mouse.update(pos=(x, y)))
        last_saved = ""
        while True:
            f = cam.read()
            if f is None:
                continue
            h, w = f.depth_mm.shape
            scale = ui.fit_scale(2 * w, h, cfg.display_max_width, cfg.display_max_height - 90)
            view = np.hstack([f.color, ui.colorize_depth(f.depth_mm, cfg.depth_view_min_mm, cfg.depth_view_max_mm)])
            cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
            if scale < 1.0:
                view = cv2.resize(view, (round(2 * w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            centre = np.nanmedian(f.depth_mm[h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4]) if np.isfinite(f.depth_mm[h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4]).any() else float("nan")
            lines = [f"스페이스 = 촬영 ({cfg.average_frames}프레임 평균, 그동안 카메라를 움직이지 마세요)     Q 또는 Esc = 끝",
                     f"화면 가운데 z = {centre:,.1f} mm" if centre == centre else "화면 가운데: 깊이 없음"]
            head = ui.banner(view.shape[1], lines + [last_saved or "왼쪽 = 컬러, 오른쪽 = 깊이 (빨강 = 가까움, 파랑 = 멂, 검정 = 깊이 없음)"])
            if mouse["pos"] is not None:
                mx, my = mouse["pos"]
                u, v = int((mx / scale) % w), int((my - head.shape[0]) / scale)
                if 0 <= v < h and 0 <= mx < view.shape[1]:
                    z = f.depth_mm[v, u]
                    text = f"마우스 자리 픽셀 ({u}, {v}):  z = {z:,.1f} mm" if z == z else f"마우스 자리 픽셀 ({u}, {v}):  깊이 없음"
                    head = ui.banner(view.shape[1], lines + [text])
            cv2.imshow(TITLE, np.vstack([head, view]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or ui.window_closed(TITLE):
                break
            if key == 32:
                folder = shoot()
                last_saved = f"저장했습니다: {folder.name}   (확인: python check_z.py)"
        cv2.destroyAllWindows()
        return 0
    finally:
        cam.stop()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    try:
        return run(parse_args(argv))
    except CameraError as e:
        print(f"\n[오류] {e}")
        return 1
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
