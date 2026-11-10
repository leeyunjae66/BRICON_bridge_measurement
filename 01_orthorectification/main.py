"""
ArUco 마커 1개로 스마트폰 사진 전체를 정사영 보정한다.

    python main.py                              파일 선택 창에서 사진을 고른다
    python main.py --image input\\사진.jpg       사진을 바로 지정
    python main.py --marker-size 199.5          인쇄한 마커를 재 보니 199.5 mm 였을 때
    python main.py --mm-per-pixel 2             출력 축척을 픽셀당 2 mm 로 (범위가 넓을 때)
    python main.py --no-show                    화면 창을 띄우지 않고 저장만

치수 오차 표: 사진 옆에 같은 이름의 .check.json (예: 사진.jpg → 사진.check.json) 을 두면,
길이를 아는 구간을 재서 comparison.png 아래에 표로 붙인다. 점 위치는 원본 사진 픽셀(회전 적용 뒤)이다.
    {"points": {"A": [185.4, 886.9], "B": [144.1, 1164.4]},
     "lengths": [{"from": "A", "to": "B", "reference_mm": 100}]}

저장 (output 폴더):
    aruco_detected.png   마커 검출 결과
    orthorectified.png   사진 전체 정사영 결과
    comparison.png       보정 전후 비교
    homography.json      변환 행렬, 마커 크기, 축척, 좌표 원점
끝나는 코드: 0 = 성공, 1 = 사진을 못 읽음·마커를 못 찾음 등, 2 = 예상하지 못한 오류.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import aruco_detector as ad
import homography as hg
import image_utils as iu
from config import Config


def parse_args(argv=None) -> argparse.Namespace:
    d = Config()
    ap = argparse.ArgumentParser(description="ArUco 마커 1개로 스마트폰 사진 전체를 정사영 보정한다.")
    ap.add_argument("--image", help="사진 파일 (없으면 파일 선택 창을 띄운다)")
    ap.add_argument("--marker-size", type=float, default=d.marker_size_mm, help=f"마커 한 변의 실제 길이 mm (기본 {d.marker_size_mm:g})")
    ap.add_argument("--mm-per-pixel", type=float, default=d.mm_per_pixel, help=f"출력 축척: 픽셀 하나가 몇 mm 인지 (기본 {d.mm_per_pixel:g})")
    ap.add_argument("--marker-id", type=int, default=d.marker_id, help=f"마커 번호 (기본 {d.marker_id})")
    ap.add_argument("--dictionary", default=d.aruco_dictionary, help=f"마커 사전 (기본 {d.aruco_dictionary})")
    ap.add_argument("--calib", default=d.calibration_file, help="카메라 보정 파일 JSON (있을 때만)")
    ap.add_argument("--check", help="검증 점 파일 JSON (없으면 사진과 같은 이름의 .check.json 을 찾는다)")
    ap.add_argument("--output", default=str(d.output_dir), help="결과를 저장할 폴더")
    ap.add_argument("--no-show", action="store_true", help="화면 창을 띄우지 않음")
    return ap.parse_args(argv)


def fail(text: str, gui: bool) -> int:
    print(f"\n[오류] {text}")
    if gui:
        iu.show_message("정사영 보정", text, error=True)
    return 1


def run(args: argparse.Namespace) -> int:
    cfg = replace(Config(), marker_size_mm=args.marker_size, mm_per_pixel=args.mm_per_pixel, marker_id=args.marker_id,
                  aruco_dictionary=args.dictionary, calibration_file=args.calib, output_dir=Path(args.output))
    gui = not args.no_show and args.image is None      # 창으로 고른 경우에만 오류도 창으로 알린다
    warnings: list[str] = []
    print(f"OpenCV {cv2.__version__}")

    # ---------------------------------------------------------------- 1. 사진 불러오기
    path = args.image
    if path is None:
        iu.enable_heic()
        path = iu.pick_image_file(cfg.input_dir)
        if path is None:
            print("사진을 고르지 않아 끝냅니다.")
            return 1
    try:
        image, info = iu.load_image(path)
    except iu.ImageLoadError as e:
        return fail(str(e), gui)
    print(f"\n[1] 사진: {info['file']}  ({info['format']})")
    print(f"    크기: 가로 {info['width']} × 세로 {info['height']} 픽셀" + ("  (회전 정보 적용함)" if info["exif_rotation_applied"] else ""))

    # ---------------------------------------------------------------- 렌즈 왜곡 (보정값이 있을 때만)
    undistorted = False
    if cfg.calibration_file:
        try:
            fixed, why = iu.undistort_image(image, iu.load_calibration(cfg.calibration_file))
        except (OSError, KeyError, ValueError) as e:
            fixed, why = None, f"보정 파일을 읽지 못했습니다 ({e})"
        if fixed is None:
            warnings.append(f"렌즈 왜곡을 보정하지 않았습니다: {why}.")
        else:
            image, undistorted = fixed, True
            print("    렌즈 왜곡을 보정했습니다.")
    if not undistorted:
        warnings.append("카메라 보정값 없이 처리했습니다. 렌즈 왜곡이 남아 있어 치수 정확도에 한계가 있습니다 (사진 가장자리일수록 큼).")

    # ---------------------------------------------------------------- 2. 마커 검출
    det = ad.detect_marker(image, cfg)
    print(f"\n[2] 마커 검출: {'성공' if det.found else '실패'}")
    if not det.found:
        return fail(det.message, gui)
    warnings += det.warnings
    print(f"    마커 ID: {det.marker_id}   (사진에서 찾은 번호 전체: {det.all_ids})")
    print("    네 모서리 (원본 사진 픽셀):")
    for name, p in zip(ad.CORNER_NAMES, det.corners):
        print(f"      {name}: ({p[0]:9.2f}, {p[1]:9.2f})")
    print(f"    사진에서 마커 한 변: 약 {det.side_px:.0f} 픽셀")
    detected = ad.draw_detection(image, det)

    # ---------------------------------------------------------------- 3. 호모그래피
    try:
        h = hg.compute_homography(det.corners, cfg.marker_size_mm)
        plan = hg.plan_output(h, (info["width"], info["height"]) if not undistorted else image.shape[1::-1], cfg.marker_size_mm, cfg)
    except hg.HomographyError as e:
        return fail(str(e), gui)
    warnings += plan.notes
    print(f"\n[3] 호모그래피 (사진 픽셀 → 마커 평면 mm, 마커 한 변 {cfg.marker_size_mm:g} mm):")
    for row in h:
        print("      [" + "  ".join(f"{v: .6e}" for v in row) + "]")

    # ---------------------------------------------------------------- 4. 사진 전체 정사영
    try:
        ortho = hg.warp_image(image, plan)
    except (cv2.error, MemoryError) as e:
        return fail(f"정사영 변환 중 오류 (출력이 너무 클 수 있습니다. --mm-per-pixel 을 키워 보세요): {e}", gui)
    fr = plan.full_range_mm
    print(f"\n[4] 정사영 결과: 가로 {plan.width} × 세로 {plan.height} 픽셀,  축척 {plan.mm_per_pixel:g} mm/픽셀")
    print(f"    담긴 범위: {plan.width * plan.mm_per_pixel:.0f} × {plan.height * plan.mm_per_pixel:.0f} mm   "
          f"(사진 전체 범위: {fr[2] - fr[0]:.0f} × {fr[3] - fr[1]:.0f} mm)")
    print(f"    출력 픽셀 (0,0) 의 평면 좌표: ({plan.origin_mm[0]:.2f}, {plan.origin_mm[1]:.2f}) mm   [원점 = 마커 좌상 모서리]")

    # ---------------------------------------------------------------- 7. 확인
    marker_mm, _ = hg.apply_homography(h, det.corners)
    sides = hg.side_lengths(marker_mm)
    marker_out, _ = hg.apply_homography(plan.h_image_to_output, det.corners)
    print("\n[5] 보정 후 마커 네 변 (위, 오른쪽, 아래, 왼쪽):")
    print("      " + ",  ".join(f"{s:.3f} mm" for s in sides))
    print("    ※ 이 네 모서리로 변환을 만들었으므로 정확히 맞는 것이 당연합니다. 정확도를 검증한 것이 아닙니다.")
    redetect = None
    again = ad.detect_marker(ortho, cfg)
    if again.found:
        redetect = hg.side_lengths(again.corners) * plan.mm_per_pixel
        print("    보정된 그림에서 마커를 다시 찾아 잰 네 변: " + ",  ".join(f"{s:.2f} mm" for s in redetect))
        print("    ※ 변환·저장 과정이 맞는지 보는 확인입니다. 같은 마커를 쓰므로 이것도 독립된 검증은 아닙니다.")
    # 길이를 아는 구간(검증 점 파일)이 있으면 재서 기준과 비교한다
    checks, check_file, check_note = [], None, ""
    check_path = Path(args.check) if args.check else Path(path).with_suffix(".check.json")
    if check_path.is_file():
        try:
            with open(check_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            checks = hg.measure_lengths(h, spec["points"], spec["lengths"])
            check_file = check_path.name
            check_note = str(spec.get("note", ""))      # 표 아래에 적을 설명 (없으면 기본 문구)
        except (OSError, ValueError, KeyError, TypeError, hg.HomographyError) as e:
            warnings.append(f"검증 점 파일({check_path.name})을 쓰지 못했습니다: {e}")
    elif args.check:
        warnings.append(f"검증 점 파일이 없습니다: {args.check}")
    if checks:
        print(f"\n    검증 길이 ({check_file}):   구간 / 기준 / 측정 / 오차")
        for c in checks:
            print(f"      {c['name']:8s} {c['reference_mm']:9.1f} mm {c['measured_mm']:9.1f} mm  {c['error_mm']:+6.1f} mm ({c['error_percent']:+.2f} %)")

    sens = hg.estimate_extrapolation_error(det.corners, cfg.marker_size_mm, image.shape[1::-1], cfg.corner_noise_px)
    print(f"\n[6] 예상 오차: 모서리를 {cfg.corner_noise_px:g} 픽셀 틀리게 찾았다고 가정했을 때 (렌즈 왜곡·바닥 굴곡 제외)")
    for s in sens:
        if s["in_photo"]:
            print(f"      마커에서 {s['distance_mm'] / 1000:.1f} m: 위치 오차 보통 ±{s['typical_error_mm']:.1f} mm, 나쁠 때 ±{s['p95_error_mm']:.1f} mm")
        else:
            print(f"      마커에서 {s['distance_mm'] / 1000:.1f} m: 사진 밖")

    # ---------------------------------------------------------------- 6. 저장
    comparison = iu.make_comparison(detected, ortho)
    # 검증 점이 있을 때만 비교 그림 아래에 치수 오차 표를 붙인다 (없으면 표를 만들지 않는다)
    if checks:
        headers = ["구간", "기준 (mm)", "측정 (mm)", "오차 (mm)", "오차 (%)"]
        rows = [[c["name"], f"{c['reference_mm']:.1f}", f"{c['measured_mm']:.1f}", f"{c['error_mm']:+.1f}", f"{c['error_percent']:+.2f}"] for c in checks]
        note = check_note or f"마커와 따로 있는 검증 점({check_file})으로 잰 값입니다. 측정 - 기준 = 오차."
        comparison = iu.append_table(comparison, f"치수 오차  (마커 한 변 {cfg.marker_size_mm:g} mm 로 계산)", headers, rows, note)
    out = cfg.output_dir
    data = {
        "source_image": {"file": info["file"], "width_px": int(image.shape[1]), "height_px": int(image.shape[0]),
                         "exif_rotation_applied": info["exif_rotation_applied"], "lens_undistorted": undistorted},
        "marker": {"dictionary": cfg.aruco_dictionary, "id": det.marker_id, "size_mm": cfg.marker_size_mm,
                   "corner_order": "좌상, 우상, 우하, 좌하 (마커 무늬 기준)",
                   "corners_image_px": det.corners.tolist(), "corners_output_px": marker_out.tolist()},
        "plane_coordinates": {"unit": "mm", "origin": "마커 좌상 모서리", "x_axis": "좌상 → 우상", "y_axis": "좌상 → 좌하"},
        "homography_image_px_to_plane_mm": h.tolist(),
        "homography_image_px_to_output_px": plan.h_image_to_output.tolist(),
        "output": {"file": "orthorectified.png", "width_px": plan.width, "height_px": plan.height,
                   "mm_per_pixel": plan.mm_per_pixel, "origin_mm": list(plan.origin_mm),
                   "pixel_to_mm": "X_mm = origin_mm[0] + u * mm_per_pixel,  Y_mm = origin_mm[1] + v * mm_per_pixel",
                   "distance_mm": "두 픽셀 사이 거리(픽셀) * mm_per_pixel",
                   "full_range_mm": list(plan.full_range_mm), "cropped": plan.cropped, "horizon_cut": plan.horizon_cut},
        "check": {"marker_sides_mm_from_homography": sides.tolist(),
                  "marker_sides_mm_redetected_in_output": None if redetect is None else redetect.tolist(),
                  "note": "마커 네 모서리로 변환을 만들었으므로 마커 변이 맞는 것은 독립된 정확도 검증이 아니다."},
        "dimension_check": {"file": check_file, "lengths": checks, "table_note": check_note,
                            "note": "마커와 따로 있는, 길이를 아는 구간을 잰 값. 파일이 없으면 비어 있다."},
        "expected_error": {"assumed_corner_noise_px": cfg.corner_noise_px, "by_distance_from_marker": sens,
                           "note": "모서리 검출 오차만 반영. 렌즈 왜곡과 평면에서 벗어난 높이는 들어 있지 않다."},
        "warnings": warnings,
    }
    try:
        iu.save_image(out / "aruco_detected.png", detected)
        iu.save_image(out / "orthorectified.png", ortho)
        iu.save_image(out / "comparison.png", comparison)
        iu.save_json(out / "homography.json", data)
    except OSError as e:
        return fail(f"결과를 저장하지 못했습니다: {e}", gui)
    print(f"\n[7] 저장: {out}")
    for name in ("aruco_detected.png", "orthorectified.png", "comparison.png", "homography.json"):
        print(f"      {name}")
    if warnings:
        print("\n[주의]")
        for w in warnings:
            print(f"  - {w}")
    print("  - 이 보정은 마커와 같은 평면에 있는 것만 바르게 폅니다. 평면에서 솟거나 꺼진 것은 위치가 틀어집니다.")

    # ---------------------------------------------------------------- 5. 화면
    if not args.no_show:
        print("\n비교 화면을 띄웁니다. 아무 키나 누르거나 창을 닫으면 끝납니다.")
        iu.show_window("ArUco orthorectification  (left: original / right: orthorectified)", comparison,
                       cfg.display_max_width, cfg.display_max_height)
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")       # 콘솔이 표시하지 못하는 글자가 있어도 멈추지 않게
    except (AttributeError, ValueError):
        pass
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        return 1
    except Exception:
        print("\n[오류] 예상하지 못한 문제가 생겼습니다:")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
