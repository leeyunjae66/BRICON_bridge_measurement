"""
정사영 그림 여러 장을 하나의 평면 좌표계로 정합한다 (2단계). 1단계(정사영)의 결과 파일만 읽는다.

    python main.py                                         파일 선택 창에서 정사영 그림을 2장 이상 고른다 (첫 번째가 기준)
    python main.py --images a\\orthorectified.png b\\orthorectified.png
    python main.py --images ... --mode marker              두 사진에 '같은 자리의 같은 마커' 가 찍혔을 때 (Mode A)
    python main.py --images ... --detector ORB             SIFT 대신 ORB
    python main.py --images ... --ransac-threshold 10      맞는 짝으로 보는 어긋남 한도 (mm)
    python main.py --images ... --stop-after features      특징점까지만 (단계별 확인)
    python main.py --images ... --check 점.json            사용자가 준 같은 실제 점으로 오차 평가
    python main.py --images ... --pick                     두 그림에서 같은 점을 직접 클릭해 오차 평가

메타데이터(homography.json)는 그림과 같은 폴더에서 찾는다. 없으면 --meta 로 주거나, 창에서 고른다.
저장 (output\\<이름>):
    feature_matches.png  inlier_matches.png  registration_overlay.png  registration_overlay_zoom.png
    registered_mosaic.png  registration.json  log.txt  debug\\ (가림 그림, 특징점 그림)
그림이 3장 이상이면 두 장씩의 그림은 pair_<고정>__<움직임> 폴더에 들어간다.
끝나는 코드: 0 = 모두 정합, 1 = 입력 오류 또는 일부 실패, 2 = 예상하지 못한 오류.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import feature_matcher as fm
import image_loader as il
import metadata as md
import registration as reg
import rigid_estimator as rg
import visualization as viz
from config import Config


def parse_args(argv=None) -> argparse.Namespace:
    d = Config()
    ap = argparse.ArgumentParser(description="정사영 그림 여러 장을 하나의 평면 좌표계로 정합한다.")
    ap.add_argument("--images", nargs="+", help="정사영 그림 2장 이상 (첫 번째가 기준). 없으면 파일 선택 창")
    ap.add_argument("--meta", nargs="+", help="그림마다의 메타데이터 JSON (그림과 같은 순서). 없으면 그림 옆에서 찾는다")
    ap.add_argument("--mm-per-pixel", type=float, help="메타데이터가 없는 그림에 쓸 축척 (직접 책임지고 줄 때만)")
    ap.add_argument("--mode", choices=["feature", "marker"], default="feature", help="feature = 특징점 (기본), marker = 같은 자리의 같은 마커 기준")
    ap.add_argument("--detector", choices=["SIFT", "ORB"], default=d.detector)
    ap.add_argument("--ratio", type=float, default=d.ratio_test, help=f"비율 검사 값 (기본 {d.ratio_test})")
    ap.add_argument("--ransac-threshold", type=float, default=d.ransac_threshold_mm, help=f"맞는 짝으로 보는 어긋남 한도 mm (기본 {d.ransac_threshold_mm:g})")
    ap.add_argument("--no-marker-mask", action="store_true", help="마커 둘레를 가리지 않음")
    ap.add_argument("--stop-after", choices=["features", "match"], help="그 단계까지만 하고 끝냄 (단계별 확인)")
    ap.add_argument("--check", help="검증 점 파일 JSON")
    ap.add_argument("--pick", action="store_true", help="첫 두 그림에서 같은 점을 클릭해 검증 점을 만든다")
    ap.add_argument("--output", default=str(d.output_dir), help="결과 폴더")
    ap.add_argument("--name", help="결과 폴더 안의 이름 (기본: 기준 그림 이름으로 만든다)")
    ap.add_argument("--no-show", action="store_true", help="화면 창을 띄우지 않음")
    return ap.parse_args(argv)


def safe_name(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", text).strip("_") or "image"


def setup_logging(folder: Path) -> logging.Logger:
    """콘솔과 log.txt 에 같은 내용을 남긴다."""
    folder.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("registration")
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    to_file = logging.FileHandler(folder / "log.txt", mode="w", encoding="utf-8")
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(console)
    log.addHandler(to_file)
    log.propagate = False
    return log


def load_items(paths: list[Path], labels: list[str], args, cfg: Config, log: logging.Logger, gui: bool) -> list[reg.ImageItem]:
    """그림과 메타데이터를 읽는다. 메타데이터가 없거나 그림과 맞지 않으면 지어내지 않고 오류로 알린다."""
    if args.meta and len(args.meta) != len(paths):
        raise md.MetadataError(f"--meta 의 개수({len(args.meta)})가 그림 개수({len(paths)})와 다릅니다.")
    items = []
    for k, (path, label) in enumerate(zip(paths, labels)):
        image = il.read_image(path)
        h, w = image.shape[:2]
        meta_path = Path(args.meta[k]) if args.meta else md.find_metadata(path)
        if meta_path is None and gui:
            picked = il.pick_metadata(path.name, path.parent)
            meta_path = Path(picked) if picked else None
        if meta_path is not None:
            meta = md.load_metadata(meta_path, (w, h))
        elif args.mm_per_pixel:
            meta = md.manual_metadata(args.mm_per_pixel, w, h)
        else:
            raise md.MetadataError(f"'{path}' 의 정사영 메타데이터를 찾지 못했습니다. 그림과 같은 폴더의 homography.json 이 필요합니다.\n"
                                   f"  {md.NEEDED}\n  축척과 원점을 추측해서 쓰지는 않습니다. 파일이 없으면 --meta 로 주거나, 축척을 직접 책임질 수 있을 때만 --mm-per-pixel 을 쓰세요.")
        log.info("[입력] %s: %s  (%d × %d 픽셀)", label, path, w, h)
        log.info("       메타데이터: %s", meta.path if meta.path else "없음 (사용자가 준 축척)")
        log.info("       축척 %.4g mm/픽셀, 픽셀 (0,0) 의 평면 좌표 (%.2f, %.2f) mm, 마커 정보 %s",
                 meta.mm_per_pixel, meta.origin_mm[0], meta.origin_mm[1],
                 f"있음 ({meta.marker_dictionary} ID {meta.marker_id}, {meta.marker_size_mm} mm)" if meta.has_marker else "없음")
        for note in meta.notes:
            log.info("       ※ %s", note)
        items.append(reg.ImageItem(label, path, image, meta))
    return items


def save_debug(item: reg.ImageItem, folder: Path, cfg: Config) -> None:
    il.write_image(folder / f"{safe_name(item.label)}_mask.png", item.detect_mask)
    il.write_image(folder / f"{safe_name(item.label)}_keypoints.jpg", viz.draw_keypoints(item.image, item.features.points_px, cfg.match_view_max_side_px))


def save_pair_images(res: reg.PairResult, fixed: reg.ImageItem, moving: reg.ImageItem, folder: Path, cfg: Config, log: logging.Logger):
    """두 장 정합의 그림 네 가지를 저장한다 → 화면에 띄울 겹친 그림 (실패했으면 None)."""
    if res.pts_fixed_px is not None and len(res.pts_fixed_px):
        il.write_image(folder / "feature_matches.png", viz.draw_matches(
            fixed.image, moving.image, res.pts_fixed_px, res.pts_moving_px, fixed.label, moving.label, cfg, (0, 140, 255), "matches after ratio test"))
        inl = res.inliers
        il.write_image(folder / "inlier_matches.png", viz.draw_matches(
            fixed.image, moving.image, res.pts_fixed_px[inl], res.pts_moving_px[inl], fixed.label, moving.label, cfg, (0, 170, 0), "RANSAC inliers"))
    if not res.success:
        return None
    fixed_from_pixel_f = fixed.meta.plane_from_pixel
    fixed_from_pixel_m = res.t_fixed_from_moving @ moving.meta.plane_from_pixel
    feet = [viz.footprint_mm(fixed.valid, fixed_from_pixel_f), viz.footprint_mm(moving.valid, fixed_from_pixel_m)]
    canvas = viz.plan_canvas(feet, max(fixed.meta.mm_per_pixel, moving.meta.mm_per_pixel), cfg)
    fw, fm_ = viz.warp_to_canvas(fixed.image, fixed.valid, fixed_from_pixel_f, canvas)
    mw, mm_ = viz.warp_to_canvas(moving.image, moving.valid, fixed_from_pixel_m, canvas)
    overlay = viz.blend_overlay(fw, fm_, mw, mm_, cfg.overlay_alpha)
    il.write_image(folder / "registration_overlay.png", overlay)
    box = viz.overlap_box(fm_, mm_)
    if box is None:
        log.info("[그림] 두 그림이 겹치는 부분이 없어 확대 그림은 만들지 않았습니다.")
    else:
        x, y, w, h = box
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        zoom = viz.plan_canvas([rg.apply(canvas.mm_from_pixel, corners)], min(fixed.meta.mm_per_pixel, moving.meta.mm_per_pixel), cfg)
        zf, zfm = viz.warp_to_canvas(fixed.image, fixed.valid, fixed_from_pixel_f, zoom)
        zm, zmm = viz.warp_to_canvas(moving.image, moving.valid, fixed_from_pixel_m, zoom)
        il.write_image(folder / "registration_overlay_zoom.png", viz.color_overlay(zf, zfm, zm, zmm))
    return overlay


def build_mosaic(items: list[reg.ImageItem], poses: list, cfg: Config):
    """정합된 그림들을 기준 그림의 평면에 놓은 한 장 → (그림, 그림 정보)."""
    placed = [(it, poses[k] @ it.meta.plane_from_pixel) for k, it in enumerate(items) if poses[k] is not None]
    feet = [viz.footprint_mm(it.valid, m) for it, m in placed]
    canvas = viz.plan_canvas(feet, max(it.meta.mm_per_pixel for it, _ in placed), cfg)
    layers = [viz.warp_to_canvas(it.image, it.valid, m, canvas) for it, m in placed]
    return viz.mosaic(layers), canvas


def pick_points(a: reg.ImageItem, b: reg.ImageItem, cfg: Config) -> dict:
    """두 그림에서 같은 실제 점을 번갈아 클릭해 고른다 (왼쪽 그림 → 오른쪽 그림 순서).

    왼쪽 클릭 = 점 찍기, 오른쪽 클릭 = 마지막 것 취소, Enter 또는 Esc = 끝.
    화면에서는 그림이 줄어 보이므로 클릭 한 픽셀이 실제로는 여러 mm 다. 정밀한 평가는 --check 파일에 좌표를 직접 적는 쪽이 낫다.
    """
    half = cfg.display_max_width // 2 - 8
    sa = min(1.0, half / a.image.shape[1], cfg.display_max_height / a.image.shape[0])
    sb = min(1.0, half / b.image.shape[1], cfg.display_max_height / b.image.shape[0])
    va = cv2.resize(a.image, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    vb = cv2.resize(b.image, None, fx=sb, fy=sb, interpolation=cv2.INTER_AREA)
    off = va.shape[1] + 16
    base = np.full((max(va.shape[0], vb.shape[0]), off + vb.shape[1], 3), 255, np.uint8)
    base[:va.shape[0], :va.shape[1]] = va
    base[:vb.shape[0], off:off + vb.shape[1]] = vb
    state = {"pending": None, "pairs": []}
    title = "pick same points: LEFT image then RIGHT image / right-click = undo / Enter = done"

    def redraw():
        img = base.copy()
        for k, (pa, pb) in enumerate(state["pairs"], 1):
            for (x, y), dx, s in ((pa, 0, sa), (pb, off, sb)):
                c = (int(x * s) + dx, int(y * s))
                cv2.drawMarker(img, c, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(img, str(k), (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if state["pending"] is not None:
            x, y = state["pending"]
            cv2.drawMarker(img, (int(x * sa), int(y * sa)), (0, 160, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.imshow(title, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["pending"] is None and x < va.shape[1] and y < va.shape[0]:
                state["pending"] = (x / sa, y / sa)
            elif state["pending"] is not None and off <= x < off + vb.shape[1] and y < vb.shape[0]:
                state["pairs"].append((state["pending"], ((x - off) / sb, y / sb)))
                state["pending"] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["pending"] is not None:
                state["pending"] = None
            elif state["pairs"]:
                state["pairs"].pop()
        redraw()

    redraw()
    cv2.setMouseCallback(title, on_mouse)
    while True:
        key = cv2.waitKey(50)
        if key in (13, 27):
            break
        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()
    return {"click_precision_mm": [a.meta.mm_per_pixel / sa, b.meta.mm_per_pixel / sb],
            "points": [{"name": f"P{k}", "pixels": {a.label: list(pa), b.label: list(pb)}} for k, (pa, pb) in enumerate(state["pairs"], 1)]}


def describe_pair(res: reg.PairResult, log: logging.Logger) -> None:
    """두 장 정합의 숫자를 요약해서 적는다."""
    s = res.stats
    log.info("")
    log.info("[결과] %s (고정) ← %s (움직임): %s", res.fixed, res.moving, "성공" if res.success else "실패")
    if "keypoints_fixed" in s:
        log.info("  특징점: %s %d개, %s %d개", res.fixed, s["keypoints_fixed"], res.moving, s["keypoints_moving"])
    if "initial_matches" in s:
        log.info("  처음 짝 %d → 비율 검사 통과 %d → 최종 짝 %d", s["initial_matches"], s["ratio_matches"], s["final_matches"])
    if "inliers" in s:
        log.info("  RANSAC 맞는 짝 %d개 (비율 %.0f %%), 퍼진 범위 %.0f mm", s["inliers"], 100 * s["inlier_ratio"], s["spread_mm"])
        log.info("  짝의 어긋남: 평균 %.2f mm, RMSE %.2f mm, 최대 %.2f mm", s["mean_mm"], s["rmse_mm"], s["max_mm"])
    if "similarity_scale" in s:
        log.info("  진단용 축척: %.4f (1 이면 두 정사영의 축척이 같음)", s["similarity_scale"])
    if res.success:
        rot, tx, ty = rg.decompose(res.t_fixed_from_moving)
        log.info("  방식: %s", "마커 기준 (Mode A)" if res.mode == "marker" else "특징점 (Mode B)")
        log.info("  회전 %.3f°, 이동 X %.1f mm, Y %.1f mm", rot, tx, ty)
    else:
        log.info("  실패 이유: %s", res.reason)
    if res.marker_compare and res.mode == "feature":
        c = res.marker_compare
        log.info("  참고: 같은 번호의 마커가 두 그림에 있습니다. 마커를 옮기지 않았다면 위 회전·이동은 0 에 가까워야 합니다 (지금 %.3f°, %.1f, %.1f mm).",
                 c["feature_rotation_deg"], c["feature_tx_mm"], c["feature_ty_mm"])
    for w in res.warnings:
        log.info("  [주의] %s", w)


def run(args: argparse.Namespace) -> int:
    cfg = replace(Config(), detector=args.detector, ratio_test=args.ratio, ransac_threshold_mm=args.ransac_threshold,
                  mask_marker=not args.no_marker_mask, output_dir=Path(args.output))
    gui = args.images is None
    paths = [Path(p) for p in (args.images or il.pick_images(cfg.input_dir))]
    if len(paths) < 2:
        print("정사영 그림을 2장 이상 골라야 합니다." if paths or not gui else "그림을 고르지 않아 끝냅니다.")
        return 1
    labels = il.make_labels(paths)
    out = cfg.output_dir / safe_name(args.name or f"reg_{labels[0]}_{len(paths)}img")
    log = setup_logging(out)
    log.info("OpenCV %s, 결과 폴더: %s", cv2.__version__, out)

    # ---------------------------------------------------------------- 1. 그림과 메타데이터
    try:
        items = load_items(paths, labels, args, cfg, log, gui)
    except (il.ImageError, md.MetadataError) as e:
        log.error("[오류] %s", e)
        return 1
    if args.mode == "marker" and any(not it.meta.has_marker for it in items):
        log.error("[오류] 마커 기준 방식(--mode marker)은 모든 그림에 마커 정보가 있어야 합니다.")
        return 1

    # ---------------------------------------------------------------- 2. 가림 그림과 특징점
    for it in items:
        reg.prepare(it, cfg, log)
        if cfg.save_debug:
            save_debug(it, out / "debug", cfg)
    if args.stop_after == "features":
        log.info("특징점 단계까지만 하고 끝냅니다. debug 폴더의 그림을 확인하세요.")
        return 0
    if args.stop_after == "match":
        for k in range(1, len(items)):
            a, b = items[k - 1], items[k]
            try:
                m = fm.match(a.features, b.features, cfg)
            except fm.FeatureError as e:
                log.error("[오류] %s ← %s: %s", a.label, b.label, e)
                continue
            log.info("[짝짓기] %s ← %s: 후보 %d → 비율 검사 %d → 최종 %d", a.label, b.label, m.n_initial, m.n_ratio, m.count)
            folder = out if len(items) == 2 else out / f"pair_{safe_name(a.label)}__{safe_name(b.label)}"
            il.write_image(folder / "feature_matches.png", viz.draw_matches(
                a.image, b.image, a.features.points_px[m.idx_a], b.features.points_px[m.idx_b], a.label, b.label, cfg, (0, 140, 255), "matches after ratio test"))
        log.info("짝짓기 단계까지만 하고 끝냅니다.")
        return 0

    # ---------------------------------------------------------------- 3. 정합
    poses, partner, pairs = reg.register_sequence(items, cfg, log, args.mode)
    by_label = {it.label: it for it in items}
    shown = None
    for res in pairs:
        folder = out if len(items) == 2 else out / f"pair_{safe_name(res.fixed)}__{safe_name(res.moving)}"
        overlay = save_pair_images(res, by_label[res.fixed], by_label[res.moving], folder, cfg, log)
        shown = overlay if overlay is not None and shown is None else shown
        describe_pair(res, log)

    # ---------------------------------------------------------------- 4. 합친 그림
    registered = [k for k in range(len(items)) if poses[k] is not None]
    mosaic_info = None
    if len(registered) >= 2:
        image, canvas = build_mosaic(items, poses, cfg)
        il.write_image(out / "registered_mosaic.png", image)
        mosaic_info = {"file": "registered_mosaic.png", "width_px": canvas.width, "height_px": canvas.height,
                       "mm_per_pixel": canvas.mm_per_pixel, "origin_mm": list(canvas.origin_mm),
                       "pixel_to_mm": "X_mm = origin_mm[0] + u * mm_per_pixel,  Y_mm = origin_mm[1] + v * mm_per_pixel  (기준 그림의 평면 좌표)",
                       "mosaic_px_from_ref_mm": canvas.pixel_from_mm, "coarsened_for_size": canvas.coarsened,
                       "images": [items[k].label for k in registered]}
        log.info("")
        log.info("[합친 그림] %d장, %d × %d 픽셀, %.4g mm/픽셀%s", len(registered), canvas.width, canvas.height, canvas.mm_per_pixel,
                 " (크기 한도 때문에 축척을 거칠게 함)" if canvas.coarsened else "")
        if len(items) > 2 or shown is None:
            shown = image if shown is None else shown

    # ---------------------------------------------------------------- 5. 검증 점 (정합 계산에 쓰지 않은 독립된 확인)
    checks, check_source = [], None
    spec = None
    if args.pick and not args.no_show and poses[1] is not None:
        spec = pick_points(items[0], items[1], cfg)
        md.save_json(out / "checkpoints.json", spec)
        check_source = "checkpoints.json (화면에서 클릭)"
        log.info("[검증 점] 클릭 한 픽셀이 실제로는 약 %.1f mm / %.1f mm 입니다 (화면에서 줄여 보이기 때문).", *spec["click_precision_mm"])
    elif args.check:
        try:
            with open(args.check, "r", encoding="utf-8") as f:
                spec = json.load(f)
            check_source = Path(args.check).name
        except (OSError, ValueError) as e:
            log.error("[오류] 검증 점 파일을 읽지 못했습니다: %s", e)
    if spec:
        try:
            checks = reg.evaluate_checkpoints(items, poses, spec)
        except (ValueError, TypeError, KeyError) as e:
            log.error("[오류] 검증 점을 계산하지 못했습니다: %s", e)
        for c in checks:
            log.info("[검증 점] %s: %s 와 %s 사이 어긋남 %.1f mm (dx %.1f, dy %.1f)", c["point"], c["image_a"], c["image_b"], c["error_mm"], c["dx_mm"], c["dy_mm"])
        if checks:
            e = np.array([c["error_mm"] for c in checks])
            log.info("[검증 점] %d개 평균 %.1f mm, RMSE %.1f mm, 최대 %.1f mm", len(e), e.mean(), np.sqrt(np.mean(e ** 2)), e.max())

    # ---------------------------------------------------------------- 6. 변환행렬 저장
    images_out = []
    for k, it in enumerate(items):
        entry = {"label": it.label, "file": str(it.path), "metadata_file": str(it.meta.path) if it.meta.path else None,
                 "width_px": int(it.image.shape[1]), "height_px": int(it.image.shape[0]),
                 "mm_per_pixel": it.meta.mm_per_pixel, "origin_mm": list(it.meta.origin_mm),
                 "plane_mm_from_pixel": it.meta.plane_from_pixel, "keypoints": it.features.count,
                 "status": "reference" if k == 0 else ("registered" if poses[k] is not None else "failed"),
                 "registered_to": items[partner[k]].label if partner[k] is not None else None}
        if poses[k] is not None:
            rot, tx, ty = rg.decompose(poses[k])
            entry.update(T_ref_from_image=poses[k], ref_mm_from_pixel=poses[k] @ it.meta.plane_from_pixel,
                         rotation_deg=rot, translation_mm=[tx, ty])
        else:
            entry.update(T_ref_from_image=None, ref_mm_from_pixel=None,
                         failure_reason="; ".join(r.reason for r in pairs if r.moving == it.label and not r.success))
        images_out.append(entry)
    pairs_out = [{"fixed": r.fixed, "moving": r.moving, "mode": r.mode, "success": r.success, "failure_reason": r.reason or None,
                  "T_fixed_from_moving": r.t_fixed_from_moving, "feature_T_fixed_from_moving": r.feature_t, "stats": r.stats,
                  "marker_compare": r.marker_compare, "warnings": r.warnings} for r in pairs]
    md.save_json(out / "registration.json", {
        "module": "02_registration", "format_version": 1, "opencv": cv2.__version__,
        "settings": {"mode": args.mode, "detector": items[0].features.detector, "ratio_test": cfg.ratio_test,
                     "mutual_check": cfg.mutual_check, "ransac_threshold_mm": cfg.ransac_threshold_mm, "marker_masked": cfg.mask_marker},
        "coordinates": {
            "unit": "mm",
            "common_plane": f"기준 그림({items[0].label})의 마커 평면 좌표. 원점 = 그 그림의 마커 좌상 모서리, x = 좌상→우상, y = 좌상→좌하"
                            if not items[0].meta.from_user else f"기준 그림({items[0].label})의 픽셀 (0,0) 이 원점인 평면 (메타데이터 없이 사용자가 준 축척)",
            "matrix_naming": "A_from_B 는 B 좌표를 A 좌표로 바꾼다:  p_A = A_from_B · p_B  (3×3, 점은 [x, y, 1] 세로 벡터)",
            "plane_mm_from_pixel": "정사영 그림의 픽셀 (u,v) → 그 그림의 마커 평면 mm",
            "T_ref_from_image": "그 그림의 마커 평면 mm → 공통 평면 mm (회전 + 이동, 축척 1)",
            "ref_mm_from_pixel": "= T_ref_from_image · plane_mm_from_pixel. 정사영 그림의 픽셀 → 공통 평면 mm. 거리 측정에는 이것을 쓴다",
            "distance": "두 점을 각자의 그림에서 ref_mm_from_pixel 로 공통 평면에 옮긴 뒤, 두 좌표 사이의 직선 거리"},
        "reference_image": items[0].label, "images": images_out, "pairs": pairs_out, "mosaic": mosaic_info,
        "check_points": {"source": check_source, "results": checks,
                         "note": "사용자가 준 같은 실제 점의 어긋남. 정합 계산에 쓰지 않았으므로 독립된 확인이다."},
        "accuracy_note": "stats 의 어긋남(mean/rmse/max)은 정합 계산에 쓴 특징점 짝이 얼마나 잘 맞는지일 뿐, 실제 치수 정확도를 검증한 값이 아니다.",
    })

    failed = [items[k].label for k in range(len(items)) if poses[k] is None]
    log.info("")
    log.info("[저장] %s", out)
    log.info("  ※ 위의 '짝의 어긋남' 은 정합 계산에 쓴 점들이 서로 얼마나 잘 맞는지입니다. 실제 치수 정확도를 검증한 것이 아닙니다.")
    log.info("     치수 정확도는 길이를 아는 물체나 검증 점(--check, --pick)으로 따로 확인해야 합니다.")
    if failed:
        log.info("  [실패] 정합하지 못해 배치하지 않은 그림: %s", ", ".join(failed))
    if not args.no_show and shown is not None:
        log.info("겹친 그림을 띄웁니다. 아무 키나 누르거나 창을 닫으면 끝납니다.")
        il.show_window("registration result", shown, cfg.display_max_width, cfg.display_max_height)
    return 1 if failed else 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
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
    finally:
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
