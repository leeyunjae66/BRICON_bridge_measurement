"""
정합의 흐름: 그림 준비 → 두 장 정합 → 여러 장 이어 붙이기 → 검증 점 평가.

두 가지 방식
    feature (Mode B, 기본) : 겹치는 바닥면의 특징점으로 회전과 이동을 구한다. 마커를 옮겼거나 마커가 한쪽에만 있어도 된다.
    marker  (Mode A)       : 두 사진에 '같은 자리의 같은 마커' 가 찍혔을 때. 두 정사영은 이미 같은 마커 좌표계라서
                             변환은 단위행렬이다. 번호가 같다고 같은 마커인 것은 아니므로, 사용자가 이 방식을 골라야만 쓴다.
어느 방식이든 가능하면 다른 방식의 결과도 같이 계산해서 비교한다.

행렬 이름: T_fixed_from_moving 은 '움직일 그림의 평면 mm' 를 '고정 그림의 평면 mm' 로 바꾼다.
여러 장일 때:  T_ref_from_i = T_ref_from_k · T_k_from_i   (k = 그림 i 를 붙인 상대 그림)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import feature_matcher as fm
import image_loader as il
import rigid_estimator as rg
from config import Config
from metadata import OrthoMeta


@dataclass
class ImageItem:
    label: str
    path: Path
    image: np.ndarray
    meta: OrthoMeta
    valid: np.ndarray | None = None          # 사진이 있는 곳 (원래 크기)
    features: fm.Features | None = None
    detect_mask: np.ndarray | None = None    # 특징점을 찾은 곳 (줄인 크기, 디버깅용)


@dataclass
class PairResult:
    fixed: str
    moving: str
    mode: str                                # 최종 변환을 정한 방식: "feature" 또는 "marker"
    success: bool = False
    reason: str = ""                         # 실패했을 때의 이유
    t_fixed_from_moving: np.ndarray | None = None
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    feature_ok: bool = False                 # 특징점 방식 자체가 성공했는지 (marker 방식일 때도 비교용으로 계산)
    feature_t: np.ndarray | None = None
    marker_compare: dict | None = None       # 마커 기준(단위행렬)과 특징점 결과의 차이
    # 그림 그리기용: 비율 검사를 통과한 짝의 픽셀 좌표와 맞는 짝 여부
    pts_fixed_px: np.ndarray | None = None
    pts_moving_px: np.ndarray | None = None
    inliers: np.ndarray | None = None


def prepare(item: ImageItem, cfg: Config, log: logging.Logger) -> None:
    """그림 한 장의 가림 그림을 만들고 특징점을 찾는다."""
    item.valid = il.valid_mask(item.image)
    mask = item.valid
    used_marker = False
    if cfg.mask_marker and item.meta.has_marker:
        mask = cv2.bitwise_and(mask, il.marker_mask(item.image.shape, item.meta.marker_corners_px, cfg.marker_mask_scale))
        used_marker = True
    item.features, item.detect_mask = fm.detect(item.image, mask, cfg)
    area = 100.0 * cv2.countNonZero(item.valid) / item.valid.size
    log.info("[특징점] %s: %d개 (%s, 그림을 %.2f배로 줄여 찾음, 사진이 있는 영역 %.0f %%, 마커 가림 %s)",
             item.label, item.features.count, item.features.detector, item.features.scale, area, "함" if used_marker else "안 함")


def _same_marker_info(a: OrthoMeta, b: OrthoMeta) -> str:
    """두 그림의 마커 정보가 같은 종류인지. 다르면 이유를 돌려준다 (같으면 빈 글)."""
    if not (a.has_marker and b.has_marker):
        return "한쪽 그림에 마커 정보가 없습니다"
    if a.marker_id != b.marker_id or a.marker_dictionary != b.marker_dictionary:
        return f"마커 번호·사전이 다릅니다 ({a.marker_dictionary} ID {a.marker_id} / {b.marker_dictionary} ID {b.marker_id})"
    if a.marker_size_mm and b.marker_size_mm and abs(a.marker_size_mm - b.marker_size_mm) > 1e-6:
        return f"마커 크기가 다릅니다 ({a.marker_size_mm} mm / {b.marker_size_mm} mm)"
    return ""


def _feature_registration(fixed: ImageItem, moving: ImageItem, cfg: Config, log: logging.Logger, res: PairResult) -> None:
    """특징점으로 T_fixed_from_moving 을 구해 res 에 채운다. 실패하면 res.reason 에 이유를 적는다."""
    st = res.stats
    st["keypoints_fixed"], st["keypoints_moving"] = fixed.features.count, moving.features.count
    try:
        m = fm.match(fixed.features, moving.features, cfg)
    except fm.FeatureError as e:
        res.reason = str(e)
        return
    st.update(initial_matches=m.n_initial, ratio_matches=m.n_ratio, final_matches=m.count)
    log.info("[짝짓기] %s ← %s: 후보 %d → 비율 검사 %d → 서로 고르기 %d", fixed.label, moving.label, m.n_initial, m.n_ratio, m.count)
    res.pts_fixed_px = fixed.features.points_px[m.idx_a]
    res.pts_moving_px = moving.features.points_px[m.idx_b]
    res.inliers = np.zeros(m.count, dtype=bool)
    if m.count < cfg.min_inliers:
        res.reason = (f"짝이 너무 적습니다 ({m.count}개, 최소 {cfg.min_inliers}개). 두 그림이 겹치지 않거나, "
                      "겹치는 곳에 무늬가 거의 없거나, 한쪽이 많이 흐립니다.")
        return
    # 픽셀 → 각 그림의 마커 평면 mm. 이후 계산은 모두 mm 로 한다.
    q = rg.apply(fixed.meta.plane_from_pixel, res.pts_fixed_px)
    p = rg.apply(moving.meta.plane_from_pixel, res.pts_moving_px)
    fit = rg.ransac_rigid(p, q, cfg.ransac_threshold_mm, cfg.ransac_iterations, cfg.ransac_seed)
    if fit is None:
        res.reason = "두 점 사이 거리가 양쪽 그림에서 같은 짝을 하나도 찾지 못했습니다 (겹치는 곳이 없거나 축척이 서로 다름)."
        return
    res.inliers = fit.inliers
    ratio = fit.n_inliers / float(m.count)
    spread = fit.spread_mm(q)
    rot, tx, ty = rg.decompose(fit.t)
    st.update(inliers=fit.n_inliers, inlier_ratio=ratio, spread_mm=spread, rotation_deg=rot, tx_mm=tx, ty_mm=ty,
              ransac_hypotheses=fit.hypotheses, ransac_threshold_mm=cfg.ransac_threshold_mm, **fit.stats())
    log.info("[강체 추정] 맞는 짝 %d / %d (%.0f %%), 어긋남 평균 %.2f mm · RMSE %.2f mm · 최대 %.2f mm, 퍼진 범위 %.0f mm",
             fit.n_inliers, m.count, 100 * ratio, st["mean_mm"] or 0, st["rmse_mm"] or 0, st["max_mm"] or 0, spread)
    # 진단용 축척: 축척을 풀어 주면 얼마가 나오는지
    sim = rg.similarity_scale(p, q, cfg.ransac_threshold_mm)
    if sim is not None:
        st["similarity_scale"], st["similarity_inliers"] = sim
        log.info("[축척 진단] 축척을 자유롭게 두면 %.4f (맞는 짝 %d개). 1 이면 두 정사영의 축척이 같다는 뜻.", sim[0], sim[1])
        if abs(sim[0] - 1.0) > cfg.scale_warning:
            res.warnings.append(f"진단용 축척이 {sim[0]:.4f} 로 1 에서 {abs(sim[0] - 1) * 100:.1f} % 벗어났습니다. "
                                "두 정사영의 축척이 서로 다릅니다 (마커 크기 설정, 마커에서 먼 곳의 정사영 오차를 확인하세요).")
    if fit.n_inliers < cfg.min_inliers:
        res.reason = f"맞는 짝이 너무 적습니다 ({fit.n_inliers}개, 최소 {cfg.min_inliers}개)."
    elif ratio < cfg.min_inlier_ratio:
        res.reason = f"맞는 짝의 비율이 너무 낮습니다 ({100 * ratio:.0f} %, 최소 {100 * cfg.min_inlier_ratio:.0f} %). 우연히 맞은 것일 수 있습니다."
    elif spread < cfg.min_spread_mm:
        res.reason = f"맞는 짝이 좁은 범위({spread:.0f} mm)에 몰려 있어 회전을 믿을 수 없습니다 (최소 {cfg.min_spread_mm:.0f} mm)."
    else:
        res.feature_ok = True
        res.feature_t = fit.t


def register_pair(fixed: ImageItem, moving: ImageItem, cfg: Config, log: logging.Logger, mode: str = "feature") -> PairResult:
    """그림 두 장을 정합한다. 결과의 t_fixed_from_moving 은 움직일 그림의 평면 mm → 고정 그림의 평면 mm."""
    res = PairResult(fixed.label, moving.label, mode)
    log.info("---- 정합: %s (고정) ← %s (움직임), 방식 = %s", fixed.label, moving.label, "마커 기준" if mode == "marker" else "특징점")
    _feature_registration(fixed, moving, cfg, log, res)
    marker_problem = _same_marker_info(fixed.meta, moving.meta)

    if mode == "marker":
        if marker_problem:
            res.reason = f"마커 기준 방식을 쓸 수 없습니다: {marker_problem}."
            return res
        # 같은 자리의 같은 마커라면 두 그림의 평면 좌표는 이미 같다 → 단위행렬
        res.success, res.t_fixed_from_moving, res.reason = True, np.eye(3), ""
        res.stats.update(rotation_deg=0.0, tx_mm=0.0, ty_mm=0.0)
        if res.feature_ok:
            rot, tx, ty = rg.decompose(res.feature_t)
            q = rg.apply(fixed.meta.plane_from_pixel, res.pts_fixed_px[res.inliers])
            p = rg.apply(moving.meta.plane_from_pixel, res.pts_moving_px[res.inliers])
            d = np.linalg.norm(p - q, axis=1)       # 마커 기준으로 그대로 놓았을 때 같은 점이 어긋난 거리
            res.marker_compare = {"feature_rotation_deg": rot, "feature_tx_mm": tx, "feature_ty_mm": ty,
                                  "marker_placement_mean_mm": float(d.mean()), "marker_placement_rmse_mm": float(np.sqrt(np.mean(d ** 2))),
                                  "marker_placement_max_mm": float(d.max()), "points": int(len(d))}
            log.info("[비교] 마커 기준으로 놓으면 같은 특징점이 평균 %.1f mm (최대 %.1f mm) 어긋납니다. 특징점 방식은 회전 %.3f°, 이동 (%.1f, %.1f) mm 를 더 주라고 합니다.",
                     d.mean(), d.max(), rot, tx, ty)
        else:
            res.warnings.append(f"특징점 방식과 비교하지 못했습니다: {res.reason or '특징점 정합 실패'}")
        return res

    # 특징점 방식
    if not res.feature_ok:
        return res
    res.success, res.t_fixed_from_moving = True, res.feature_t
    if not marker_problem:
        rot, tx, ty = rg.decompose(res.feature_t)
        res.marker_compare = {"feature_rotation_deg": rot, "feature_tx_mm": tx, "feature_ty_mm": ty,
                              "note": "같은 번호의 마커가 두 그림에 있습니다. 마커를 옮기지 않고 찍었다면 이 값이 0 에 가까워야 합니다. "
                                      "0 이 아니면 마커를 옮겼거나, 정사영 오차가 있다는 뜻입니다."}
    return res


def register_sequence(items: list[ImageItem], cfg: Config, log: logging.Logger, mode: str = "feature"):
    """첫 번째 그림을 기준으로 나머지를 차례로 붙인다 → (자세 목록, 붙인 상대 목록, 모든 두 장 정합 결과).

    그림 i 는 먼저 바로 앞 그림(i-1)에 붙여 본다. 안 되면 기준 그림, 그다음 이미 붙은 다른 그림에 붙여 본다.
    어디에도 못 붙이면 그 그림은 실패로 두고 (자세 = None) 배치하지 않는다.
    """
    n = len(items)
    poses: list[np.ndarray | None] = [None] * n       # T_ref_from_i
    partner: list[int | None] = [None] * n
    poses[0] = np.eye(3)
    pairs: list[PairResult] = []
    for i in range(1, n):
        order = [i - 1, 0] + [k for k in range(i - 2, 0, -1)]
        tried = []
        for k in order:
            if k in tried or poses[k] is None:
                continue
            tried.append(k)
            if len(tried) > 3:
                break
            res = register_pair(items[k], items[i], cfg, log, mode)
            pairs.append(res)
            if res.success:
                poses[i] = poses[k] @ res.t_fixed_from_moving     # ref ← k ← i
                partner[i] = k
                break
            log.warning("[실패] %s ← %s: %s", items[k].label, items[i].label, res.reason)
        if poses[i] is None:
            log.warning("[실패] %s 는 어느 그림에도 붙이지 못해 배치하지 않습니다.", items[i].label)
    return poses, partner, pairs


def evaluate_checkpoints(items: list[ImageItem], poses: list, spec: dict) -> list[dict]:
    """사용자가 준 '같은 실제 점' 이 정합 뒤에 얼마나 어긋나는지 잰다 (정합 계산에 쓰지 않은 독립된 확인).

    spec: {"points": [{"name": "P1", "pixels": {"그림이름": [u, v], "다른그림이름": [u, v]}}, ...]}  (정사영 그림 픽셀)
    """
    by_label = {it.label: k for k, it in enumerate(items)}
    out = []
    for pt in spec.get("points", []):
        name = str(pt.get("name", f"P{len(out) + 1}"))
        located = {}
        for label, px in dict(pt.get("pixels", {})).items():
            if label not in by_label:
                raise ValueError(f"검증 점 '{name}' 의 그림 이름 '{label}' 이 이번 정합에 없습니다 (있는 이름: {list(by_label)}).")
            k = by_label[label]
            if poses[k] is None:
                continue
            ref_from_pixel = poses[k] @ items[k].meta.plane_from_pixel
            located[label] = rg.apply(ref_from_pixel, np.array([px], dtype=np.float64))[0]
        labels = list(located)
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                d = located[labels[b]] - located[labels[a]]
                out.append({"point": name, "image_a": labels[a], "image_b": labels[b],
                            "dx_mm": float(d[0]), "dy_mm": float(d[1]), "error_mm": float(np.linalg.norm(d))})
    return out
