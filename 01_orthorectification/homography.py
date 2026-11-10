"""
호모그래피(평면 변환) 계산과 사진 전체 정사영 보정.

    1. 마커 네 모서리(사진 픽셀) ↔ 마커 평면의 실제 좌표(mm) 로 변환 행렬 H 를 구한다.
    2. H 로 사진의 네 귀퉁이가 평면 위 어디로 가는지 보고, 출력 그림이 덮을 범위를 정한다.
    3. 범위가 잘리지 않게 옮기는 행렬을 곱해 사진 전체를 변환한다 (cv2.warpPerspective).

좌표 약속 (마커 평면, 단위 mm):
    원점 = 마커 좌상 모서리,  x = 좌상 → 우상 방향,  y = 좌상 → 좌하 방향.
출력 그림의 픽셀 (u, v) 는 평면의 (origin_x + u × mm_per_pixel,  origin_y + v × mm_per_pixel) 이다.

한계: 이 변환은 '마커와 같은 평면에 있는 것' 만 바르게 편다. 평면에서 솟았거나 꺼진 것은 틀린 자리에 놓인다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import Config


class HomographyError(Exception):
    """변환을 구할 수 없을 때 (마커가 선처럼 찌그러져 찍힌 경우 등)."""


def marker_plane_corners(size_mm: float) -> np.ndarray:
    """마커 네 모서리의 실제 좌표 (mm): 좌상, 우상, 우하, 좌하. 마커 크기를 바꾸면 같이 바뀐다."""
    s = float(size_mm)
    return np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]])


def compute_homography(corners_px: np.ndarray, size_mm: float) -> np.ndarray:
    """사진 픽셀 → 마커 평면 mm 로 가는 3×3 행렬. 마커 가운데에서 w = 1 이 되도록 크기를 맞춘다."""
    src = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    if size_mm <= 0:
        raise HomographyError("마커 크기는 0 보다 커야 합니다.")
    x, y = src[:, 0], src[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 16.0:
        raise HomographyError("마커가 너무 작거나 선처럼 찌그러져 찍혀서 변환을 구할 수 없습니다.")
    h = cv2.getPerspectiveTransform(src.astype(np.float32), marker_plane_corners(size_mm).astype(np.float32))
    if not np.all(np.isfinite(h)):
        raise HomographyError("변환 행렬을 구하지 못했습니다.")
    c = src.mean(axis=0)
    w = h[2, 0] * c[0] + h[2, 1] * c[1] + h[2, 2]
    if abs(w) < 1e-12:
        raise HomographyError("변환 행렬이 불안정합니다.")
    return h / w


def apply_homography(h: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """점 (N,2) 을 변환한다 → (변환된 점 (N,2), w 값 (N,)). w 가 0 에 가까우면 지평선 근처다."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ h.T
    w = q[:, 2]
    safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return q[:, :2] / safe[:, None], w


def _clip_by_w(poly: np.ndarray, h: np.ndarray, min_w: float) -> np.ndarray:
    """다각형에서 w ≥ min_w 인 쪽만 남긴다 (지평선 너머와 너무 먼 곳을 잘라냄)."""
    def w_of(p):
        return h[2, 0] * p[0] + h[2, 1] * p[1] + h[2, 2] - min_w

    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        wa, wb = w_of(a), w_of(b)
        if wa >= 0:
            out.append(a)
        if (wa >= 0) != (wb >= 0):
            t = wa / (wa - wb)
            out.append(a + t * (b - a))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


@dataclass
class OrthoPlan:
    h_image_to_mm: np.ndarray            # 사진 픽셀 → 평면 mm
    h_image_to_output: np.ndarray        # 사진 픽셀 → 출력 그림 픽셀
    width: int                           # 출력 그림 크기 (픽셀)
    height: int
    mm_per_pixel: float
    origin_mm: tuple[float, float]       # 출력 그림 픽셀 (0,0) 의 평면 좌표 (mm)
    full_range_mm: tuple[float, float, float, float]   # 사진 전체가 덮는 범위 (x_min, y_min, x_max, y_max)
    horizon_cut: bool = False            # 지평선 쪽 먼 곳을 버렸는지
    cropped: bool = False                # 크기 한도 때문에 범위를 잘랐는지
    notes: list[str] = field(default_factory=list)

    def output_to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        """출력 그림의 픽셀 좌표 (N,2) → 평면 좌표 (mm)."""
        return np.asarray(pts_px, dtype=np.float64).reshape(-1, 2) * self.mm_per_pixel + np.array(self.origin_mm)


def _fit_window(lo: float, hi: float, size: float, anchor: float, keep_lo: float, keep_hi: float) -> tuple[float, float]:
    """[lo, hi] 안에서 길이 size 인 창을 고른다. anchor 를 가운데 두되, [keep_lo, keep_hi](마커)가 들어오게 민다."""
    if hi - lo <= size:
        return lo, hi
    a = min(max(anchor - size / 2.0, lo), hi - size)
    if size >= keep_hi - keep_lo:
        a = min(a, keep_lo)
        a = max(a, keep_hi - size)
        a = min(max(a, lo), hi - size)
    return a, a + size


def plan_output(h: np.ndarray, image_size: tuple[int, int], marker_size_mm: float, cfg: Config) -> OrthoPlan:
    """사진 전체가 평면 위에서 차지하는 범위를 구하고, 출력 그림 크기와 옮김 행렬을 정한다."""
    mpp = float(cfg.mm_per_pixel)
    if mpp <= 0:
        raise HomographyError("mm_per_pixel 은 0 보다 커야 합니다.")
    w_img, h_img = image_size
    rect = np.array([[0.0, 0.0], [w_img - 1.0, 0.0], [w_img - 1.0, h_img - 1.0], [0.0, h_img - 1.0]])

    # 1) 사진 네 귀퉁이 중 지평선 너머(또는 너무 먼 곳)에 해당하는 부분을 잘라낸다
    _, w_corner = apply_homography(h, rect)
    poly = _clip_by_w(rect, h, cfg.horizon_min_w)
    if len(poly) < 3:
        raise HomographyError("사진에서 평면으로 펼 수 있는 부분이 없습니다.")
    horizon_cut = bool(np.any(w_corner < cfg.horizon_min_w))

    # 2) 남은 부분이 평면 위에서 차지하는 범위 (mm)
    mm, _ = apply_homography(h, poly)
    x_min, y_min = mm.min(axis=0)
    x_max, y_max = mm.max(axis=0)
    full = (float(x_min), float(y_min), float(x_max), float(y_max))

    # 3) 범위가 너무 크면 잘라낸다 (축척은 그대로 두고 범위만 줄인다)
    notes = []
    centre_mm, w_centre = apply_homography(h, np.array([[(w_img - 1) / 2.0, (h_img - 1) / 2.0]]))
    anchor = centre_mm[0] if w_centre[0] >= cfg.horizon_min_w else np.array([marker_size_mm / 2.0] * 2)
    max_side_mm = cfg.max_output_side_px * mpp
    side_x = min(x_max - x_min, max_side_mm)
    side_y = min(y_max - y_min, max_side_mm)
    over = (side_x / mpp) * (side_y / mpp) / float(cfg.max_output_pixels)
    if over > 1.0:
        side_x /= math.sqrt(over)
        side_y /= math.sqrt(over)
    cropped = bool(side_x < (x_max - x_min) - 1e-6 or side_y < (y_max - y_min) - 1e-6)
    margin = 0.25 * marker_size_mm
    x_min, x_max = _fit_window(x_min, x_max, side_x, anchor[0], -margin, marker_size_mm + margin)
    y_min, y_max = _fit_window(y_min, y_max, side_y, anchor[1], -margin, marker_size_mm + margin)
    if horizon_cut:
        notes.append("사진의 일부가 지평선 쪽(너무 먼 곳)이라 그 부분은 버렸습니다.")
    if cropped:
        notes.append(f"사진 전체 범위({full[2] - full[0]:.0f} × {full[3] - full[1]:.0f} mm)가 출력 한도를 넘어 "
                     f"{x_max - x_min:.0f} × {y_max - y_min:.0f} mm 만 담았습니다. 전체를 담으려면 mm_per_pixel 을 키우세요.")

    width = max(1, int(math.ceil((x_max - x_min) / mpp)))
    height = max(1, int(math.ceil((y_max - y_min) / mpp)))

    # 4) 평면 mm → 출력 픽셀: 범위의 왼쪽 위가 (0,0) 이 되게 옮기고 축척을 곱한다
    shift = np.array([[1.0 / mpp, 0.0, -x_min / mpp], [0.0, 1.0 / mpp, -y_min / mpp], [0.0, 0.0, 1.0]])
    return OrthoPlan(h, shift @ h, width, height, mpp, (float(x_min), float(y_min)), full, horizon_cut, cropped, notes)


def warp_image(bgr: np.ndarray, plan: OrthoPlan) -> np.ndarray:
    """사진 전체를 정사영 그림으로 변환한다. 사진에 없던 빈 곳은 검은색."""
    return cv2.warpPerspective(bgr, plan.h_image_to_output, (plan.width, plan.height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def side_lengths(points: np.ndarray) -> np.ndarray:
    """네 점을 차례로 이은 네 변의 길이: 위, 오른쪽, 아래, 왼쪽."""
    p = np.asarray(points, dtype=np.float64).reshape(4, 2)
    return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)


def measure_lengths(h: np.ndarray, points_px: dict, lengths: list[dict]) -> list[dict]:
    """길이를 아는 구간을 재서 기준과 비교한다 (마커와 따로 있는 점이므로 독립된 정확도 확인이 된다).

    points_px: {"A": [x, y], ...} 원본 사진 픽셀.   lengths: [{"from": "A", "to": "B", "reference_mm": 100}, ...]
    """
    out = []
    for item in lengths:
        a, b, ref = str(item["from"]), str(item["to"]), float(item["reference_mm"])
        if a not in points_px or b not in points_px:
            raise HomographyError(f"검증 파일에 점 '{a if a not in points_px else b}' 의 위치가 없습니다.")
        mm, w = apply_homography(h, np.array([points_px[a], points_px[b]], dtype=np.float64))
        if np.any(w <= 0) or ref <= 0:
            raise HomographyError(f"검증 구간 {a}-{b} 를 계산할 수 없습니다 (점이 평면 범위 밖이거나 기준 길이가 0 이하).")
        got = float(np.linalg.norm(mm[0] - mm[1]))
        out.append({"name": f"{a}-{b}", "reference_mm": ref, "measured_mm": got,
                    "error_mm": got - ref, "error_percent": (got - ref) / ref * 100.0})
    return out


def estimate_extrapolation_error(corners_px: np.ndarray, size_mm: float, image_size: tuple[int, int], noise_px: float,
                                 distances_mm=(500, 1000, 2000, 3000, 5000), trials: int = 300, seed: int = 0) -> list[dict]:
    """모서리를 noise_px 만큼 틀리게 찾았다고 가정했을 때, 마커에서 떨어진 거리별로 위치가 얼마나 틀어지는지 예상한다.

    마커 하나로 사진 전체를 펴면 마커에서 멀수록 오차가 빠르게 커진다. 그 크기를 가늠하기 위한 값이며,
    렌즈 왜곡이나 바닥이 평평하지 않은 것은 들어 있지 않다 (실제 오차는 이보다 클 수 있다).
    """
    rng = np.random.default_rng(seed)
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    h0 = compute_homography(corners, size_mm)
    h0_inv = np.linalg.inv(h0)
    centre = np.array([size_mm / 2.0, size_mm / 2.0])
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    w_img, h_img = image_size
    hs = []
    for _ in range(trials):
        try:
            hs.append(compute_homography(corners + rng.normal(0, noise_px, (4, 2)), size_mm))
        except HomographyError:
            continue
    out = []
    for d in distances_mm:
        ring = centre + d * np.column_stack([np.cos(ang), np.sin(ang)])
        px, w = apply_homography(h0_inv, ring)
        ok = (w > 0) & (px[:, 0] >= 0) & (px[:, 0] <= w_img - 1) & (px[:, 1] >= 0) & (px[:, 1] <= h_img - 1)
        if not ok.any() or not hs:
            out.append({"distance_mm": float(d), "in_photo": False, "typical_error_mm": None, "p95_error_mm": None})
            continue
        err = np.concatenate([np.linalg.norm(apply_homography(h, px[ok])[0] - ring[ok], axis=1) for h in hs])
        err = err[np.isfinite(err)]
        out.append({"distance_mm": float(d), "in_photo": True,
                    "typical_error_mm": float(np.sqrt(np.mean(err ** 2))), "p95_error_mm": float(np.percentile(err, 95))})
    return out
