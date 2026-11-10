"""
정합 결과를 그림으로 만든다.

    draw_matches()   : 두 그림을 나란히 놓고 짝을 선으로 잇는다 (feature_matches.png, inlier_matches.png)
    plan_canvas()    : 여러 그림을 공통 평면(mm)에 놓았을 때의 전체 범위와 그림 크기를 정한다
    warp_to_canvas() : 그림 한 장을 공통 평면의 자리에 옮겨 그린다 (사진이 없는 검은 곳은 '없음' 으로 따로 기억)
    blend_overlay()  : 두 그림을 반투명하게 겹친다 (registration_overlay.png)
    color_overlay()  : 겹친 부분을 자홍/초록으로 겹친다. 잘 맞으면 회색, 어긋나면 색 테두리가 보인다
    mosaic()         : 여러 그림을 한 장으로 놓는다 (registered_mosaic.png)

합친 그림의 픽셀 (u, v) 는 공통 평면의 (origin_x + u × mm_per_pixel, origin_y + v × mm_per_pixel) 이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


@dataclass
class Canvas:
    mm_per_pixel: float
    origin_mm: tuple[float, float]     # 합친 그림의 픽셀 (0,0) 이 놓인 공통 평면 좌표
    width: int
    height: int
    coarsened: bool = False            # 크기 한도 때문에 축척을 거칠게 했는지

    @property
    def pixel_from_mm(self) -> np.ndarray:
        s, (ox, oy) = self.mm_per_pixel, self.origin_mm
        return np.array([[1.0 / s, 0.0, -ox / s], [0.0, 1.0 / s, -oy / s], [0.0, 0.0, 1.0]])

    @property
    def mm_from_pixel(self) -> np.ndarray:
        s, (ox, oy) = self.mm_per_pixel, self.origin_mm
        return np.array([[s, 0.0, ox], [0.0, s, oy], [0.0, 0.0, 1.0]])


def footprint_mm(valid: np.ndarray, target_from_pixel: np.ndarray) -> np.ndarray:
    """그림에서 사진이 있는 부분을 둘러싼 사각형의 네 귀퉁이를 목표 좌표(mm)로 옮긴 것 (4,2)."""
    x, y, w, h = cv2.boundingRect(valid)
    if w == 0 or h == 0:
        hh, ww = valid.shape[:2]
        x, y, w, h = 0, 0, ww, hh
    c = np.array([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]], dtype=np.float64)
    return c @ target_from_pixel[:2, :2].T + target_from_pixel[:2, 2]


def plan_canvas(footprints: list[np.ndarray], mm_per_pixel: float, cfg: Config) -> Canvas:
    """네 귀퉁이 목록을 모두 담는 그림의 크기를 정한다. 한도를 넘으면 축척을 거칠게 해서 줄인다 (범위는 자르지 않는다)."""
    pts = np.vstack(footprints)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w_mm, h_mm = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    mpp = float(mm_per_pixel)
    need = max(w_mm / cfg.max_canvas_side_px, h_mm / cfg.max_canvas_side_px, math.sqrt(w_mm * h_mm / cfg.max_canvas_pixels))
    coarsened = need > mpp
    if coarsened:
        mpp = need * 1.0001
    return Canvas(mpp, (float(x0), float(y0)), int(math.ceil(w_mm / mpp)) + 1, int(math.ceil(h_mm / mpp)) + 1, coarsened)


def warp_to_canvas(image: np.ndarray, valid: np.ndarray, common_from_pixel: np.ndarray, canvas: Canvas) -> tuple[np.ndarray, np.ndarray]:
    """그림을 공통 평면의 제자리에 옮겨 그린다 → (그림, 사진이 있는 곳). 변환은 회전·이동·축척뿐이라 2×3 으로 충분하다."""
    m = (canvas.pixel_from_mm @ common_from_pixel)[:2]
    size = (canvas.width, canvas.height)
    out = cv2.warpAffine(image, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(valid, m, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))       # 가장자리의 어두운 번짐을 한 픽셀 깎는다
    return out, mask


def blend_overlay(fixed: np.ndarray, fixed_mask: np.ndarray, moving: np.ndarray, moving_mask: np.ndarray, alpha: float) -> np.ndarray:
    """둘 다 있는 곳은 반투명하게 섞고, 한쪽만 있는 곳은 그 그림을 그대로 둔다. 검은 빈 곳은 섞지 않는다."""
    f, m = fixed_mask > 0, moving_mask > 0
    out = np.zeros_like(fixed)
    both = f & m
    out[f & ~m] = fixed[f & ~m]
    out[m & ~f] = moving[m & ~f]
    mixed = cv2.addWeighted(fixed, 1.0 - alpha, moving, alpha, 0.0)
    out[both] = mixed[both]
    return out


def color_overlay(fixed: np.ndarray, fixed_mask: np.ndarray, moving: np.ndarray, moving_mask: np.ndarray) -> np.ndarray:
    """고정 그림 = 자홍, 움직인 그림 = 초록으로 겹친다. 같은 무늬가 정확히 겹치면 회색으로 보인다."""
    gf = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    gm = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    f, m = fixed_mask > 0, moving_mask > 0
    gf = np.where(f, gf, 0).astype(np.uint8)
    gm = np.where(m, gm, 0).astype(np.uint8)
    return cv2.merge([gf, gm, gf])       # B = 고정, G = 움직임, R = 고정


def overlap_box(fixed_mask: np.ndarray, moving_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """두 그림이 겹치는 부분을 둘러싼 사각형 (x, y, w, h). 겹치지 않으면 None."""
    both = cv2.bitwise_and(fixed_mask, moving_mask)
    if cv2.countNonZero(both) == 0:
        return None
    return cv2.boundingRect(both)


def mosaic(layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """여러 그림을 한 장에 놓는다. 겹치는 곳은 앞 순서(기준 그림 먼저)의 그림을 쓴다 → 이음매에서 어긋남이 드러난다."""
    out = np.zeros_like(layers[0][0])
    filled = np.zeros(layers[0][1].shape, dtype=bool)
    for img, mask in layers:
        take = (mask > 0) & ~filled
        out[take] = img[take]
        filled |= take
    return out


def draw_keypoints(image: np.ndarray, points_px: np.ndarray, max_side: int) -> np.ndarray:
    """특징점을 점으로 찍은 그림 (디버깅용)."""
    h, w = image.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    out = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else image.copy()
    for x, y in np.round(points_px * s).astype(int):
        cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def _ascii(text: str, fallback: str) -> str:
    return text if text.isascii() else fallback       # OpenCV 글씨는 한글을 못 쓴다


def draw_matches(img_a: np.ndarray, img_b: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray, label_a: str, label_b: str,
                 cfg: Config, color: tuple[int, int, int], caption: str) -> np.ndarray:
    """두 그림을 나란히 놓고 짝을 선으로 잇는다. 선이 너무 많으면 고르게 뽑아 max_drawn_matches 개만 그린다."""
    def shrink(img):
        h, w = img.shape[:2]
        s = min(1.0, cfg.match_view_max_side_px / float(max(h, w)))
        small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
        return small, s

    a, sa = shrink(img_a)
    b, sb = shrink(img_b)
    top = 40
    height = max(a.shape[0], b.shape[0]) + top
    gap = 16
    canvas = np.full((height, a.shape[1] + gap + b.shape[1], 3), 255, np.uint8)
    canvas[top:top + a.shape[0], :a.shape[1]] = a
    x_b = a.shape[1] + gap
    canvas[top:top + b.shape[0], x_b:x_b + b.shape[1]] = b
    n = len(pts_a)
    idx = np.arange(n)
    if n > cfg.max_drawn_matches:
        idx = np.random.default_rng(0).choice(n, cfg.max_drawn_matches, replace=False)
    for k in idx:
        p = (int(round(pts_a[k, 0] * sa)), int(round(pts_a[k, 1] * sa)) + top)
        q = (int(round(pts_b[k, 0] * sb)) + x_b, int(round(pts_b[k, 1] * sb)) + top)
        cv2.line(canvas, p, q, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, q, 3, color, -1, cv2.LINE_AA)
    text = f"{_ascii(label_a, 'fixed')}  <-  {_ascii(label_b, 'moving')}    {caption}: {n}" + (f" (drawn {len(idx)})" if len(idx) < n else "")
    cv2.putText(canvas, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas
