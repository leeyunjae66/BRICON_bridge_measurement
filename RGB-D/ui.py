"""화면 표시와 그림 저장에 쓰는 작은 도구들."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("malgun.ttf", size)        # 한글 (맑은 고딕)
    except OSError:
        return ImageFont.load_default()


def banner(width: int, lines: list[str], size: int = 18) -> np.ndarray:
    """그림 위에 붙일 글 띠. OpenCV 글씨는 한글을 못 쓰므로 Pillow 로 그린다."""
    line_h = int(size * 1.55)
    im = Image.new("RGB", (width, line_h * len(lines) + 8), (32, 32, 32))
    draw = ImageDraw.Draw(im)
    font = _font(size)
    for k, text in enumerate(lines):
        draw.text((10, 4 + k * line_h), text, font=font, fill=(255, 255, 255))
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def colorize_depth(depth_mm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """깊이를 색으로: 가까움 = 빨강, 멂 = 파랑, 깊이가 없는 곳 = 검정."""
    ok = np.isfinite(depth_mm)
    norm = np.clip((np.where(ok, depth_mm, hi) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out[~ok] = 0
    return out


def fit_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    return min(max_w / w, max_h / h, 1.0)


def save_image(path: Path, image: np.ndarray) -> None:
    """그림 저장 (한글 경로 가능, 16비트 PNG 가능)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"그림을 저장 형식으로 바꾸지 못했습니다: {path.name}")
    buf.tofile(str(path))


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise OSError(f"그림을 읽지 못했습니다: {path}")
    return img


def window_closed(title: str) -> bool:
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
