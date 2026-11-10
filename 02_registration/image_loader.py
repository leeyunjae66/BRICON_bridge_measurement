"""
그림 읽기·저장, 파일 선택 창, 가림 그림(마스크) 만들기.

가림 그림은 0 = 쓰지 않는 곳, 255 = 쓰는 곳이다.
    valid_mask   : 정사영 그림에서 사진이 실제로 있는 곳 (검은 빈 영역 제외)
    marker_mask  : 마커와 그 둘레 종이를 뺀 곳
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageError(Exception):
    """그림을 읽거나 쓰지 못했을 때."""


def pick_images(initial_dir: Path) -> list[str]:
    """정사영 그림을 2장 이상 고르는 창. 취소하면 빈 목록."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        paths = filedialog.askopenfilenames(
            title="정합할 정사영 그림 선택 (2장 이상, 첫 번째가 기준)",
            initialdir=str(initial_dir) if Path(initial_dir).is_dir() else None,
            filetypes=[("그림", "*.png *.jpg *.jpeg"), ("모든 파일", "*.*")])
    finally:
        root.destroy()
    return list(paths)


def pick_metadata(image_name: str, initial_dir: Path) -> str | None:
    """그림 옆에서 메타데이터를 못 찾았을 때 직접 고르는 창. 취소하면 None."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title=f"'{image_name}' 의 정사영 메타데이터(homography.json) 선택",
            initialdir=str(initial_dir) if Path(initial_dir).is_dir() else None,
            filetypes=[("JSON", "*.json"), ("모든 파일", "*.*")])
    finally:
        root.destroy()
    return path or None


def read_image(path: Path) -> np.ndarray:
    """그림을 BGR 로 읽는다. 한글 경로도 읽을 수 있게 파일을 통째로 읽은 뒤 푼다."""
    p = Path(path)
    if not p.is_file():
        raise ImageError(f"파일이 없습니다: {p}")
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError as e:
        raise ImageError(f"파일을 읽지 못했습니다: {p.name} ({e})") from e
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ImageError(f"그림 파일로 읽을 수 없습니다 (깨졌거나 다른 형식): {p.name}")
    return img


def write_image(path: Path, image: np.ndarray) -> None:
    """그림을 저장한다 (한글 경로 가능)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise ImageError(f"그림을 저장 형식으로 바꾸지 못했습니다: {path.name}")
    buf.tofile(str(path))


def make_labels(paths: list[Path]) -> list[str]:
    """그림마다 구분할 이름을 붙인다. 1단계 결과는 파일 이름이 모두 orthorectified.png 이므로 폴더 이름을 쓴다."""
    labels = []
    for p in paths:
        p = Path(p)
        name = p.parent.name if p.stem.lower() == "orthorectified" and p.parent.name else p.stem
        base, k = name, 2
        while name in labels:
            name = f"{base}_{k}"
            k += 1
        labels.append(name)
    return labels


def valid_mask(image: np.ndarray) -> np.ndarray:
    """사진이 실제로 있는 곳. 정사영하면서 생긴 검은 빈 영역(값이 정확히 0)을 뺀다."""
    mask = (image.max(axis=2) > 0).astype(np.uint8) * 255
    # 사진 안의 아주 어두운 점 때문에 생긴 작은 구멍은 메운다
    k = max(3, (max(image.shape[:2]) // 400) | 1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


def marker_mask(shape: tuple[int, int], corners_px: np.ndarray | None, scale: float) -> np.ndarray:
    """마커 네 모서리를 중심에서 scale 배로 넓힌 사각형을 뺀 가림 그림. 마커 정보가 없으면 전부 쓰는 곳."""
    mask = np.full(shape[:2], 255, np.uint8)
    if corners_px is None:
        return mask
    c = np.asarray(corners_px, dtype=np.float64)
    centre = c.mean(axis=0)
    poly = np.round(centre + (c - centre) * scale).astype(np.int32)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 0)
    return mask


def fit_to_box(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def show_window(title: str, image: np.ndarray, max_w: int, max_h: int) -> None:
    """그림을 창에 띄우고, 아무 키를 누르거나 창을 닫을 때까지 기다린다."""
    try:
        cv2.imshow(title, fit_to_box(image, max_w, max_h))
    except cv2.error:
        print("[알림] 화면 창을 띄울 수 없는 환경입니다. 저장된 파일을 열어 확인하세요.")
        return
    while True:
        if cv2.waitKey(50) != -1:
            break
        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()
