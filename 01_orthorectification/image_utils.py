"""
사진 읽기·저장·화면 표시.

- 사진은 Pillow 로 읽는다: 스마트폰 사진의 회전 정보(EXIF)를 적용할 수 있고, 한글이 든 경로도 읽을 수 있다.
- 저장은 메모리에서 압축한 뒤 파일로 쓴다: cv2.imwrite 는 한글 경로에서 실패하기 때문이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

# 스마트폰의 고해상도 사진(1억 화소 이상)도 열 수 있게 Pillow 의 크기 경고 한도를 넓힌다
Image.MAX_IMAGE_PIXELS = 400_000_000

SUPPORTED = (".jpg", ".jpeg", ".png", ".heic", ".heif")


class ImageLoadError(Exception):
    """사진을 읽지 못했을 때."""


def enable_heic() -> bool:
    """HEIC 지원을 켠다. pillow-heif 가 설치돼 있지 않으면 False."""
    try:
        import pillow_heif
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    return True


def pick_image_file(initial_dir: Path) -> str | None:
    """Windows 파일 선택 창을 띄운다. 취소하면 None."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)   # 선택 창이 다른 창 뒤에 숨지 않게
    try:
        path = filedialog.askopenfilename(
            title="스마트폰 사진 선택",
            initialdir=str(initial_dir) if Path(initial_dir).is_dir() else None,
            filetypes=[("사진", "*.jpg *.jpeg *.png *.heic *.heif"), ("모든 파일", "*.*")],
        )
    finally:
        root.destroy()
    return path or None


def show_message(title: str, text: str, error: bool = False) -> None:
    """알림 창. 창을 띄울 수 없는 환경이면 조용히 넘어간다 (내용은 콘솔에도 찍는다)."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        (messagebox.showerror if error else messagebox.showinfo)(title, text, parent=root)
        root.destroy()
    except Exception:
        pass


def load_image(path: str) -> tuple[np.ndarray, dict]:
    """사진을 읽어 (BGR 그림, 정보) 를 돌려준다. 회전 정보를 적용하고 해상도는 그대로 둔다."""
    p = Path(path)
    if not p.is_file():
        raise ImageLoadError(f"파일이 없습니다: {path}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise ImageLoadError(f"지원하지 않는 형식입니다: {ext or '(확장자 없음)'}  (지원: JPG, JPEG, PNG, HEIC)")
    if ext in (".heic", ".heif") and not enable_heic():
        raise ImageLoadError("HEIC 사진을 읽으려면 pillow-heif 가 필요합니다:  pip install pillow-heif")
    try:
        with Image.open(p) as im:
            fmt = im.format
            orientation = im.getexif().get(0x0112, 1)      # 1 = 회전 없음
            im = ImageOps.exif_transpose(im)                # 스마트폰을 세워 찍은 사진을 바로 세운다
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                # 16비트 PNG 등: 8비트로 줄인다
                a = np.asarray(im, dtype=np.float64)
                top = a.max() if a.max() > 0 else 1.0
                rgb = np.repeat((a / top * 255.0).astype(np.uint8)[:, :, None], 3, axis=2)
            else:
                rgb = np.asarray(im.convert("RGB"))
    except UnidentifiedImageError as e:
        raise ImageLoadError(f"사진 파일로 읽을 수 없습니다 (깨졌거나 다른 형식): {p.name}") from e
    except (OSError, ValueError, MemoryError) as e:
        raise ImageLoadError(f"사진을 읽는 중 오류: {p.name}  ({e})") from e
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    info = {"file": p.name, "format": fmt, "width": int(bgr.shape[1]), "height": int(bgr.shape[0]),
            "exif_orientation": int(orientation or 1), "exif_rotation_applied": bool(orientation not in (1, None))}
    return bgr, info


def save_image(path: Path, bgr: np.ndarray) -> None:
    """그림을 저장한다 (한글 경로 가능). 확장자로 형식을 정한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise OSError(f"그림을 저장 형식으로 바꾸지 못했습니다: {path.name}")
    buf.tofile(str(path))


def _plain(value):
    """numpy 값을 JSON 이 아는 보통 값으로 바꾼다."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"JSON 으로 저장할 수 없는 값: {type(value).__name__}")


def save_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, default=_plain)   # 다 만든 뒤에 쓴다 (반쯤 쓴 파일이 남지 않게)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------- 렌즈 보정 (보정값이 있을 때만)

def load_calibration(path: str) -> dict:
    """카메라 보정 파일을 읽는다. 형식은 config.py 설명 참고."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    k = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).ravel()
    if k.shape != (3, 3) or dist.size < 4:
        raise ValueError("camera_matrix 는 3×3, dist_coeffs 는 4개 이상이어야 합니다.")
    return {"camera_matrix": k, "dist_coeffs": dist, "width": int(d["image_width"]), "height": int(d["image_height"])}


def undistort_image(bgr: np.ndarray, calib: dict) -> tuple[np.ndarray | None, str]:
    """렌즈 왜곡을 편다. 보정값의 해상도가 사진과 다르면 보정하지 않고 (None, 이유) 를 돌려준다.

    해상도가 다른 사진에 보정값을 억지로 맞추면 오히려 틀어지므로, 값을 지어내지 않고 건너뛴다.
    """
    h, w = bgr.shape[:2]
    if (w, h) != (calib["width"], calib["height"]):
        why = "방향이 다릅니다 (가로·세로가 바뀜)" if (h, w) == (calib["width"], calib["height"]) else "해상도가 다릅니다"
        return None, f"보정 파일({calib['width']}×{calib['height']})과 사진({w}×{h})의 {why}"
    return cv2.undistort(bgr, calib["camera_matrix"], calib["dist_coeffs"]), ""


# ---------------------------------------------------------------- 화면 표시

def fit_to_box(bgr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """그림이 상자보다 크면 비율을 지켜 줄인다 (작으면 그대로)."""
    h, w = bgr.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _label(bgr: np.ndarray, text: str, fallback: str) -> np.ndarray:
    """그림 왼쪽 위에 제목을 쓴다. 한글 글꼴(맑은 고딕)이 없으면 영어 제목으로 쓴다."""
    size = max(18, bgr.shape[0] // 28)
    try:
        font = ImageFont.truetype("malgun.ttf", size)
    except OSError:
        out = bgr.copy()
        cv2.putText(out, fallback, (12, size + 12), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, fallback, (12, size + 12), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    im = Image.fromarray(bgr[:, :, ::-1])
    draw = ImageDraw.Draw(im)
    box = draw.textbbox((12, 10), text, font=font)
    draw.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
    draw.text((12, 10), text, font=font, fill=(255, 255, 255))
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def make_comparison(left: np.ndarray, right: np.ndarray, panel_height: int = 1200) -> np.ndarray:
    """왼쪽(원본)·오른쪽(정사영)을 같은 높이로 나란히 붙인다. 화면·비교용이므로 줄여서 만든다."""
    panels = []
    for img, text, fallback in ((left, "원본 (마커 검출)", "Original"), (right, "정사영 보정", "Orthorectified")):
        h, w = img.shape[:2]
        s = min(panel_height / h, 2400 / w)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
        canvas = np.zeros((panel_height, small.shape[1], 3), np.uint8)
        y0 = (panel_height - small.shape[0]) // 2
        canvas[y0:y0 + small.shape[0]] = small
        panels.append(_label(canvas, text, fallback))
    gap = np.full((panel_height, 12, 3), 255, np.uint8)
    return np.hstack([panels[0], gap, panels[1]])


def append_table(bgr: np.ndarray, title: str, headers: list[str], rows: list[list[str]], note: str = "") -> np.ndarray:
    """그림 아래에 흰 바탕의 표를 붙인다 (제목 한 줄, 머리글, 내용, 설명 한 줄)."""
    width = bgr.shape[1]
    size = max(20, width // 70)
    try:
        font = ImageFont.truetype("malgun.ttf", size)
        bold = ImageFont.truetype("malgunbd.ttf", size)
    except OSError:
        font = bold = ImageFont.load_default()      # 한글 글꼴이 없는 PC: 글자가 깨질 수 있다
    line = int(size * 1.9)
    pad = size
    n_col = len(headers)
    col_w = min((width - 2 * pad) // n_col, size * 13)
    table_w = col_w * n_col
    height = pad + line + (len(rows) + 1) * line + (line if note else 0) + pad
    im = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text((pad, pad), title, font=bold, fill=(0, 0, 0))
    top = pad + line
    draw.rectangle((pad, top, pad + table_w, top + line), fill=(230, 230, 230))
    for r, cells in enumerate([headers] + rows):
        y = top + r * line
        for c, text in enumerate(cells):
            x0 = pad + c * col_w
            w = draw.textlength(str(text), font=bold if r == 0 else font)
            # 첫 칸(이름)은 왼쪽, 숫자 칸은 오른쪽으로 붙인다
            x = x0 + size // 2 if c == 0 else x0 + col_w - size // 2 - w
            draw.text((x, y + (line - size) // 2 - size // 6), str(text), font=bold if r == 0 else font, fill=(0, 0, 0))
    for r in range(len(rows) + 2):
        draw.line((pad, top + r * line, pad + table_w, top + r * line), fill=(120, 120, 120), width=1)
    for c in range(n_col + 1):
        draw.line((pad + c * col_w, top, pad + c * col_w, top + (len(rows) + 1) * line), fill=(120, 120, 120), width=1)
    if note:
        draw.text((pad, top + (len(rows) + 1) * line + size // 3), note, font=font, fill=(90, 90, 90))
    return np.vstack([bgr, np.ascontiguousarray(np.asarray(im)[:, :, ::-1])])


def show_window(title: str, bgr: np.ndarray, max_w: int, max_h: int) -> None:
    """그림을 창에 띄우고, 아무 키를 누르거나 창을 닫을 때까지 기다린다."""
    try:
        cv2.imshow(title, fit_to_box(bgr, max_w, max_h))
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
