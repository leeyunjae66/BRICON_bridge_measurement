"""
정사영 결과의 메타데이터(homography.json) 읽기와 좌표계 정의.

이 모듈은 1단계(정사영) 코드를 불러 쓰지 않는다. 1단계가 저장한 파일만 읽는다.

좌표계 네 가지
    1. 원본 사진 픽셀        : 스마트폰 사진 (이 모듈에서는 쓰지 않는다. 1단계 파일에 변환이 들어 있을 뿐)
    2. 정사영 그림 픽셀 (u,v) : orthorectified.png 의 픽셀
    3. 마커 평면 좌표 (mm)    : 그 사진의 마커 좌상 모서리가 원점. x = 좌상 → 우상, y = 좌상 → 좌하
    4. 공통 평면 좌표 (mm)    : 기준 그림(첫 번째 그림)의 마커 평면 좌표

행렬 이름 약속:  A_from_B 는 'B 좌표를 A 좌표로 바꾸는' 3×3 행렬이다.  p_A = A_from_B · p_B
    plane_from_pixel   : 정사영 픽셀 → 그 그림의 마커 평면 mm      (1단계 파일의 축척·원점에서 만든다)
    T_ref_from_i       : 그림 i 의 평면 mm → 기준 그림의 평면 mm   (이 모듈이 구하는 것)
    ref_from_pixel_i   = T_ref_from_i · plane_from_pixel_i

1단계 파일에서 꼭 필요한 것
    output.mm_per_pixel   픽셀 하나가 몇 mm 인지
    output.origin_mm      정사영 그림의 픽셀 (0,0) 이 평면의 어디인지 (mm)
    output.width_px, output.height_px   그림 크기 (그림과 파일이 짝이 맞는지 확인용)
있으면 쓰는 것
    marker.corners_output_px   정사영 그림에서 마커 네 모서리 (마커 가리기, 마커 기준 방식)
    marker.id, marker.dictionary, marker.size_mm
이 값들이 없으면 지어내지 않고 오류로 알린다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class MetadataError(Exception):
    """메타데이터가 없거나, 필요한 값이 빠졌거나, 그림과 맞지 않을 때."""


NEEDED = ("필요한 값: output.mm_per_pixel (픽셀당 mm), output.origin_mm (그림의 픽셀 (0,0) 이 놓인 평면 좌표 mm), "
          "output.width_px, output.height_px")


@dataclass
class OrthoMeta:
    mm_per_pixel: float
    origin_mm: tuple[float, float]
    width: int
    height: int
    path: Path | None = None
    marker_corners_px: np.ndarray | None = None     # (4,2) 정사영 그림 픽셀: 좌상, 우상, 우하, 좌하
    marker_id: int | None = None
    marker_dictionary: str | None = None
    marker_size_mm: float | None = None
    source_file: str | None = None
    from_user: bool = False                         # 파일 없이 사용자가 축척만 준 경우 (원점은 그림의 (0,0), 마커 정보 없음)
    notes: list[str] = field(default_factory=list)

    @property
    def plane_from_pixel(self) -> np.ndarray:
        """정사영 픽셀 → 마커 평면 mm."""
        s, (ox, oy) = self.mm_per_pixel, self.origin_mm
        return np.array([[s, 0.0, ox], [0.0, s, oy], [0.0, 0.0, 1.0]])

    @property
    def pixel_from_plane(self) -> np.ndarray:
        """마커 평면 mm → 정사영 픽셀."""
        return np.linalg.inv(self.plane_from_pixel)

    @property
    def has_marker(self) -> bool:
        return self.marker_corners_px is not None and not self.from_user


def find_metadata(image_path: Path) -> Path | None:
    """그림 옆에서 메타데이터 파일을 찾는다. 1단계는 같은 폴더에 homography.json 으로 저장한다."""
    p = Path(image_path)
    for cand in (p.parent / "homography.json", p.with_suffix(".json"), p.parent / f"{p.stem}_homography.json"):
        if cand.is_file():
            return cand
    return None


def load_metadata(json_path: Path, image_size: tuple[int, int] | None = None) -> OrthoMeta:
    """1단계가 저장한 homography.json 을 읽는다. image_size = (가로, 세로) 를 주면 그림과 짝이 맞는지도 본다."""
    path = Path(json_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        raise MetadataError(f"메타데이터 파일을 읽지 못했습니다: {path.name} ({e})") from e
    out = d.get("output") if isinstance(d, dict) else None
    if not isinstance(out, dict):
        raise MetadataError(f"{path.name} 에 'output' 항목이 없습니다. {NEEDED}")
    missing = [k for k in ("mm_per_pixel", "origin_mm", "width_px", "height_px") if k not in out]
    if missing:
        raise MetadataError(f"{path.name} 의 output 에 {missing} 이(가) 없습니다. {NEEDED}")
    try:
        mpp = float(out["mm_per_pixel"])
        origin = (float(out["origin_mm"][0]), float(out["origin_mm"][1]))
        w, h = int(out["width_px"]), int(out["height_px"])
    except (TypeError, ValueError, IndexError) as e:
        raise MetadataError(f"{path.name} 의 output 값을 숫자로 읽지 못했습니다 ({e}).") from e
    if not (mpp > 0 and np.isfinite(mpp) and np.all(np.isfinite(origin))):
        raise MetadataError(f"{path.name} 의 축척 또는 원점 값이 올바르지 않습니다 (mm_per_pixel = {mpp}, origin_mm = {origin}).")
    if image_size is not None and (w, h) != tuple(image_size):
        raise MetadataError(f"{path.name} 은 {w}×{h} 그림의 정보인데 고른 그림은 {image_size[0]}×{image_size[1]} 입니다. 짝이 맞는 파일인지 확인하세요.")
    meta = OrthoMeta(mpp, origin, w, h, path=path)
    marker = d.get("marker") if isinstance(d.get("marker"), dict) else {}
    corners = marker.get("corners_output_px")
    if corners is not None:
        c = np.array(corners, dtype=np.float64)
        if c.shape == (4, 2) and np.all(np.isfinite(c)):
            meta.marker_corners_px = c
    meta.marker_id = marker.get("id")
    meta.marker_dictionary = marker.get("dictionary")
    meta.marker_size_mm = marker.get("size_mm")
    src = d.get("source_image") if isinstance(d.get("source_image"), dict) else {}
    meta.source_file = src.get("file")
    if out.get("cropped"):
        meta.notes.append("1단계에서 출력 한도 때문에 범위를 잘라낸 그림입니다.")
    if out.get("horizon_cut"):
        meta.notes.append("1단계에서 지평선 쪽 먼 곳을 버린 그림입니다 (먼 곳은 많이 늘어나 있어 정합이 어렵습니다).")
    if not src.get("lens_undistorted", False):
        meta.notes.append("렌즈 왜곡을 보정하지 않은 사진으로 만든 정사영입니다.")
    return meta


def manual_metadata(mm_per_pixel: float, width: int, height: int) -> OrthoMeta:
    """메타데이터 파일이 없을 때 사용자가 축척을 직접 준 경우. 원점은 그림의 픽셀 (0,0) 으로 두고, 마커 기준 방식은 쓸 수 없다."""
    if not mm_per_pixel > 0:
        raise MetadataError("mm_per_pixel 은 0 보다 커야 합니다.")
    return OrthoMeta(float(mm_per_pixel), (0.0, 0.0), width, height, from_user=True,
                     notes=["메타데이터 파일 없이 사용자가 준 축척을 썼습니다. 원점은 그림의 픽셀 (0,0) 입니다."])


def save_json(path: Path, data: dict) -> None:
    """결과를 JSON 으로 저장한다 (numpy 값은 보통 값으로 바꾼다)."""
    def plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(f"JSON 으로 저장할 수 없는 값: {type(v).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, default=plain)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
