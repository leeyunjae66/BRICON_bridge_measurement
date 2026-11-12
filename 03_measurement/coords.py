"""
그림의 픽셀을 실제 평면 좌표(mm)로 바꾸는 부분. 화면(GUI)과는 따로 떨어져 있어 혼자 시험할 수 있다.

앞 단계의 코드를 불러 쓰지 않고, 그림 옆에 저장된 JSON 만 읽는다. 열 수 있는 그림은 세 가지다.

    종류      파일                                  좌표 정보                     픽셀 → mm
    ortho     1단계 orthorectified.png              homography.json 의 output      원점 + 픽셀 × 축척
    photo     1단계 aruco_detected.png (원본 사진)   homography.json 의 변환행렬    평면 변환(호모그래피)
    mosaic    2단계 registered_mosaic.png           registration.json 의 mosaic    원점 + 픽셀 × 축척

좌표 약속: 픽셀 (u, v) 는 픽셀 가운데 기준이다 (맨 왼쪽 위 픽셀의 가운데가 (0, 0)).
평면 좌표는 mm 이고, 원점은 마커 좌상 모서리다 (mosaic 은 기준 그림의 마커).
좌표 정보가 없거나 그림과 맞지 않으면 추측하지 않고 CoordError 로 알린다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class CoordError(Exception):
    """그림의 좌표 정보를 찾지 못했거나 쓸 수 없을 때."""


KIND_NAMES = {"ortho": "정사영 그림 (1단계)", "photo": "원본 사진 (1단계)", "mosaic": "합친 그림 (2단계)"}


def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """네 점 src (…,4,2) 를 dst (4,2) 로 보내는 3×3 평면 변환. src 를 여러 묶음 주면 한꺼번에 푼다."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    x, y = src[..., 0], src[..., 1]
    big_x, big_y = np.broadcast_to(dst[:, 0], x.shape), np.broadcast_to(dst[:, 1], x.shape)
    zero, one = np.zeros_like(x), np.ones_like(x)
    rows_x = np.stack([x, y, one, zero, zero, zero, -x * big_x, -y * big_x], axis=-1)
    rows_y = np.stack([zero, zero, zero, x, y, one, -x * big_y, -y * big_y], axis=-1)
    a = np.concatenate([rows_x, rows_y], axis=-2)                  # (…,8,8)
    b = np.concatenate([big_x, big_y], axis=-1)[..., None]         # (…,8,1)
    h = np.linalg.solve(a, b)[..., 0]
    return np.concatenate([h, np.ones(h.shape[:-1] + (1,))], axis=-1).reshape(h.shape[:-1] + (3, 3))


def normalise(h: np.ndarray, ref_pt) -> np.ndarray:
    """기준 점에서 w = 1 이 되게 행렬의 크기를 맞춘다 (… ,3,3). 이렇게 해 두면 'w > 0 = 카메라 앞' 으로 판단할 수 있다."""
    ref = np.array([float(ref_pt[0]), float(ref_pt[1]), 1.0])
    w = h[..., 2, :] @ ref
    return h / np.where(np.abs(w) < 1e-12, 1e-12, w)[..., None, None]


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """점 (N,2) 에 3×3 변환 (또는 (T,3,3) 여러 개) 을 적용한다. 지평선 너머(w ≤ 0)의 점은 NaN 이 된다.

    행렬은 normalise() 로 맞춘 것이어야 한다 (축척+이동 행렬은 w 가 늘 1 이라 그대로 된다).
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([p, np.ones(len(p))])
    q = hom @ np.swapaxes(h, -1, -2)                               # (N,3) 또는 (T,N,3)
    w = q[..., 2:3]
    ok = w > 1e-9
    return np.where(ok, q[..., :2] / np.where(ok, w, 1.0), np.nan)


@dataclass
class Mapper:
    """그림 한 장의 '픽셀 → 평면 mm' 변환과 그 설명."""
    kind: str
    image_path: Path
    meta_path: Path
    width: int
    height: int
    plane_from_pixel: np.ndarray                       # 3×3. ortho·mosaic 은 축척+이동, photo 는 평면 변환
    frame: str                                         # 평면 좌표계 설명
    marker_size_mm: float | None = None
    marker_corners_photo_px: np.ndarray | None = None  # 원본 사진에서 마커 네 모서리 (오차 예상에 쓴다)
    photo_from_pixel: np.ndarray | None = None         # 이 그림의 픽셀 → 원본 사진 픽셀
    notes: list[str] = field(default_factory=list)

    @property
    def kind_name(self) -> str:
        return KIND_NAMES[self.kind]

    def to_mm(self, pts_px) -> np.ndarray:
        """픽셀 (N,2) → 평면 mm (N,2)."""
        return apply_h(self.plane_from_pixel, pts_px)

    def inside(self, p_px) -> bool:
        return -0.5 <= p_px[0] <= self.width - 0.5 and -0.5 <= p_px[1] <= self.height - 0.5

    def measure(self, a_px, b_px, corner_noise_px: float = 0.3) -> dict:
        """두 픽셀 사이의 실제 거리. 계산할 수 없는 점(사진의 지평선 너머)이면 CoordError."""
        mm = self.to_mm([a_px, b_px])
        if not np.all(np.isfinite(mm)):
            raise CoordError("이 점은 평면 위의 위치로 바꿀 수 없습니다 (사진에서 지평선 너머).")
        d = mm[1] - mm[0]
        return {"a_mm": (float(mm[0, 0]), float(mm[0, 1])), "b_mm": (float(mm[1, 0]), float(mm[1, 1])),
                "dx_mm": float(d[0]), "dy_mm": float(d[1]), "length_mm": float(math.hypot(d[0], d[1])),
                "sigma_mm": self.length_sigma(a_px, b_px, corner_noise_px)}

    def local_mm_per_pixel(self, p_px) -> float | None:
        """이 자리에서 픽셀 하나가 몇 mm 인지 (원본 사진에서는 자리마다 다르다)."""
        p = np.asarray(p_px, dtype=np.float64)
        mm = self.to_mm([p, p + [1.0, 0.0], p + [0.0, 1.0]])
        if not np.all(np.isfinite(mm)):
            return None
        return float(max(np.linalg.norm(mm[1] - mm[0]), np.linalg.norm(mm[2] - mm[0])))

    def length_sigma(self, a_px, b_px, corner_noise_px: float = 0.3, trials: int = 300) -> float | None:
        """마커 모서리를 corner_noise_px 만큼 틀리게 읽었다고 가정했을 때, 이 길이가 흔들리는 정도 (표준편차, mm).

        마커 1개로 편 그림은 마커에서 멀수록, 길이가 길수록 이 값이 커진다. 렌즈 왜곡이나 바닥 굴곡, 마커 크기 오차는
        들어 있지 않으므로 실제 오차는 이보다 클 수 있다. 계산에 필요한 정보가 없으면(합친 그림) None.
        """
        if self.marker_corners_photo_px is None or self.photo_from_pixel is None or not self.marker_size_mm:
            return None
        photo = apply_h(self.photo_from_pixel, [a_px, b_px])
        if not np.all(np.isfinite(photo)):
            return None
        s = float(self.marker_size_mm)
        plane = np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]])
        rng = np.random.default_rng(0)                 # 같은 선은 늘 같은 값이 나오게 고정
        noisy = self.marker_corners_photo_px[None] + rng.normal(0.0, corner_noise_px, (trials, 4, 2))
        try:
            hs = normalise(homography_from_points(noisy, plane), self.marker_corners_photo_px.mean(axis=0))
        except np.linalg.LinAlgError:
            return None
        mm = apply_h(hs, photo)                        # (trials, 2, 2)
        length = np.linalg.norm(mm[:, 1] - mm[:, 0], axis=1)
        length = length[np.isfinite(length)]
        return float(np.std(length)) if len(length) > 10 else None


def _scale_matrix(mm_per_pixel: float, origin) -> np.ndarray:
    return np.array([[mm_per_pixel, 0.0, float(origin[0])], [0.0, mm_per_pixel, float(origin[1])], [0.0, 0.0, 1.0]])


def _from_stage1(d: dict, meta_path: Path, image_path: Path, size: tuple[int, int]) -> Mapper:
    """1단계 homography.json 으로 변환을 만든다. 그림이 정사영 그림인지 원본 사진인지는 이름과 크기로 가린다."""
    out, src, marker = d.get("output") or {}, d.get("source_image") or {}, d.get("marker") or {}
    name = image_path.name.lower()
    if name == "comparison.png":
        raise CoordError("comparison.png 는 줄여서 나란히 붙인 그림이라 좌표 정보가 없습니다. "
                         "orthorectified.png 나 aruco_detected.png 를 여세요.")
    ortho_size = (out.get("width_px"), out.get("height_px"))
    photo_size = (src.get("width_px"), src.get("height_px"))
    is_ortho, is_photo = size == ortho_size, size == photo_size
    if is_ortho and is_photo:                          # 두 그림의 크기가 우연히 같으면 이름으로 가린다
        is_ortho, is_photo = name == str(out.get("file", "orthorectified.png")).lower(), name == "aruco_detected.png"
        if is_ortho == is_photo:
            raise CoordError("정사영 그림과 원본 사진의 크기가 같아서 어느 쪽인지 알 수 없습니다. 1단계가 저장한 파일 이름 그대로 여세요.")
    if not (is_ortho or is_photo):
        raise CoordError(f"{meta_path.name} 의 정보와 그림 크기가 맞지 않습니다 (그림 {size[0]}×{size[1]}, "
                         f"정사영 {ortho_size[0]}×{ortho_size[1]}, 원본 사진 {photo_size[0]}×{photo_size[1]}).")
    notes = []
    size_mm = marker.get("size_mm")
    if size_mm:
        notes.append(f"마커 한 변을 {size_mm:g} mm 로 보고 계산한 치수입니다. 실제 크기가 다르면 그 비율만큼 틀립니다.")
    if not src.get("lens_undistorted", False):
        notes.append("렌즈 왜곡을 보정하지 않은 사진입니다.")
    notes.append("마커와 같은 평면에 있는 점만 맞습니다. 마커에서 멀수록 오차가 커집니다.")
    corners = np.array(marker["corners_image_px"], dtype=np.float64) if "corners_image_px" in marker else None
    if corners is not None and corners.shape != (4, 2):
        corners = None
    frame = "마커 평면 (원점 = 마커 좌상 모서리, x = 좌상→우상, y = 좌상→좌하)"
    if is_ortho:
        for key in ("mm_per_pixel", "origin_mm"):
            if key not in out:
                raise CoordError(f"{meta_path.name} 의 output 에 {key} 가 없어 픽셀을 mm 로 바꿀 수 없습니다.")
        photo_from_pixel = None
        if "homography_image_px_to_output_px" in d and "corners_output_px" in marker:
            inv = np.linalg.inv(np.array(d["homography_image_px_to_output_px"], dtype=np.float64))
            photo_from_pixel = normalise(inv, np.array(marker["corners_output_px"], dtype=np.float64).mean(axis=0))
        if out.get("cropped"):
            notes.append("1단계에서 출력 한도 때문에 범위를 잘라낸 그림입니다.")
        return Mapper("ortho", image_path, meta_path, size[0], size[1], _scale_matrix(float(out["mm_per_pixel"]), out["origin_mm"]),
                      frame, size_mm, corners, photo_from_pixel, notes)
    if "homography_image_px_to_plane_mm" not in d:
        raise CoordError(f"{meta_path.name} 에 homography_image_px_to_plane_mm 이 없어 원본 사진의 픽셀을 mm 로 바꿀 수 없습니다.")
    h = np.array(d["homography_image_px_to_plane_mm"], dtype=np.float64)
    if corners is not None:
        h = normalise(h, corners.mean(axis=0))        # 마커 자리에서 w = 1 (1단계도 이렇게 저장하지만 한 번 더 맞춘다)
    return Mapper("photo", image_path, meta_path, size[0], size[1], h, frame, size_mm, corners, np.eye(3), notes)


def _from_stage2(d: dict, meta_path: Path, image_path: Path, size: tuple[int, int]) -> Mapper:
    """2단계 registration.json 의 mosaic 항목으로 변환을 만든다."""
    name = image_path.name.lower()
    if name.startswith("registration_overlay"):
        raise CoordError("겹쳐 보기 그림(registration_overlay)의 좌표 정보는 저장돼 있지 않습니다. registered_mosaic.png 를 여세요.")
    mos = d.get("mosaic")
    if not isinstance(mos, dict):
        raise CoordError(f"{meta_path.name} 에 합친 그림(mosaic) 정보가 없습니다. 정합이 실패한 결과일 수 있습니다.")
    if size != (mos.get("width_px"), mos.get("height_px")):
        raise CoordError(f"{meta_path.name} 의 합친 그림 크기({mos.get('width_px')}×{mos.get('height_px')})와 "
                         f"연 그림({size[0]}×{size[1]})이 다릅니다. registered_mosaic.png 를 여세요.")
    for key in ("mm_per_pixel", "origin_mm"):
        if key not in mos:
            raise CoordError(f"{meta_path.name} 의 mosaic 에 {key} 가 없어 픽셀을 mm 로 바꿀 수 없습니다.")
    ref = d.get("reference_image", "기준 그림")
    notes = [f"{len(mos.get('images', []))}장을 합친 그림입니다. 좌표는 기준 그림({ref})의 마커 평면입니다."]
    if mos.get("coarsened_for_size"):
        notes.append("크기 한도 때문에 축척을 거칠게 한 그림입니다 (픽셀 하나가 원래보다 큽니다).")
    notes.append("그림 사이 이음매를 건너 잰 치수에는 정합 오차가 더해집니다.")
    return Mapper("mosaic", image_path, meta_path, size[0], size[1], _scale_matrix(float(mos["mm_per_pixel"]), mos["origin_mm"]),
                  f"기준 그림({ref})의 마커 평면 (원점 = 그 마커의 좌상 모서리)", None, None, None, notes)


def read_mapper(meta_path: Path, image_path: Path, size: tuple[int, int]) -> Mapper:
    """JSON 한 개를 읽어 이 그림의 변환을 만든다. 어느 단계의 파일인지는 내용으로 가린다."""
    meta_path, image_path = Path(meta_path), Path(image_path)
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        raise CoordError(f"{meta_path.name} 을 읽지 못했습니다 ({e}).") from e
    if not isinstance(d, dict):
        raise CoordError(f"{meta_path.name} 은 좌표 정보 파일이 아닙니다.")
    try:
        if d.get("module") == "02_registration" or "mosaic" in d:
            return _from_stage2(d, meta_path, image_path, tuple(size))
        if "output" in d or "homography_image_px_to_plane_mm" in d:
            return _from_stage1(d, meta_path, image_path, tuple(size))
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as e:
        raise CoordError(f"{meta_path.name} 의 값을 읽는 중 문제가 생겼습니다 ({e}).") from e
    raise CoordError(f"{meta_path.name} 은 1단계(homography.json)나 2단계(registration.json)의 파일이 아닙니다.")


def find_mapper(image_path: Path, size: tuple[int, int]) -> Mapper:
    """그림과 같은 폴더에서 좌표 정보 파일을 찾아 변환을 만든다. 못 찾으면 이유를 모아 CoordError 로 알린다."""
    image_path = Path(image_path)
    folder = image_path.parent
    candidates = [folder / "homography.json", folder / "registration.json", image_path.with_suffix(".json")]
    reasons = []
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            return read_mapper(cand, image_path, size)
        except CoordError as e:
            reasons.append(str(e))
    if reasons:
        raise CoordError("\n".join(reasons))
    raise CoordError(f"'{image_path.name}' 옆에서 좌표 정보 파일을 찾지 못했습니다.\n"
                     "같은 폴더에 homography.json (1단계) 또는 registration.json (2단계) 이 있어야 합니다.\n"
                     "축척을 추측해서 재지는 않습니다.")
