"""
깊이 계산 (카메라 없이도 쓸 수 있는 순수 계산).

좌표 약속 (카메라 좌표, 단위 mm)
    원점 = 컬러 카메라의 중심,  X = 사진의 오른쪽,  Y = 사진의 아래쪽,  Z = 카메라가 바라보는 방향.
    여기서 z 는 '카메라에서 그 점까지의 직선 거리' 가 아니라, 카메라가 바라보는 방향으로 잰 거리다
    (카메라 면에서 수직으로 떨어진 거리). RealSense 가 주는 깊이값이 이 z 다.
    깊이가 잡히지 않은 픽셀은 NaN 이다.

    average_depth()      여러 프레임을 픽셀별로 평균 (잡음 줄이기) + 프레임 사이 흔들림(표준편차)
    deproject()          픽셀 (u, v) 와 z → 3차원 점 (X, Y, Z)
    fit_plane()          점들에 평면 맞추기 → 평평한 정도
    region_stats()       고른 영역의 평균 거리, 잡음, 평평한 정도
    height_above_plane() 기준 평면에서 얼마나 솟았는지 (높이를 아는 물체로 정확도 확인)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Intrinsics:
    """카메라 내부 값 (깊이를 컬러에 맞춰 정렬했으므로 컬러 카메라의 값이다)."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "none"                 # 왜곡 모델 이름: "none", "inverse_brown_conrady", "brown_conrady" 등
    coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coeffs"] = list(self.coeffs)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Intrinsics":
        return Intrinsics(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
                          str(d.get("model", "none")), tuple(float(c) for c in d.get("coeffs", (0, 0, 0, 0, 0))))


def average_depth(stack: np.ndarray, min_valid_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """깊이 프레임 여러 장 (N,H,W; 없는 곳 NaN) → (평균 mm, 프레임 사이 표준편차 mm, 깊이가 잡힌 프레임 수).

    N 장 중 min_valid_ratio 이상에서 값이 잡힌 픽셀만 평균을 준다. 나머지는 NaN.
    """
    stack = np.asarray(stack, dtype=np.float32)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0, dtype=np.float64)
    enough = count >= max(1, math.ceil(min_valid_ratio * stack.shape[0]))
    mean = np.where(enough, total / np.maximum(count, 1), np.nan)
    sq = np.where(valid, (stack - mean[None]) ** 2, 0.0).sum(axis=0, dtype=np.float64)
    std = np.where(enough & (count > 1), np.sqrt(sq / np.maximum(count - 1, 1)), np.nan)
    return mean.astype(np.float32), std.astype(np.float32), count.astype(np.int32)


def _undistorted_rays(intr: Intrinsics, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """픽셀 → z = 1 평면 위의 (x, y). RealSense SDK 의 rs2_deproject_pixel_to_point 와 같은 식."""
    x = (u - intr.cx) / intr.fx
    y = (v - intr.cy) / intr.fy
    k1, k2, p1, p2, k3 = (list(intr.coeffs) + [0.0] * 5)[:5]
    if not any((k1, k2, p1, p2, k3)):
        return x, y
    model = intr.model.lower()
    if "inverse" in model:
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    if "brown" in model:
        # 왜곡을 넣는 식의 역은 한 번에 풀리지 않아 되풀이해서 푼다
        xo, yo = x.copy(), y.copy()
        for _ in range(10):
            r2 = x * x + y * y
            icdist = 1.0 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
            x, y = (xo - dx) * icdist, (yo - dy) * icdist
        return x, y
    return x, y        # 그 밖의 모델은 왜곡을 무시한다 (D400 계열 컬러 카메라에서는 나오지 않는다)


def deproject(intr: Intrinsics, depth_mm: np.ndarray) -> np.ndarray:
    """깊이 지도 (H,W) → 픽셀마다의 3차원 점 (H,W,3) mm. 깊이가 없는 곳은 NaN."""
    h, w = depth_mm.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x, y = _undistorted_rays(intr, u, v)
    z = depth_mm.astype(np.float64)
    return np.dstack([x * z, y * z, z])


def deproject_pixel(intr: Intrinsics, u: float, v: float, z_mm: float) -> np.ndarray:
    """픽셀 하나 → 3차원 점 (mm)."""
    x, y = _undistorted_rays(intr, np.array([float(u)]), np.array([float(v)]))
    return np.array([x[0] * z_mm, y[0] * z_mm, z_mm])


@dataclass
class Plane:
    normal: np.ndarray      # 길이 1, 카메라 쪽을 향한다 (Z 성분이 음수)
    point: np.ndarray       # 평면 위의 한 점 (맞춘 점들의 무게중심)
    rms_mm: float           # 점들이 평면에서 벗어난 정도 (제곱평균의 제곱근)
    max_mm: float           # 가장 많이 벗어난 점
    n: int                  # 맞추는 데 쓴 점의 수

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """평면에서의 거리 (mm). + = 평면보다 카메라에 가까움 (아래를 보고 찍었다면 위로 솟음)."""
        return (np.asarray(pts, dtype=np.float64) - self.point) @ self.normal

    @property
    def tilt_deg(self) -> float:
        """카메라가 바라보는 방향과 평면의 수직 방향 사이의 각 (0 = 정면으로 마주봄)."""
        return math.degrees(math.acos(min(1.0, abs(float(self.normal[2])))))


def fit_plane(points: np.ndarray, reject_sigma: float = 3.0) -> Plane | None:
    """점들 (M,3) 에 평면을 맞춘다. 한 번 맞춘 뒤 많이 벗어난 점(3σ)을 빼고 다시 맞춘다. 점이 3개 미만이면 None."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 3:
        return None

    def once(q):
        c = q.mean(axis=0)
        normal = np.linalg.svd(q - c, full_matrices=False)[2][2]
        if normal[2] > 0:
            normal = -normal
        return c, normal

    c, normal = once(p)
    d = (p - c) @ normal
    keep = np.abs(d) <= reject_sigma * max(float(d.std()), 1e-9)
    if 3 <= keep.sum() < len(p):
        c, normal = once(p[keep])
        d = (p[keep] - c) @ normal
    return Plane(normal, c, float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max()), int(len(d)))


def region_points(xyz: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, int]:
    """사각 영역 (x0, y0, x1, y1; 끝은 포함하지 않음) 의 점들 (M,3) 과 그 영역의 전체 픽셀 수."""
    x0, y0, x1, y1 = box
    part = xyz[y0:y1, x0:x1].reshape(-1, 3)
    return part[np.all(np.isfinite(part), axis=1)], part.shape[0]


def region_stats(xyz: np.ndarray, box: tuple[int, int, int, int], noise_mm: np.ndarray | None = None) -> dict | None:
    """고른 영역의 통계. 깊이가 잡힌 점이 3개 미만이면 None.

    z_mean / z_median : 평균·중간 거리 (mm)
    plane_rms / max   : 평면을 맞췄을 때 벗어난 정도. 평평한 면을 찍었다면 이 값이 곧 깊이 오차의 크기다
    tilt_deg          : 그 면이 카메라와 얼마나 비스듬한지
    temporal_noise    : 같은 픽셀이 프레임마다 흔들린 정도의 중간값 (촬영 때 저장한 값이 있을 때)
    """
    pts, total = region_points(xyz, box)
    plane = fit_plane(pts)
    if plane is None:
        return None
    out = {"pixels": total, "valid": int(len(pts)), "valid_percent": 100.0 * len(pts) / max(total, 1),
           "z_mean_mm": float(pts[:, 2].mean()), "z_median_mm": float(np.median(pts[:, 2])),
           "z_min_mm": float(pts[:, 2].min()), "z_max_mm": float(pts[:, 2].max()),
           "plane_rms_mm": plane.rms_mm, "plane_max_mm": plane.max_mm, "tilt_deg": plane.tilt_deg}
    if noise_mm is not None:
        x0, y0, x1, y1 = box
        part = noise_mm[y0:y1, x0:x1]
        part = part[np.isfinite(part)]
        out["temporal_noise_mm"] = float(np.median(part)) if part.size else None
    return out


def height_above_plane(plane: Plane, points: np.ndarray) -> dict | None:
    """점들이 기준 평면에서 얼마나 솟았는지 (mm). + = 카메라 쪽. 높이를 아는 물체를 올려놓고 찍어 비교한다."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) == 0:
        return None
    d = plane.signed_distance(p)
    return {"n": int(len(d)), "mean_mm": float(d.mean()), "median_mm": float(np.median(d)), "std_mm": float(d.std())}


def save_ply(path, xyz: np.ndarray, color_bgr: np.ndarray) -> int:
    """깊이가 잡힌 점만 색과 함께 PLY 점군 파일로 저장한다 (단위 mm). 저장한 점의 수를 돌려준다."""
    ok = np.all(np.isfinite(xyz), axis=2)
    pts = xyz[ok].astype("<f4")
    rgb = color_bgr[ok][:, ::-1].astype(np.uint8)
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\ncomment unit mm, camera frame (X right, Y down, Z forward)\n"
              f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    return int(len(pts))
