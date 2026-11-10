"""
강체 변환(회전 + 이동, 축척 고정 1) 추정.

짝지은 점 p (움직일 그림, mm) 와 q (고정 그림, mm) 로  q ≈ T · p  인 3×3 행렬 T 를 구한다.

    fit_rigid()        : 최소제곱으로 회전과 이동을 구한다 (축척은 1 로 고정).
    ransac_rigid()     : 틀린 짝이 섞여 있어도 되도록, 두 점씩 뽑아 만든 후보 중 맞는 짝이 가장 많은 것을 고른 뒤 다듬는다.
    similarity_scale() : 진단용. 축척까지 자유롭게 풀었을 때의 축척 (1 에서 벗어나면 두 정사영의 축척이 서로 다르다는 뜻).

강체와 닮음(Similarity)의 차이
    강체  : 회전 1 + 이동 2 = 미지수 3개. 길이를 바꾸지 않는다. 치수 측정에는 이쪽이 맞다.
    닮음  : 여기에 축척 1개가 더 있다. 두 그림의 축척이 조금 달라도 겹쳐 보이게 만들지만, 그만큼 길이를 바꿔 버린다.
그래서 최종 변환은 강체로 구하고, 닮음의 축척은 '정사영이 잘 됐는지' 를 보는 숫자로만 쓴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def make_rigid(angle_rad: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def decompose(t: np.ndarray) -> tuple[float, float, float]:
    """3×3 강체 행렬 → (회전각 도, x 이동 mm, y 이동 mm). 회전은 x 축에서 y 축 쪽으로 도는 방향이 +."""
    return math.degrees(math.atan2(t[1, 0], t[0, 0])), float(t[0, 2]), float(t[1, 2])


def apply(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """점 (N,2) 에 3×3 행렬을 적용한다 (마지막 줄이 [0,0,1] 인 변환)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return p @ t[:2, :2].T + t[:2, 2]


def invert(t: np.ndarray) -> np.ndarray:
    """강체 행렬의 역행렬 (A_from_B → B_from_A)."""
    r = t[:2, :2]
    out = np.eye(3)
    out[:2, :2] = r.T
    out[:2, 2] = -r.T @ t[:2, 2]
    return out


def fit_rigid(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """q ≈ T · p 인 강체 변환을 최소제곱으로 구한다 (점이 2개 이상 필요)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    a, b = p - pc, q - qc
    angle = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    c, s = math.cos(angle), math.sin(angle)
    r = np.array([[c, -s], [s, c]])
    t = qc - r @ pc
    return make_rigid(angle, float(t[0]), float(t[1]))


def residuals(t: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """짝마다 어긋난 거리 (mm)."""
    return np.linalg.norm(apply(t, p) - np.asarray(q, dtype=np.float64), axis=1)


@dataclass
class RigidResult:
    t: np.ndarray                # q ≈ t · p
    inliers: np.ndarray          # (N,) 참/거짓
    residual_mm: np.ndarray      # (N,) 모든 짝의 어긋남
    hypotheses: int              # 실제로 따져 본 후보 수

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())

    def stats(self) -> dict:
        r = self.residual_mm[self.inliers]
        if len(r) == 0:
            return {"mean_mm": None, "rmse_mm": None, "max_mm": None}
        return {"mean_mm": float(r.mean()), "rmse_mm": float(np.sqrt(np.mean(r ** 2))), "max_mm": float(r.max())}

    def spread_mm(self, q: np.ndarray) -> float:
        """맞는 짝들이 퍼져 있는 범위 (고정 그림 좌표에서, 둘러싼 사각형의 대각선 길이)."""
        pts = np.asarray(q)[self.inliers]
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def ransac_rigid(p: np.ndarray, q: np.ndarray, threshold_mm: float, iterations: int = 3000, seed: int = 0) -> RigidResult | None:
    """틀린 짝이 섞인 대응점에서 강체 변환을 구한다. 후보를 하나도 만들 수 없으면 None.

    두 짝을 뽑으면 강체 변환이 하나 정해진다. 뽑은 두 점 사이 거리가 두 그림에서 다르면 (축척 1 에 어긋나므로) 버린다.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, iterations)
    j = rng.integers(0, n, iterations)
    dp, dq = p[j] - p[i], q[j] - q[i]
    lp, lq = np.hypot(dp[:, 0], dp[:, 1]), np.hypot(dq[:, 0], dq[:, 1])
    min_base = max(20.0, 4.0 * threshold_mm)        # 너무 가까운 두 점으로는 회전을 정할 수 없다
    ok = (i != j) & (lp > min_base) & (np.abs(lp - lq) < 2.0 * threshold_mm)
    if not ok.any():
        return None
    i, j, dp, dq = i[ok], j[ok], dp[ok], dq[ok]
    ang = np.arctan2(dq[:, 1], dq[:, 0]) - np.arctan2(dp[:, 1], dp[:, 0])
    c, s = np.cos(ang), np.sin(ang)
    mp, mq = 0.5 * (p[i] + p[j]), 0.5 * (q[i] + q[j])
    tx = mq[:, 0] - (c * mp[:, 0] - s * mp[:, 1])
    ty = mq[:, 1] - (s * mp[:, 0] + c * mp[:, 1])
    thr2 = threshold_mm ** 2
    best, best_count, best_cost = -1, -1, np.inf
    for k0 in range(0, len(ang), 256):
        k1 = min(len(ang), k0 + 256)
        rx = c[k0:k1, None] * p[None, :, 0] - s[k0:k1, None] * p[None, :, 1] + tx[k0:k1, None] - q[None, :, 0]
        ry = s[k0:k1, None] * p[None, :, 0] + c[k0:k1, None] * p[None, :, 1] + ty[k0:k1, None] - q[None, :, 1]
        d2 = rx * rx + ry * ry
        inl = d2 < thr2
        count = inl.sum(axis=1)
        cost = np.where(inl, d2, thr2).sum(axis=1)   # 맞는 짝 수가 같으면 더 잘 맞는 쪽
        k = int(np.lexsort((cost, -count))[0])
        if count[k] > best_count or (count[k] == best_count and cost[k] < best_cost):
            best, best_count, best_cost = k0 + k, int(count[k]), float(cost[k])
    t = make_rigid(float(ang[best]), float(tx[best]), float(ty[best]))
    inliers = residuals(t, p, q) < threshold_mm
    # 맞는 짝 전체로 다시 맞추고, 맞는 짝 목록이 더 바뀌지 않을 때까지 되풀이한다
    for _ in range(10):
        if inliers.sum() < 2:
            break
        t_new = fit_rigid(p[inliers], q[inliers])
        new = residuals(t_new, p, q) < threshold_mm
        if new.sum() < inliers.sum():
            break
        t = t_new
        if np.array_equal(new, inliers):
            break
        inliers = new
    return RigidResult(t, inliers, residuals(t, p, q), int(len(ang)))


def similarity_scale(p: np.ndarray, q: np.ndarray, threshold_mm: float) -> tuple[float, int] | None:
    """진단용: 축척까지 풀었을 때의 (축척, 맞는 짝 수). q ≈ 축척 · 회전 · p + 이동.

    OpenCV 의 estimateAffinePartial2D 는 축척을 함께 구하므로 여기서만 쓰고, 최종 변환에는 쓰지 않는다.
    """
    p = np.asarray(p, dtype=np.float32).reshape(-1, 1, 2)
    q = np.asarray(q, dtype=np.float32).reshape(-1, 1, 2)
    if len(p) < 3:
        return None
    try:
        m, inl = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=float(threshold_mm),
                                             maxIters=5000, confidence=0.999, refineIters=10)
    except cv2.error:
        return None
    if m is None or inl is None:
        return None
    return float(math.hypot(m[0, 0], m[1, 0])), int(inl.sum())
