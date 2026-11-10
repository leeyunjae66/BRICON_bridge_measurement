"""
특징점 찾기와 짝짓기.

    detect()  : 그림 한 장에서 특징점과 설명값(descriptor)을 뽑는다. 가림 그림 밖에서는 찾지 않는다.
    match()   : 두 그림의 특징점을 설명값이 닮은 것끼리 짝짓고, 비율 검사(와 서로 고르기)로 애매한 짝을 버린다.

여기서 나오는 좌표는 모두 '정사영 그림의 픽셀' 이다 (줄여서 찾았더라도 원래 크기의 좌표로 되돌려 준다).
mm 로 바꾸는 일과 기하 검사(RANSAC)는 registration.py / rigid_estimator.py 가 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


class FeatureError(Exception):
    """특징점을 찾거나 짝지을 수 없을 때."""


@dataclass
class Features:
    points_px: np.ndarray        # (N,2) 정사영 그림 픽셀 좌표 (원래 크기 기준)
    descriptors: np.ndarray | None
    detector: str                # 실제로 쓴 방법 ("SIFT" 또는 "ORB")
    scale: float                 # 특징점을 찾을 때 그림을 줄인 배율 (1 = 줄이지 않음)

    @property
    def count(self) -> int:
        return len(self.points_px)


@dataclass
class Matches:
    idx_a: np.ndarray            # 짝마다 그림 A 의 특징점 번호
    idx_b: np.ndarray            # 짝마다 그림 B 의 특징점 번호
    n_initial: int               # 처음에 후보 짝을 얻은 특징점 수 (비율 검사 전)
    n_ratio: int                 # 비율 검사를 통과한 수 (A→B 방향)
    n_mutual: int                # 서로 고르기까지 통과한 수 (= 최종 짝 수)

    @property
    def count(self) -> int:
        return len(self.idx_a)


def create_detector(cfg: Config):
    """설정에 맞는 검출기를 만든다 → (검출기, 실제 이름). SIFT 가 없는 OpenCV 면 ORB 로 바꾼다."""
    name = cfg.detector.upper()
    if name == "SIFT":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=cfg.max_features, contrastThreshold=cfg.sift_contrast_threshold), "SIFT"
        name = "ORB"
    if name == "ORB":
        return cv2.ORB_create(nfeatures=cfg.max_features, scaleFactor=1.2, nlevels=8), "ORB"
    raise FeatureError(f"알 수 없는 특징점 방법입니다: {cfg.detector} (SIFT 또는 ORB)")


def detect(image: np.ndarray, mask: np.ndarray, cfg: Config) -> tuple[Features, np.ndarray]:
    """특징점을 찾는다 → (특징점, 실제로 쓴 가림 그림(줄인 크기)).

    mask: 원래 크기의 가림 그림 (255 = 찾는 곳). 경계 근처는 border_margin_px 만큼 더 깎아 낸다.
    """
    detector, used = create_detector(cfg)
    h, w = image.shape[:2]
    scale = min(1.0, cfg.feature_max_side_px / float(max(h, w)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_mask = mask
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if cfg.use_clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if cfg.border_margin_px > 0:
        k = 2 * int(cfg.border_margin_px) + 1
        small_mask = cv2.erode(small_mask, np.ones((k, k), np.uint8))
    if cv2.countNonZero(small_mask) == 0:
        return Features(np.zeros((0, 2)), None, used, scale), small_mask
    keypoints, desc = detector.detectAndCompute(gray, small_mask)
    if not keypoints or desc is None:
        return Features(np.zeros((0, 2)), None, used, scale), small_mask
    pts = np.array([k.pt for k in keypoints], dtype=np.float64)
    pts = (pts + 0.5) / scale - 0.5            # 줄인 그림의 좌표 → 원래 크기의 좌표 (픽셀 가운데 기준)
    return Features(pts, desc, used, scale), small_mask


def _knn_ratio(desc_q: np.ndarray, desc_t: np.ndarray, norm: int, ratio: float) -> tuple[dict[int, int], int]:
    """desc_q 의 각 특징점에 대해 desc_t 에서 가장 닮은 둘을 찾고 비율 검사를 한다 → ({q번호: t번호}, 후보를 얻은 수)."""
    knn = cv2.BFMatcher(norm).knnMatch(desc_q, desc_t, k=2)
    passed: dict[int, int] = {}
    n_initial = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        n_initial += 1
        best, second = pair
        if best.distance < ratio * second.distance:
            passed[best.queryIdx] = best.trainIdx
    return passed, n_initial


def match(a: Features, b: Features, cfg: Config) -> Matches:
    """두 그림의 특징점을 짝짓는다. 실패 원인은 FeatureError 의 글로 알린다."""
    if a.detector != b.detector:
        raise FeatureError("두 그림의 특징점을 서로 다른 방법으로 뽑았습니다.")
    for name, f in (("첫째", a), ("둘째", b)):
        if f.descriptors is None or f.count < 2:
            raise FeatureError(f"{name} 그림에서 특징점을 거의 찾지 못했습니다 ({f.count}개). "
                               "그림이 너무 밋밋하거나 흐리거나, 쓸 수 있는 영역이 너무 작습니다.")
    norm = cv2.NORM_L2 if a.detector == "SIFT" else cv2.NORM_HAMMING
    forward, n_initial = _knn_ratio(a.descriptors, b.descriptors, norm, cfg.ratio_test)
    pairs = list(forward.items())
    n_ratio = len(pairs)
    if cfg.mutual_check and pairs:
        backward, _ = _knn_ratio(b.descriptors, a.descriptors, norm, cfg.ratio_test)
        pairs = [(i, j) for i, j in pairs if backward.get(j) == i]
    idx_a = np.array([p[0] for p in pairs], dtype=int)
    idx_b = np.array([p[1] for p in pairs], dtype=int)
    return Matches(idx_a, idx_b, n_initial, n_ratio, len(pairs))
