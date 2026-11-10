"""
ArUco 마커 찾기.

    1. 사진을 줄여서 마커를 찾는다 (고해상도 사진에서도 빠르고 잘 찾힘). 못 찾으면 다른 크기로 다시 시도.
    2. 번호가 맞는 마커를 고른다.
    3. 네 모서리를 원본 해상도에서 다시 정밀하게 다듬는다 (Subpixel Refinement).

모서리 순서는 OpenCV 가 주는 그대로다: 마커 자체의 좌상 → 우상 → 우하 → 좌하 (시계 방향).
마커를 돌려 놓았으면 '사진에서의 왼쪽 위' 가 아니라 '마커 무늬 기준 왼쪽 위' 가 첫 번째다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import Config

CORNER_NAMES = ("좌상", "우상", "우하", "좌하")


@dataclass
class Detection:
    found: bool
    message: str = ""
    corners: np.ndarray | None = None        # (4,2) 원본 사진 픽셀 좌표: 좌상, 우상, 우하, 좌하
    marker_id: int | None = None
    all_ids: list[int] = field(default_factory=list)   # 사진에서 찾은 모든 마커 번호
    side_px: float = 0.0                     # 사진에서 마커 한 변의 평균 길이 (픽셀)
    warnings: list[str] = field(default_factory=list)


def _dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV 에 aruco 모듈이 없습니다. opencv-contrib-python 을 설치하세요.")
    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"알 수 없는 마커 사전 이름입니다: {name}")
    code = getattr(cv2.aruco, name)
    # OpenCV 4.7 이상과 그 이전의 함수 이름이 다르다
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(code)
    return cv2.aruco.Dictionary_get(code)


def _parameters():
    try:
        p = cv2.aruco.DetectorParameters()            # OpenCV 4.7 이상
    except (AttributeError, TypeError):
        p = cv2.aruco.DetectorParameters_create()     # 그 이전
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # 비스듬히 찍혀 찌그러진 마커, 작게 찍힌 마커도 잡도록 조건을 조금 넓힌다
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.minMarkerPerimeterRate = 0.02
    p.polygonalApproxAccuracyRate = 0.05
    p.perspectiveRemovePixelPerCell = 8
    return p


def _detect_once(gray: np.ndarray, dictionary, params):
    """OpenCV 버전에 맞는 방법으로 한 번 찾는다 → (모서리 목록, 번호 목록)."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        return [], []
    return [c.reshape(4, 2).astype(np.float64) for c in corners], [int(i) for i in ids.ravel()]


def _quad_area(c: np.ndarray) -> float:
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _refine(gray: np.ndarray, corners: np.ndarray, cfg: Config) -> tuple[np.ndarray, bool]:
    """원본 해상도에서 모서리 위치를 소수점 픽셀까지 다듬는다. 실패하면 처음 값을 그대로 쓴다."""
    side = float(np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
    # 살펴볼 창의 반쪽 크기: 마커 한 칸(변의 1/6)보다 작아야 옆 칸 무늬에 끌려가지 않는다
    win = int(np.clip(side / 30.0, 3, 25))
    h, w = gray.shape[:2]
    if np.any(corners[:, 0] < win + 2) or np.any(corners[:, 1] < win + 2) or np.any(corners[:, 0] > w - win - 3) or np.any(corners[:, 1] > h - win - 3):
        return corners, False                      # 모서리가 사진 가장자리에 붙어 창을 둘 수 없음
    pts = corners.reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cfg.subpixel_max_iter, cfg.subpixel_epsilon)
    try:
        refined = cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria).reshape(4, 2).astype(np.float64)
    except cv2.error:
        return corners, False
    if np.max(np.linalg.norm(refined - corners, axis=1)) > win:
        return corners, False                      # 너무 멀리 움직였으면 엉뚱한 곳에 붙은 것
    return refined, True


def detect_marker(bgr: np.ndarray, cfg: Config) -> Detection:
    """사진에서 cfg.marker_id 번 마커를 찾아 네 모서리를 돌려준다. 못 찾아도 예외를 던지지 않는다."""
    try:
        dictionary = _dictionary(cfg.aruco_dictionary)
        params = _parameters()
    except RuntimeError as e:
        return Detection(False, str(e))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    long_side = max(h, w)

    # 줄인 크기 → 원본 크기 → 더 줄인 크기 → 조금 큰 크기 순서로 시도한다
    sizes = []
    for target in (cfg.detect_long_side_px, long_side, cfg.detect_long_side_px // 2, cfg.detect_long_side_px * 2):
        t = min(int(target), long_side)
        if t >= 200 and t not in sizes:
            sizes.append(t)

    seen: set[int] = set()
    best = None
    for t in sizes:
        scale = t / long_side
        small = gray if scale >= 1.0 else cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        try:
            corners, ids = _detect_once(small, dictionary, params)
        except cv2.error as e:
            return Detection(False, f"마커 검출 중 OpenCV 오류: {e}")
        seen.update(ids)
        cands = [c for c, i in zip(corners, ids) if i == cfg.marker_id]
        if cands:
            c = max(cands, key=_quad_area)
            # 줄인 그림의 좌표를 원본 좌표로 (픽셀 가운데 기준)
            best = ((c + 0.5) / scale - 0.5, len(cands))
            break

    if best is None:
        others = f" 사진에서 찾은 다른 번호: {sorted(seen)}." if seen else ""
        return Detection(False, f"ID {cfg.marker_id} 마커({cfg.aruco_dictionary})를 찾지 못했습니다.{others}\n"
                                "  - 마커 전체와 둘레의 흰 여백이 사진 안에 다 들어왔는지\n"
                                "  - 빛 반사·그림자·흐림이 없는지, 마커가 너무 작게 찍히지 않았는지 확인하세요.",
                         all_ids=sorted(seen))

    corners, count = best
    det = Detection(True, corners=corners, marker_id=cfg.marker_id, all_ids=sorted(seen))
    if count > 1:
        det.warnings.append(f"ID {cfg.marker_id} 마커가 {count}개 보여서 가장 크게 찍힌 것을 썼습니다.")
    refined, ok = _refine(gray, corners, cfg)
    det.corners = refined
    if not ok:
        det.warnings.append("모서리 정밀 다듬기를 하지 못해 처음 찾은 위치를 그대로 썼습니다 (마커가 사진 가장자리에 너무 가깝거나 흐림).")
    det.side_px = float(np.mean(np.linalg.norm(refined - np.roll(refined, -1, axis=0), axis=1)))
    if det.side_px < 60:
        det.warnings.append(f"마커가 작게 찍혔습니다 (한 변 약 {det.side_px:.0f} 픽셀). 마커에서 먼 곳의 오차가 커집니다.")
    return det


def draw_detection(bgr: np.ndarray, det: Detection) -> np.ndarray:
    """원본 사진에 마커 외곽선과 네 모서리(0 = 좌상, 1 = 우상, 2 = 우하, 3 = 좌하)를 그린다."""
    out = bgr.copy()
    if not det.found or det.corners is None:
        return out
    t = max(2, max(out.shape[:2]) // 500)          # 사진이 클수록 선도 굵게
    pts = np.round(det.corners).astype(np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), t, cv2.LINE_AA)
    colors = ((0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 0, 255))   # 빨강, 주황, 파랑, 분홍
    for i, (p, col) in enumerate(zip(pts, colors)):
        cv2.circle(out, (int(p[0]), int(p[1])), t * 3, col, -1, cv2.LINE_AA)
        cv2.putText(out, str(i), (int(p[0]) + t * 4, int(p[1]) - t * 4), cv2.FONT_HERSHEY_SIMPLEX, t * 0.5, (0, 0, 0), t * 3, cv2.LINE_AA)
        cv2.putText(out, str(i), (int(p[0]) + t * 4, int(p[1]) - t * 4), cv2.FONT_HERSHEY_SIMPLEX, t * 0.5, col, t, cv2.LINE_AA)
    # 번호는 마커 무늬를 가리지 않게 마커 아래쪽에 쓴다
    text = f"ID {det.marker_id}"
    org = (int(pts[:, 0].min()), min(out.shape[0] - t * 4, int(pts[:, 1].max()) + t * 24))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (0, 0, 0), t * 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (0, 255, 255), t, cv2.LINE_AA)
    return out
