"""
정합 모듈 설정값.

여기 있는 값이 기본값이고, 자주 바꾸는 값은 실행할 때 옵션으로도 바꿀 수 있다 (python main.py --help).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # ---------------------------------------------------------------- 특징점
    detector: str = "SIFT"              # "SIFT" (기본) 또는 "ORB"
    max_features: int = 8000            # 그림 한 장에서 뽑는 특징점 최대 개수
    feature_max_side_px: int = 2400     # 특징점을 찾을 때 그림의 긴 변을 이 크기로 줄인다 (큰 정사영 그림의 속도·메모리 보호)
    sift_contrast_threshold: float = 0.03   # 작을수록 무늬가 약한 콘크리트면에서도 특징점이 많이 나온다 (OpenCV 기본 0.04)
    use_clahe: bool = True              # 특징점을 찾기 전에 명암을 고르게 편다 (밋밋한 바닥면에 도움)
    ratio_test: float = 0.75            # Lowe 비율 검사: 가장 닮은 짝이 둘째로 닮은 짝보다 이 비율만큼 더 닮아야 통과
    mutual_check: bool = True           # A→B 와 B→A 양쪽에서 서로를 고른 짝만 남긴다

    # ---------------------------------------------------------------- 가림 (특징점을 찾지 않을 곳)
    border_margin_px: int = 12          # 검은 배경과의 경계에서 이만큼 안쪽만 쓴다 (특징점을 찾는 크기 기준 픽셀)
    mask_marker: bool = True            # 마커 둘레를 가린다 (마커를 옮겨 찍었으면 마커 위 특징점이 틀린 짝을 만든다)
    marker_mask_scale: float = 2.3      # 마커 네 모서리를 중심에서 이 배율로 넓힌 범위를 가린다 (마커가 인쇄된 종이까지 덮도록)

    # ---------------------------------------------------------------- 강체 변환 추정 (RANSAC)
    ransac_threshold_mm: float = 5.0    # 짝의 어긋남이 이 값보다 작으면 맞는 짝(inlier)으로 본다
    ransac_iterations: int = 3000       # 시도 횟수
    ransac_seed: int = 0                # 같은 입력이면 같은 결과가 나오게 고정
    min_inliers: int = 12               # 맞는 짝이 이보다 적으면 정합 실패
    min_inlier_ratio: float = 0.10      # 비율 검사를 통과한 짝 중 맞는 짝의 비율이 이보다 낮으면 실패
    min_spread_mm: float = 150.0        # 맞는 짝이 이보다 좁은 범위에 몰려 있으면 실패 (회전을 믿을 수 없음)
    scale_warning: float = 0.02         # 진단용 축척이 1 에서 이만큼 넘게 벗어나면 경고 (정사영 축척이 서로 다르다는 뜻)

    # ---------------------------------------------------------------- 그림 출력
    overlay_alpha: float = 0.5          # 겹쳐 보기에서 위 그림의 투명도
    max_canvas_side_px: int = 8000      # 합친 그림 한 변의 최대 픽셀
    max_canvas_pixels: int = 40_000_000  # 합친 그림 전체 픽셀 한도 (넘으면 축척을 거칠게 해서 줄인다)
    match_view_max_side_px: int = 1400  # 짝 그림에서 한 장의 긴 변
    max_drawn_matches: int = 400        # 짝 그림에 그리는 선의 최대 개수
    display_max_width: int = 1600
    display_max_height: int = 850
    save_debug: bool = True             # 단계별 중간 결과(가림 그림, 특징점 그림)를 debug 폴더에 저장

    # ---------------------------------------------------------------- 폴더
    input_dir: Path = PROJECT_DIR / "input"
    output_dir: Path = PROJECT_DIR / "output"
