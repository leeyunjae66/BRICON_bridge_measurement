"""
설정값 모음.

여기 있는 값이 기본값이고, 자주 바꾸는 값(마커 크기, 출력 축척 등)은 실행할 때 옵션으로도 바꿀 수 있다.
    python main.py --marker-size 199.5 --mm-per-pixel 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # ---------------------------------------------------------------- 마커
    aruco_dictionary: str = "DICT_4X4_50"   # 마커 사전 이름 (cv2.aruco 에 있는 이름 그대로)
    marker_id: int = 0                      # 찾을 마커 번호
    # 마커 한 변의 실제 길이 (mm). 흰 여백을 뺀 '검은 정사각형 바깥 변' 을 자로 재서 넣는다.
    # 인쇄하면 200 mm 가 정확히 안 나오는 일이 많으므로, 잰 값이 다르면 여기를 고친다.
    marker_size_mm: float = 200.0

    # ---------------------------------------------------------------- 출력
    mm_per_pixel: float = 1.0               # 정사영 그림의 축척: 픽셀 하나가 몇 mm 인지
    max_output_side_px: int = 10000         # 출력 그림 한 변의 최대 픽셀 수 (넘으면 범위를 잘라낸다)
    max_output_pixels: int = 60_000_000     # 출력 그림 전체 픽셀 수 한도 (메모리 보호)
    # 지평선 쪽 먼 곳은 끝없이 늘어나므로 버린다. 마커 자리의 값을 1 로 봤을 때 이 값보다 작은 곳은 쓰지 않는다.
    # (0.1 이면 마커 자리보다 깊이 방향으로 약 100배 넘게 늘어나는 곳을 버린다)
    horizon_min_w: float = 0.1

    # ---------------------------------------------------------------- 모서리 다듬기
    detect_long_side_px: int = 1600         # 마커를 찾을 때 줄여서 보는 크기 (고해상도 사진을 빠르고 안정적으로 찾기 위함)
    subpixel_max_iter: int = 50             # 모서리 위치 다듬기 반복 횟수
    subpixel_epsilon: float = 0.01          # 모서리 위치 다듬기 멈춤 기준 (픽셀)

    # ---------------------------------------------------------------- 오차 예상
    corner_noise_px: float = 0.3            # '모서리를 이만큼 틀리게 찾았다면' 하고 가정하는 값 (픽셀)

    # ---------------------------------------------------------------- 렌즈 보정 (선택)
    # 카메라 보정 파일(JSON)이 있으면 경로를 넣는다. 없으면 None 으로 두고, 렌즈 왜곡은 보정하지 않는다.
    # 형식: {"image_width": 4000, "image_height": 3000, "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [k1,k2,p1,p2,k3]}
    # 보정값은 이 프로그램이 사진을 읽은 방향(회전 적용 뒤)과 같은 방향·같은 해상도로 구한 것이어야 한다.
    calibration_file: str | None = None

    # ---------------------------------------------------------------- 화면·폴더
    display_max_width: int = 1600           # 화면에 띄울 때의 최대 크기 (저장 파일은 줄이지 않는다)
    display_max_height: int = 850
    input_dir: Path = PROJECT_DIR / "input"
    output_dir: Path = PROJECT_DIR / "output"
