"""측정 화면 설정값."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    window_size: str = "1560x900"           # 처음 창 크기
    side_panel_width: int = 380             # 오른쪽 측정 목록 칸의 폭
    photo_panel_width: int = 300            # 왼쪽 사진 목록 칸의 폭
    zoom_step: float = 1.25                 # 휠 한 칸에 확대되는 배율
    max_zoom: float = 64.0                  # 가장 크게 확대했을 때 그림 픽셀 하나의 화면 픽셀 수
    corner_noise_px: float = 0.3            # 예상 오차 계산에서 가정하는 마커 모서리 읽기 오차 (픽셀)
    line_color: str = "#ffd400"             # 측정선
    selected_color: str = "#00e5ff"         # 목록에서 고른 측정선
    pending_color: str = "#ff5ad1"          # 시작점만 찍은 상태의 선
    background: str = "#2b2b2b"
    # 첫 목록의 처음 값. 마커 크기와 결합 방식은 목록(탭)마다 따로 두고, 새 목록은 보고 있던 목록의 값을 물려받는다.
    marker_size_mm: float = 150.0           # 마커 한 변의 실제 길이
    registration_mode: str = "marker"       # "marker" = 마커를 안 옮김, "feature" = 마커를 옮김
    # 목록 이름·넣은 사진·계산 결과·목록별 값을 기억해 두는 파일. 창을 다시 켜면 그대로 되살린다.
    session_file: Path = PROJECT_DIR / "session.json"
    # 파일 선택 창이 처음 여는 폴더
    photo_dir: Path = PROJECT_DIR.parent / "01_orthorectification" / "input"     # 사진 추가
    start_dir: Path = PROJECT_DIR.parent / "01_orthorectification" / "output"    # 결과 열기
