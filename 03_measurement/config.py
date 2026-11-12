"""측정 화면 설정값."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    window_size: str = "1400x860"           # 처음 창 크기
    side_panel_width: int = 380             # 오른쪽 목록 칸의 폭
    zoom_step: float = 1.25                 # 휠 한 칸에 확대되는 배율
    max_zoom: float = 64.0                  # 가장 크게 확대했을 때 그림 픽셀 하나의 화면 픽셀 수
    corner_noise_px: float = 0.3            # 예상 오차 계산에서 가정하는 마커 모서리 읽기 오차 (픽셀)
    line_color: str = "#ffd400"             # 측정선
    selected_color: str = "#00e5ff"         # 목록에서 고른 측정선
    pending_color: str = "#ff5ad1"          # 시작점만 찍은 상태의 선
    background: str = "#2b2b2b"
    # 파일 선택 창이 처음 여는 폴더: 1단계 결과 폴더가 있으면 거기부터
    start_dir: Path = PROJECT_DIR.parent / "01_orthorectification" / "output"
