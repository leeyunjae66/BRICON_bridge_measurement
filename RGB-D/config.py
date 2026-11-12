"""RGB-D 촬영·검증 설정값."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # ---------------------------------------------------------------- 카메라 영상 크기
    color_size: tuple[int, int] = (1280, 720)       # 컬러 (가로, 세로). 깊이는 이 컬러 사진의 픽셀에 맞춰 정렬해서 저장한다
    # 깊이 영상 크기는 카메라마다 가장 정확한 값이 다르다 (Intel 권장: D435 계열 848×480, D415·D455 는 1280×720).
    # None 이면 연결된 카메라 이름을 보고 고른다. 직접 정하려면 (가로, 세로) 를 넣는다.
    depth_size: tuple[int, int] | None = None
    fps: int = 30
    emitter: bool = True                            # 적외선 무늬 투사 (밋밋한 면에서 깊이를 잡는 데 필요)
    visual_preset: str | None = None                # 예: "High Accuracy". None 이면 카메라의 현재 설정 그대로

    # ---------------------------------------------------------------- 촬영
    warmup_frames: int = 30                         # 켠 직후 노출이 안정될 때까지 버리는 프레임 수
    average_frames: int = 30                        # 촬영 한 번에 평균할 프레임 수 (많을수록 잡음이 줄지만 그동안 움직이면 안 된다)
    min_valid_ratio: float = 0.5                    # 그 프레임들 중 이 비율 이상에서 깊이가 잡힌 픽셀만 값을 준다

    # ---------------------------------------------------------------- 화면
    depth_view_min_mm: float = 200.0                # 깊이 색 지도의 범위 (가까움 = 빨강, 멂 = 파랑)
    depth_view_max_mm: float = 3000.0
    display_max_width: int = 1700
    display_max_height: int = 900

    output_dir: Path = PROJECT_DIR / "output"
