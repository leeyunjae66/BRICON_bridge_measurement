"""
정사영·정합 결과 그림에서 두 점 사이의 실제 거리를 잰다 (3단계). 앞 단계의 결과 파일만 읽는다.

    python main.py                                              창을 띄우고 '열기' 로 그림을 고른다
    python main.py --image ..\\01_orthorectification\\output\\photo_4\\orthorectified.png

열 수 있는 그림 (그림과 같은 폴더에 좌표 정보 JSON 이 있어야 한다)
    1단계: orthorectified.png (펴진 그림), aruco_detected.png (원본 사진)   ← homography.json
    2단계: registered_mosaic.png (합친 그림)                               ← registration.json
쓰는 법: 휠 = 확대·축소, 오른쪽 단추 끌기 = 이동, 왼쪽 단추 = 시작점 → 끝점, Esc = 시작점 취소, Delete = 고른 측정 지우기.
"""

from __future__ import annotations

import argparse
import sys

from config import Config
from viewer import MeasureApp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="정사영·정합 결과 그림에서 두 점 사이의 실제 거리를 잰다.")
    ap.add_argument("--image", help="처음에 열 그림 (없으면 창에서 '열기' 로 고른다)")
    args = ap.parse_args(argv)
    try:
        # 화면 배율(125 %, 150 % …)을 쓰는 PC 에서 글자와 그림이 흐려지지 않게 한다
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = MeasureApp(Config(), args.image)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
