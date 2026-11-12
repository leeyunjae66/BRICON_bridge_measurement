"""
화면 확대·이동 계산. 그림 좌표와 화면 좌표를 서로 바꾼다 (화면 도구와 떨어져 있어 혼자 시험할 수 있다).

    그림 좌표 (u, v) : 픽셀 가운데 기준. 맨 왼쪽 위 픽셀의 가운데가 (0, 0), 그 픽셀의 왼쪽 위 귀퉁이는 (-0.5, -0.5).
    화면 좌표 (sx, sy): 그림을 그리는 판(canvas)의 왼쪽 위가 (0, 0).
    scale            : 그림 픽셀 하나가 화면에서 몇 픽셀인지.
    (ox, oy)         : 그림의 왼쪽 위 귀퉁이가 놓인 화면 좌표.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Viewport:
    scale: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    def to_screen(self, u: float, v: float) -> tuple[float, float]:
        return (u + 0.5) * self.scale + self.ox, (v + 0.5) * self.scale + self.oy

    def to_image(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.ox) / self.scale - 0.5, (sy - self.oy) / self.scale - 0.5

    def zoom_at(self, sx: float, sy: float, factor: float, min_scale: float, max_scale: float) -> None:
        """화면의 (sx, sy) 자리에 있는 그림 점이 그 자리에 그대로 있도록 확대·축소한다."""
        new = min(max(self.scale * factor, min_scale), max_scale)
        k = new / self.scale
        self.ox = sx - (sx - self.ox) * k
        self.oy = sy - (sy - self.oy) * k
        self.scale = new

    def pan(self, dx: float, dy: float) -> None:
        self.ox += dx
        self.oy += dy

    def fit(self, img_w: int, img_h: int, view_w: int, view_h: int, margin: int = 10) -> None:
        """그림 전체가 화면에 들어오게 맞춘다."""
        self.scale = max(1e-6, min((view_w - 2 * margin) / img_w, (view_h - 2 * margin) / img_h))
        self.ox = (view_w - img_w * self.scale) / 2.0
        self.oy = (view_h - img_h * self.scale) / 2.0

    def centre_on(self, u: float, v: float, view_w: int, view_h: int) -> None:
        """그림의 (u, v) 가 화면 한가운데에 오게 옮긴다."""
        self.ox = view_w / 2.0 - (u + 0.5) * self.scale
        self.oy = view_h / 2.0 - (v + 0.5) * self.scale

    def visible_box(self, img_w: int, img_h: int, view_w: int, view_h: int):
        """화면에 보이는 그림 부분 → ((x0, y0, x1, y1) 그림 픽셀 범위, (화면 x, 화면 y), (화면 가로, 화면 세로)). 안 보이면 None.

        범위는 픽셀 경계에 맞춘 정수라서, 크게 확대했을 때 픽셀이 반쯤 잘려 흔들리지 않는다.
        """
        x0 = max(0, math.floor(-self.ox / self.scale))
        y0 = max(0, math.floor(-self.oy / self.scale))
        x1 = min(img_w, math.ceil((view_w - self.ox) / self.scale))
        y1 = min(img_h, math.ceil((view_h - self.oy) / self.scale))
        if x1 <= x0 or y1 <= y0:
            return None
        dest = (round(self.ox + x0 * self.scale), round(self.oy + y0 * self.scale))
        size = (max(1, round((x1 - x0) * self.scale)), max(1, round((y1 - y0) * self.scale)))
        return (x0, y0, x1, y1), dest, size
