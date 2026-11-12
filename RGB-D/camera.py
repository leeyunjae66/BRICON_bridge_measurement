"""
RealSense 카메라 다루기 (pyrealsense2 = RealSense SDK 의 파이썬 판).

깊이와 컬러를 같이 받고, 깊이를 컬러 사진의 픽셀에 맞춰 정렬한다.
그래서 돌려주는 깊이 지도는 컬러 사진과 크기가 같고, 같은 (u, v) 가 같은 곳을 가리킨다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from depthmath import Intrinsics

try:
    import pyrealsense2 as rs
except ImportError:                                  # 설치 전에도 다른 파일(계산, 검사)은 쓸 수 있게
    rs = None


class CameraError(Exception):
    """카메라를 쓸 수 없을 때 (연결 안 됨, 영상 모드 지원 안 함 등)."""


@dataclass
class Frame:
    color: np.ndarray        # (H,W,3) BGR
    depth_mm: np.ndarray     # (H,W) float32, 컬러에 맞춰 정렬된 z (mm). 깊이가 없는 곳은 NaN


def _best_depth_size(name: str) -> tuple[int, int]:
    """카메라 이름으로 깊이 영상 크기를 고른다 (Intel 권장값)."""
    return (848, 480) if "D43" in name else (1280, 720)


class Camera:
    def __init__(self, cfg: Config):
        if rs is None:
            raise CameraError("pyrealsense2 가 설치돼 있지 않습니다:  .venv\\Scripts\\python -m pip install -r requirements.txt")
        self.cfg = cfg
        self.pipeline = None
        self.align = None
        self.depth_scale_mm = 1.0
        self.intrinsics: Intrinsics | None = None
        self.rs_intrinsics = None
        self.info: dict = {}

    def start(self) -> None:
        """카메라를 켠다. 원하는 영상 크기가 안 되면 작은 크기로 낮춰 다시 시도한다."""
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise CameraError("RealSense 카메라를 찾지 못했습니다. USB 케이블이 꽂혀 있는지, 다른 프로그램(RealSense Viewer 등)이 쓰고 있지 않은지 확인하세요.")
        dev = devices[0]
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
        firmware = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else "?"
        depth_size = self.cfg.depth_size or _best_depth_size(name)
        tries = [(self.cfg.color_size, depth_size, self.cfg.fps), ((1280, 720), (1280, 720), 30), ((640, 480), (640, 480), 30), ((640, 480), (640, 480), 15)]
        errors = []
        profile = None
        for color_size, d_size, fps in dict.fromkeys(tries):
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, d_size[0], d_size[1], rs.format.z16, fps)
            config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps)
            pipeline = rs.pipeline()
            try:
                profile = pipeline.start(config)
            except RuntimeError as e:
                errors.append(f"컬러 {color_size[0]}×{color_size[1]}, 깊이 {d_size[0]}×{d_size[1]}, {fps}fps: {e}")
                continue
            self.pipeline = pipeline
            used = (color_size, d_size, fps)
            break
        if profile is None:
            hint = "  (USB 2 로 연결되면 큰 영상 모드를 쓸 수 없습니다. USB 3 포트와 케이블을 쓰세요.)" if usb.startswith("2") else ""
            raise CameraError("카메라를 켜지 못했습니다." + hint + "\n  " + "\n  ".join(errors))

        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale_mm = float(sensor.get_depth_scale()) * 1000.0       # 깊이 원래 값 1 이 몇 mm 인지
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0 if self.cfg.emitter else 0.0)
        preset = None
        if self.cfg.visual_preset and sensor.supports(rs.option.visual_preset):
            rng = sensor.get_option_range(rs.option.visual_preset)
            for k in range(int(rng.min), int(rng.max) + 1):
                if sensor.get_option_value_description(rs.option.visual_preset, k).lower() == self.cfg.visual_preset.lower():
                    sensor.set_option(rs.option.visual_preset, k)
                    preset = self.cfg.visual_preset
        self.align = rs.align(rs.stream.color)
        ri = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.rs_intrinsics = ri
        model = str(ri.model).split(".")[-1].lower()
        self.intrinsics = Intrinsics(ri.width, ri.height, ri.fx, ri.fy, ri.ppx, ri.ppy, model, tuple(ri.coeffs))
        self.info = {"name": name, "serial": serial, "firmware": firmware, "usb": usb,
                     "color_size": list(used[0]), "depth_size": list(used[1]), "fps": used[2],
                     "depth_unit_mm": self.depth_scale_mm, "emitter": self.cfg.emitter, "visual_preset": preset,
                     "fallback_note": "; ".join(errors) if errors else None}

    def read(self, timeout_ms: int = 5000) -> Frame | None:
        """한 프레임을 받는다. 깊이나 컬러가 빠진 프레임이면 None."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as e:
            raise CameraError(f"카메라에서 영상이 오지 않습니다 ({e}). 케이블이 빠졌는지 확인하세요.") from e
        aligned = self.align.process(frames)
        d, c = aligned.get_depth_frame(), aligned.get_color_frame()
        if not d or not c:
            return None
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale_mm
        depth[depth <= 0] = np.nan                    # 0 은 '깊이를 못 잡음' 이라는 뜻
        return Frame(np.asanyarray(c.get_data()).copy(), depth)

    def sdk_point(self, u: float, v: float, z_mm: float) -> np.ndarray:
        """SDK 가 계산한 3차원 점 (우리 계산이 SDK 와 같은지 확인할 때 쓴다)."""
        return np.array(rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [float(u), float(v)], float(z_mm)))

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
