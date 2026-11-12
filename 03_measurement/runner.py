"""
앞 단계 프로그램을 '그대로' 실행하는 부분. 화면의 정사영·결합 단추가 이것을 부른다.

1단계(01_orthorectification)와 2단계(02_registration)의 코드를 불러 쓰거나 바꾸지 않는다.
각 단계의 main.py 를 그 단계의 가상환경으로 따로 실행하고, 그 프로그램이 저장한 결과 파일만 읽는다.
그래서 명령창에서 직접 실행한 것과 결과가 똑같다.

    run_orthorectification(사진, 마커 크기)   →  01_orthorectification\\output\\<사진 이름>\\orthorectified.png 등
    run_registration(정사영 그림들, 방식)      →  02_registration\\output\\<이름>\\registered_mosaic.png 등
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
STAGE1_DIR = PROJECT_DIR.parent / "01_orthorectification"
STAGE2_DIR = PROJECT_DIR.parent / "02_registration"
MODES = {"marker": "마커를 안 옮김 (마커 기준)", "feature": "마커를 옮김 (특징점)"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)      # Windows 에서 검은 명령창이 깜빡이지 않게


class RunnerError(Exception):
    """앞 단계 프로그램을 실행할 수 없을 때 (폴더나 가상환경이 없음 등)."""


@dataclass
class RunResult:
    ok: bool
    returncode: int
    output: str                  # 프로그램이 화면에 찍은 글 전체 (실행 기록)
    summary: str                 # 화면에 보여 줄 짧은 요약 또는 실패 이유
    out_dir: Path
    result_image: Path | None    # 성공했을 때 화면에 띄울 그림


def safe_name(text: str) -> str:
    """폴더 이름으로 쓸 수 없는 글자를 _ 로 바꾼다."""
    return re.sub(r'[\\/:*?"<>|\s]+', "_", text).strip("_") or "image"


def _run(stage_dir: Path, args: list[str], timeout: float) -> tuple[int, str]:
    """그 단계의 가상환경으로 main.py 를 실행하고 (끝나는 코드, 찍힌 글) 을 돌려준다."""
    python = stage_dir / ".venv" / "Scripts" / "python.exe"
    main = stage_dir / "main.py"
    if not main.is_file():
        raise RunnerError(f"프로그램을 찾지 못했습니다: {main}")
    if not python.is_file():
        raise RunnerError(f"{stage_dir.name} 의 가상환경이 없습니다.\n그 폴더에서 다음을 실행해 설치하세요:\n"
                          "  py -3 -m venv .venv\n  .venv\\Scripts\\python -m pip install -r requirements.txt")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        p = subprocess.run([str(python), str(main), *args], cwd=str(stage_dir), capture_output=True, env=env,
                           timeout=timeout, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired as e:
        raise RunnerError(f"{stage_dir.name} 이 {timeout:.0f}초 안에 끝나지 않아 멈췄습니다.") from e
    except OSError as e:
        raise RunnerError(f"{stage_dir.name} 을 실행하지 못했습니다: {e}") from e
    text = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace").strip()
    return p.returncode, text + (f"\n[stderr]\n{err}" if err else "")


def _error_text(output: str) -> str:
    """찍힌 글에서 '[오류]' 로 시작하는 부분을 뽑는다 (없으면 마지막 몇 줄)."""
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("[오류]"):
            block = [line[len("[오류]"):].strip()]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or nxt.startswith("["):
                    break
                block.append(nxt.strip())
            return "\n".join(block)
    tail = [ln for ln in lines if ln.strip()][-6:]
    return "\n".join(tail) or "이유를 알 수 없습니다. 실행 기록을 확인하세요."


def _pick_lines(output: str, keys: tuple[str, ...]) -> str:
    return "\n".join(ln.strip() for ln in output.splitlines() if any(k in ln for k in keys))


def _fresh(path: Path, started: float) -> bool:
    """이번 실행에서 새로 만들어진 파일인지 (예전 결과를 이번 결과로 착각하지 않게)."""
    return path.is_file() and path.stat().st_mtime >= started - 2.0


def run_orthorectification(photo: Path, marker_size_mm: float, out_dir: Path | None = None,
                           stage_dir: Path = STAGE1_DIR, timeout: float = 600.0) -> RunResult:
    """1단계를 실행해 사진 한 장을 정사영한다. 결과는 out_dir (기본: 1단계 output\\<사진 이름>) 에 저장된다."""
    photo = Path(photo).resolve()
    out_dir = Path(out_dir) if out_dir else stage_dir / "output" / safe_name(photo.stem)
    started = time.time()
    code, output = _run(stage_dir, ["--image", str(photo), "--marker-size", f"{float(marker_size_mm):g}",
                                    "--no-show", "--output", str(out_dir)], timeout)
    image = out_dir / "orthorectified.png"
    ok = code == 0 and _fresh(image, started) and _fresh(out_dir / "homography.json", started)
    if ok:
        summary = _pick_lines(output, ("마커 검출", "한 변: 약", "정사영 결과", "담긴 범위", "렌즈 왜곡", "버렸습니다", "한도를 넘어"))
    else:
        summary = _error_text(output)
    return RunResult(ok, code, output, summary, out_dir, image if ok else None)


def run_registration(ortho_images: list[Path], mode: str = "marker", name: str | None = None, out_root: Path | None = None,
                     stage_dir: Path = STAGE2_DIR, timeout: float = 1800.0) -> RunResult:
    """2단계를 실행해 정사영 그림들을 정합한다. 첫 번째 그림이 기준이다.

    mode: "marker" = 마커를 옮기지 않고 찍음 (마커 기준), "feature" = 마커를 옮겨 찍음 (특징점).
    결과는 out_root (기본: 2단계 output) 아래의 name 폴더에 저장된다.
    """
    if mode not in MODES:
        raise RunnerError(f"알 수 없는 결합 방식입니다: {mode}")
    images = [Path(p).resolve() for p in ortho_images]
    if len(images) < 2:
        raise RunnerError("결합하려면 정사영한 그림이 2장 이상 필요합니다.")
    out_root = Path(out_root) if out_root else stage_dir / "output"
    name = safe_name(name or f"gui_{images[0].parent.name}_{len(images)}img_{mode}")
    out_dir = out_root / name
    started = time.time()
    code, output = _run(stage_dir, ["--images", *map(str, images), "--mode", mode, "--no-show",
                                    "--output", str(out_root), "--name", name], timeout)
    image = out_dir / "registered_mosaic.png"
    # 일부 그림만 실패해도(끝나는 코드 1) 합친 그림이 새로 만들어졌으면 결과를 보여 준다
    ok = code in (0, 1) and _fresh(image, started) and _fresh(out_dir / "registration.json", started)
    keys = ("[결과]", "방식:", "회전 ", "맞는 짝", "진단용 축척", "실패 이유", "[주의]", "[비교]", "[합친 그림]", "[실패]", "[오류]")
    summary = _pick_lines(output, keys) if ok else _error_text(output)
    return RunResult(ok, code, output, summary, out_dir, image if ok else None)
