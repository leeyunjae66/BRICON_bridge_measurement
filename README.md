# 교량 바닥판 이미지 기반 치수 측정

스마트폰 사진에서 ArUco 마커를 기준으로 바닥판의 실제 치수를 재는 프로그램이다. 단계마다 폴더가 따로 있고, 서로의 코드를 불러 쓰지 않는다 (앞 단계의 결과 파일만 읽는다).

| 폴더 | 하는 일 | 상태 |
|---|---|---|
| `01_orthorectification` | 마커 1개로 사진 전체를 정사영 보정 | 구현 |
| `02_registration` | 정사영 그림 여러 장을 하나의 평면 좌표계로 정합 | 구현 |
| `03_measurement` | 정합된 그림에서 두 점 사이 거리 측정 | 예정 |

## 설치 (Windows, Python 3.11 / 3.12)

단계마다 가상환경을 따로 만든다. 각 폴더에서:

```
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`opencv-python` 과 `opencv-contrib-python` 을 같이 설치하면 충돌하므로 `requirements.txt` 대로 contrib 하나만 설치한다.

## 1단계: 정사영 보정

```
cd 01_orthorectification
.venv\Scripts\python main.py                          파일 선택 창에서 사진을 고른다
.venv\Scripts\python main.py --image input\사진.jpg --marker-size 150 --output output\사진1
```

- 마커: `DICT_4X4_50`, ID 0. 한 변의 실제 길이(검은 정사각형 바깥 변)를 `--marker-size` 로 준다 (기본 200 mm).
- 결과: `aruco_detected.png`, `orthorectified.png`, `comparison.png`, `homography.json`
- 사진 옆에 같은 이름의 `.check.json` (길이를 아는 두 점의 위치와 기준 길이)을 두면 `comparison.png` 아래에 치수 오차 표가 붙는다. 형식은 `main.py` 맨 위 설명 참고.
- 결과 파일 이름이 고정이므로, 사진마다 `--output` 으로 폴더를 따로 준다.

## 2단계: 정합

```
cd 02_registration
.venv\Scripts\python main.py                          파일 선택 창에서 정사영 그림을 2장 이상 고른다 (첫 번째가 기준)
.venv\Scripts\python main.py --images ..\01_orthorectification\output\사진1\orthorectified.png ..\01_orthorectification\output\사진2\orthorectified.png
.venv\Scripts\python main.py --images ... --mode marker      마커를 옮기지 않고 카메라만 옮겨 찍었을 때
.venv\Scripts\python -m unittest test_registration -v        검사
```

- 메타데이터는 그림과 같은 폴더의 `homography.json` 을 읽는다. 없으면 축척을 추측하지 않고 오류로 알린다.
- 방식 두 가지
  - `feature` (기본): 겹치는 바닥면의 특징점(SIFT/ORB)으로 회전과 이동을 구한다. 마커를 옮겨 가며 찍었을 때 쓴다.
  - `marker`: 같은 자리의 같은 마커가 두 사진에 찍혔을 때. 두 정사영은 이미 같은 좌표계이므로 그대로 겹친다.
- 변환은 강체(회전 + 이동, 축척 1 고정)다. 축척은 진단용으로만 계산한다.
- 결과: `feature_matches.png`, `inlier_matches.png`, `registration_overlay.png`, `registration_overlay_zoom.png`, `registered_mosaic.png`, `registration.json`, `log.txt`

### 좌표계와 행렬 이름

`A_from_B` 는 B 좌표를 A 좌표로 바꾸는 3×3 행렬이다 (`p_A = A_from_B · p_B`).

- 마커 평면 좌표 (mm): 원점 = 마커 좌상 모서리, x = 좌상 → 우상, y = 좌상 → 좌하
- `plane_mm_from_pixel`: 정사영 그림 픽셀 → 그 그림의 마커 평면 mm
- `T_ref_from_image`: 그 그림의 마커 평면 mm → 기준 그림의 마커 평면 mm
- `ref_mm_from_pixel` = `T_ref_from_image · plane_mm_from_pixel`: 거리 측정에 쓰는 행렬

## 알아 둘 한계

- 평면 변환이므로 마커와 같은 평면에 있는 것만 바르게 펴진다. 평면에서 솟거나 꺼진 것은 위치가 틀어진다.
- 마커 1개로 사진 전체를 펴면 마커에서 멀수록 오차가 빠르게 커진다. 150 mm 마커, 1050 × 1400 픽셀 사진으로 시험했을 때 마커 바로 옆(20 cm 안)은 1~3 mm, 0.5 m 떨어진 곳은 약 2 cm 어긋났다.
- 카메라 보정값이 없으면 렌즈 왜곡은 보정하지 않는다.
- 정합에서 나오는 '짝의 어긋남' 은 정합에 쓴 점끼리 얼마나 맞는지일 뿐, 치수 정확도를 검증한 값이 아니다. 치수 정확도는 길이를 아는 물체나 검증 점으로 따로 확인한다.

## 저장소에 올리지 않는 것

사진(`input/`)과 결과(`output/`), 가상환경(`.venv/`)은 올리지 않는다 (`.gitignore`).
