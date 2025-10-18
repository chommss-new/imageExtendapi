# Image Extender API (Stable Diffusion)

Stable Diffusion을 사용하여 이미지를 자연스럽게 확장하는 FastAPI 기반 웹 서비스입니다.

## 주요 기능

- **이미지 확장**: 상하좌우 원하는 픽셀만큼 AI로 자연스럽게 확장
- **목표 크기 확장**: 특정 크기로 이미지를 확장 (원본 위치 지정 가능)
- **인페인팅**: 사용자 정의 마스크 영역을 AI로 채우기
- **Stable Diffusion 2 Inpainting**: 최고 품질의 AI 생성

## 기술 스택

- **FastAPI**: 고성능 웹 프레임워크
- **Stable Diffusion 2**: Hugging Face Diffusers 라이브러리
- **PIL/Pillow**: 이미지 처리
- **NumPy**: 배열 처리
- **OpenCV**: 이미지 변환

## 설치

```bash
pip install fastapi uvicorn diffusers transformers accelerate torch pillow numpy opencv-python
```

## 실행

```bash
python main.py
# 또는
uvicorn main:app --host 0.0.0.0 --port 8000
```

서버가 시작되면 http://localhost:8000 에서 접속 가능합니다.

## API 엔드포인트

### 1. POST /extend
이미지를 지정된 픽셀만큼 확장합니다.

**Parameters:**
- `file`: 이미지 파일
- `top`: 위쪽 확장 픽셀 (0-2000)
- `bottom`: 아래쪽 확장 픽셀 (0-2000)
- `left`: 왼쪽 확장 픽셀 (0-2000)
- `right`: 오른쪽 확장 픽셀 (0-2000)
- `method`: 확장 방법 (기본값: "sd")
- `format`: 출력 포맷 (png/jpeg/webp)

### 2. POST /extend-to-size
이미지를 목표 크기로 확장합니다.

**Parameters:**
- `file`: 이미지 파일
- `target_width`: 목표 너비 (1-4096)
- `target_height`: 목표 높이 (1-4096)
- `position`: 이미지 위치 (center/top-left/top-right/bottom-left/bottom-right)
- `method`: 확장 방법 (기본값: "sd")
- `format`: 출력 포맷 (png/jpeg/webp)

### 3. POST /inpaint
사용자가 지정한 마스크 영역을 AI로 채웁니다.

**Parameters:**
- `file`: 원본 이미지 파일
- `mask`: 마스크 이미지 (흰색=채울 영역)
- `prompt`: SD 생성 프롬프트
- `format`: 출력 포맷 (png/jpeg/webp)

### 4. GET /health
서버 상태 확인

### 5. GET /methods
사용 가능한 확장 방법 목록

## 테스트 페이지

프로젝트에 포함된 HTML 테스트 페이지:

- **test_client.html**: 일반 이미지 확장 테스트
- **test_inpaint.html**: 마스크 기반 인페인팅 테스트

## 라이센스

- **Stable Diffusion 2**: CreativeML Open RAIL++-M (상업적 사용 가능)
- **본 프로젝트**: MIT License

## 시스템 요구사항

- **최소**: CPU, 8GB RAM
- **권장**: CUDA GPU, 16GB RAM (10-20배 빠름)
- **저장 공간**: ~5GB (Stable Diffusion 모델)

## 참고

첫 실행 시 Stable Diffusion 모델이 자동으로 다운로드됩니다 (~5GB).
GPU가 없으면 CPU 모드로 동작하며, 처리 속도가 느릴 수 있습니다.
