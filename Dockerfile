# CUDA 12.8.0 베이스 이미지 사용 (RTX 5080 sm_120 지원)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# Python 및 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크 생성
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# PyTorch 2.7.0 설치 (RTX 5080 sm_120 지원)
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

# Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# 나머지 패키지 설치
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# FastAPI 포트 노출
EXPOSE 8001

# Uvicorn으로 FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
