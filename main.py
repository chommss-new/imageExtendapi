"""
이미지 확장 API 서버
FastAPI를 사용하여 이미지 확장 서비스를 제공합니다.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
import numpy as np
from typing import Optional
from image_extender import ImageExtender
from prompt_translator import get_translator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Image Extender API",
    description="이미지를 배경에 맞춰 확장하는 API",
    version="1.0.0"
)

# CORS 설정 (웹에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ImageExtender 인스턴스 (서버 시작시 한번만 생성)
extender = ImageExtender(method="sd")


@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Image Extender API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extend": "이미지를 지정된 픽셀만큼 확장",
            "POST /extend-to-size": "이미지를 목표 크기로 확장",
            "POST /inpaint": "마스크 영역을 AI로 채우기 (사용자 정의 마스크)",
            "GET /health": "서버 상태 확인",
            "GET /methods": "사용 가능한 방법 목록"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "image-extender"}


@app.post("/extend")
async def extend_image(
    file: UploadFile = File(..., description="확장할 이미지 파일"),
    top: int = Form(0, ge=0, le=2000, description="위쪽 확장 픽셀"),
    bottom: int = Form(0, ge=0, le=2000, description="아래쪽 확장 픽셀"),
    left: int = Form(0, ge=0, le=2000, description="왼쪽 확장 픽셀"),
    right: int = Form(0, ge=0, le=2000, description="오른쪽 확장 픽셀"),
    method: str = Form("sd", description="확장 방법 (sd - Stable Diffusion)"),
    format: str = Form("png", description="출력 포맷 (png/jpeg/webp)")
):
    """
    이미지를 지정된 픽셀만큼 확장합니다.

    - **file**: 이미지 파일 (PNG, JPEG, WebP 등)
    - **top**: 위쪽으로 확장할 픽셀 수
    - **bottom**: 아래쪽으로 확장할 픽셀 수
    - **left**: 왼쪽으로 확장할 픽셀 수
    - **right**: 오른쪽으로 확장할 픽셀 수
    - **method**: 확장 방법 (sd/lama/inpaint/replicate/reflect)
    - **format**: 출력 이미지 포맷sd
    """
    try:
        # 파일 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

        # 이미지 로드
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        original_size = image.size
        logger.info(f"Original image size: {original_size}")

        # 확장 값 검증
        if top + bottom + left + right == 0:
            raise HTTPException(status_code=400, detail="최소 한 방향은 확장해야 합니다.")

        # ImageExtender 설정
        logger.info(f"Creating ImageExtender with method={method}")
        temp_extender = ImageExtender(method=method)
        logger.info(f"ImageExtender created with temp_extender.method={temp_extender.method}")

        # 이미지 확장
        extended_image = temp_extender.extend_image(
            image=image,
            top=top,
            bottom=bottom,
            left=left,
            right=right
        )

        extended_size = extended_image.size
        logger.info(f"Extended image size: {extended_size}")

        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()

        # 포맷별 저장 옵션
        save_format = format.upper()
        if save_format == "JPEG" or save_format == "JPG":
            # JPEG는 RGB 모드만 지원
            if extended_image.mode in ('RGBA', 'LA', 'P'):
                extended_image = extended_image.convert('RGB')
            extended_image.save(img_byte_arr, format='JPEG', quality=95)
            media_type = "image/jpeg"
        elif save_format == "WEBP":
            extended_image.save(img_byte_arr, format='WEBP', quality=95)
            media_type = "image/webp"
        else:  # PNG (기본값)
            extended_image.save(img_byte_arr, format='PNG')
            media_type = "image/png"

        img_byte_arr.seek(0)

        # 스트리밍 응답
        return StreamingResponse(
            img_byte_arr,
            media_type=media_type,
            headers={
                "X-Original-Size": f"{original_size[0]}x{original_size[1]}",
                "X-Extended-Size": f"{extended_size[0]}x{extended_size[1]}",
                "X-Method": method
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extending image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"이미지 확장 중 오류 발생: {str(e)}")


@app.post("/extend-to-size")
async def extend_to_size(
    file: UploadFile = File(..., description="확장할 이미지 파일"),
    target_width: int = Form(..., ge=1, le=4096, description="목표 너비"),
    target_height: int = Form(..., ge=1, le=4096, description="목표 높이"),
    position: str = Form("center", description="이미지 위치 (center/top-left/top-right/bottom-left/bottom-right)"),
    method: str = Form("inpaint", description="확장 방법"),
    format: str = Form("png", description="출력 포맷")
):
    """
    이미지를 목표 크기로 확장합니다.

    - **file**: 이미지 파일
    - **target_width**: 목표 너비
    - **target_height**: 목표 높이
    - **position**: 원본 이미지의 위치
    - **method**: 확장 방법
    - **format**: 출력 이미지 포맷
    """
    try:
        # 파일 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

        # 이미지 로드
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        original_size = image.size
        logger.info(f"Original image size: {original_size}")

        # ImageExtender 설정
        temp_extender = ImageExtender(method=method)

        # 이미지 확장
        extended_image = temp_extender.extend_to_size(
            image=image,
            target_width=target_width,
            target_height=target_height,
            position=position
        )

        extended_size = extended_image.size
        logger.info(f"Extended image size: {extended_size}")

        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()

        save_format = format.upper()
        if save_format == "JPEG" or save_format == "JPG":
            if extended_image.mode in ('RGBA', 'LA', 'P'):
                extended_image = extended_image.convert('RGB')
            extended_image.save(img_byte_arr, format='JPEG', quality=95)
            media_type = "image/jpeg"
        elif save_format == "WEBP":
            extended_image.save(img_byte_arr, format='WEBP', quality=95)
            media_type = "image/webp"
        else:
            extended_image.save(img_byte_arr, format='PNG')
            media_type = "image/png"

        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type=media_type,
            headers={
                "X-Original-Size": f"{original_size[0]}x{original_size[1]}",
                "X-Extended-Size": f"{extended_size[0]}x{extended_size[1]}",
                "X-Method": method,
                "X-Position": position
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extending image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"이미지 확장 중 오류 발생: {str(e)}")


@app.post("/inpaint")
async def inpaint_image(
    file: UploadFile = File(..., description="원본 이미지 파일"),
    mask: UploadFile = File(..., description="마스크 이미지 (흰색=채울 영역)"),
    prompt: str = Form("natural background, high quality, detailed", description="SD 프롬프트 (한글/일본어/중국어/영어 지원)"),
    negative_prompt: str = Form("blurry, low quality, distorted, artifacts", description="네거티브 프롬프트 (다국어 지원)"),
    format: str = Form("png", description="출력 포맷 (png/jpeg/webp)")
):
    """
    사용자가 지정한 마스크 영역을 Stable Diffusion으로 채웁니다.

    - **file**: 원본 이미지 파일
    - **mask**: 마스크 이미지 (흰색 영역이 채워질 부분)
    - **prompt**: 생성할 내용 설명 프롬프트 (한글, 일본어, 중국어, 영어 모두 지원)
    - **negative_prompt**: 피하고 싶은 내용 (다국어 지원)
    - **format**: 출력 이미지 포맷

    프롬프트는 자동으로 영어로 번역되어 Stable Diffusion에 전달됩니다.
    """
    try:
        # 파일 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        if not mask.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="마스크는 이미지 파일이어야 합니다.")

        # 이미지 로드
        image_contents = await file.read()
        mask_contents = await mask.read()

        image_pil = Image.open(io.BytesIO(image_contents)).convert("RGB")
        mask_pil = Image.open(io.BytesIO(mask_contents)).convert("L")

        original_size = image_pil.size
        logger.info(f"Original image size: {original_size}")
        logger.info(f"Mask size: {mask_pil.size}")

        # 크기 확인
        if image_pil.size != mask_pil.size:
            raise HTTPException(
                status_code=400,
                detail=f"이미지와 마스크 크기가 다릅니다. 이미지: {image_pil.size}, 마스크: {mask_pil.size}"
            )

        # NumPy 배열로 변환
        image_array = np.array(image_pil)
        mask_array = np.array(mask_pil)

        # 프롬프트 자동 번역 (한글/일본어/중국어 → 영어)
        translator = get_translator()
        original_prompt = prompt
        original_negative = negative_prompt
        translated_prompt, translated_negative = translator.translate_with_fallback(prompt, negative_prompt)

        logger.info(f"Original prompt: {original_prompt}")
        logger.info(f"Translated prompt: {translated_prompt}")
        if original_negative != translated_negative:
            logger.info(f"Original negative: {original_negative}")
            logger.info(f"Translated negative: {translated_negative}")

        # Stable Diffusion inpainting
        logger.info(f"Using Stable Diffusion inpainting")
        from sd_inpainter import get_sd_inpainter
        inpainter = get_sd_inpainter()
        result_array = inpainter.inpaint(
            image=image_array,
            mask=mask_array,
            prompt=translated_prompt,
            negative_prompt=translated_negative
        )

        # NumPy to PIL
        result_pil = Image.fromarray(result_array)

        logger.info(f"Inpainting completed successfully")

        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()

        # 포맷별 저장 옵션
        save_format = format.upper()
        if save_format == "JPEG" or save_format == "JPG":
            if result_pil.mode in ('RGBA', 'LA', 'P'):
                result_pil = result_pil.convert('RGB')
            result_pil.save(img_byte_arr, format='JPEG', quality=95)
            media_type = "image/jpeg"
        elif save_format == "WEBP":
            result_pil.save(img_byte_arr, format='WEBP', quality=95)
            media_type = "image/webp"
        else:  # PNG (기본값)
            result_pil.save(img_byte_arr, format='PNG')
            media_type = "image/png"

        img_byte_arr.seek(0)

        # 스트리밍 응답
        return StreamingResponse(
            img_byte_arr,
            media_type=media_type,
            headers={
                "X-Original-Size": f"{original_size[0]}x{original_size[1]}",
                "X-Method": "sd",
                "X-Original-Prompt": original_prompt,
                "X-Translated-Prompt": translated_prompt
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in inpainting: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인페인팅 중 오류 발생: {str(e)}")


@app.get("/methods")
async def get_methods():
    """사용 가능한 확장 방법 목록"""
    return {
        "methods": [
            {
                "name": "sd",
                "description": "Stable Diffusion을 사용한 최고 품질 AI 확장",
                "quality": "excellent",
                "speed": "medium (GPU에서 빠름)",
                "requires": "Stable Diffusion 모델 다운로드 필요 (~5GB, 첫 실행 시)",
                "license": "무료 상업적 사용 가능 (CreativeML Open RAIL++-M)"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Image Extender API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
