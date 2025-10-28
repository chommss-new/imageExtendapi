"""
이미지 확장 유틸리티
Stable Diffusion을 사용하여 배경을 자연스럽게 확장합니다.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging
from sd_inpainter import get_sd_inpainter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageExtender:
    """Stable Diffusion을 사용하여 이미지를 배경에 맞춰 확장하는 클래스"""

    def __init__(self, method: str = "sd", sd_prompt: str = None):
        """
        Args:
            method: 확장 방법 ('sd' - Stable Diffusion)
            sd_prompt: Stable Diffusion 프롬프트
        """
        if method != "sd":
            logger.warning(f"Method '{method}' not supported. Using 'sd' instead.")
            method = "sd"

        self.method = method
        self.sd_prompt = sd_prompt or "natural background, high quality, detailed, seamless, photorealistic"

        # SD 인페인터 초기화
        try:
            logger.info("Loading Stable Diffusion model (this may take a while on first run)...")
            self._sd_inpainter = get_sd_inpainter()
            logger.info(f"ImageExtender initialized with Stable Diffusion inpainting")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion: {e}")
            raise

    def extend_image(
        self,
        image: Image.Image,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
        **kwargs
    ) -> Image.Image:
        """
        이미지를 Stable Diffusion으로 확장합니다.

        Args:
            image: PIL Image 객체
            top: 위쪽 확장 픽셀
            bottom: 아래쪽 확장 픽셀
            left: 왼쪽 확장 픽셀
            right: 오른쪽 확장 픽셀

        Returns:
            확장된 PIL Image
        """
        logger.info(f"Extending image: top={top}, bottom={bottom}, left={left}, right={right}")
        return self._extend_with_sd(image, top, bottom, left, right, **kwargs)

    def _extend_with_sd(
        self,
        image: Image.Image,
        top: int, bottom: int, left: int, right: int,
        **kwargs
    ) -> Image.Image:
        """Stable Diffusion을 사용한 최고 품질 AI 확장"""

        # PIL to NumPy (RGB)
        img_array = np.array(image.convert('RGB'))
        h, w = img_array.shape[:2]
        new_h = h + top + bottom
        new_w = w + left + right

        # 1단계: 초기 캔버스 생성 (REPLICATE 방식)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        canvas_bgr = cv2.copyMakeBorder(
            img_bgr,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REPLICATE
        )
        canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

        # 2단계: 마스크 생성 (확장 영역만)
        mask = np.zeros((new_h, new_w), dtype=np.uint8)

        # 약간의 overlap으로 자연스러운 전환 (큰 확장에는 더 큰 overlap 필요)
        expand_sizes = [x for x in [top, bottom, left, right] if x > 0]
        if expand_sizes:
            max_expand = max(expand_sizes)
            # 큰 확장(100px 이상)에는 40-50px overlap, 작은 확장에는 10-20px
            if max_expand >= 100:
                overlap = min(50, max_expand // 4)
            else:
                overlap = min(20, max_expand // 3)
        else:
            overlap = 10

        if top > 0:
            mask[0:top+overlap, :] = 255
        if bottom > 0:
            mask[new_h-bottom-overlap:new_h, :] = 255
        if left > 0:
            mask[:, 0:left+overlap] = 255
        if right > 0:
            mask[:, new_w-right-overlap:new_w] = 255

        # 원본 영역 보호
        mask[top+overlap:top+h-overlap, left+overlap:left+w-overlap] = 0

        # 3단계: Stable Diffusion inpainting
        prompt = kwargs.get('prompt', self.sd_prompt)
        num_inference_steps = kwargs.get('num_inference_steps', 75)  # 높은 품질
        guidance_scale = kwargs.get('guidance_scale', 7.5)  # 자연스러운 생성
        strength = kwargs.get('strength', 0.9)  # 강한 생성력

        try:
            result = self._sd_inpainter.inpaint(
                image=canvas,
                mask=mask,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )

            logger.info("SD inpainting completed successfully")
            return Image.fromarray(result)

        except Exception as e:
            logger.error(f"SD inpainting failed: {e}")
            # 실패 시 원본 캔버스 반환
            return Image.fromarray(canvas)

    def extend_to_size(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int,
        position: str = "center"
    ) -> Image.Image:
        """
        이미지를 목표 크기로 확장합니다.

        Args:
            image: PIL Image
            target_width: 목표 너비
            target_height: 목표 높이
            position: 원본 이미지 위치 ('center', 'top-left', 'top-right', 'bottom-left', 'bottom-right')

        Returns:
            확장된 이미지
        """
        w, h = image.size

        if w >= target_width and h >= target_height:
            logger.warning("Image is already larger than target size")
            return image

        # 확장할 픽셀 계산
        extra_w = max(0, target_width - w)
        extra_h = max(0, target_height - h)

        if position == "center":
            left = extra_w // 2
            right = extra_w - left
            top = extra_h // 2
            bottom = extra_h - top
        elif position == "top-left":
            left, top = 0, 0
            right, bottom = extra_w, extra_h
        elif position == "top-right":
            left, top = extra_w, 0
            right, bottom = 0, extra_h
        elif position == "bottom-left":
            left, top = 0, extra_h
            right, bottom = extra_w, 0
        elif position == "bottom-right":
            left, top = extra_w, extra_h
            right, bottom = 0, 0
        else:
            raise ValueError(f"Unknown position: {position}")

        return self.extend_image(image, top, bottom, left, right)


def test_extend():
    """테스트 함수"""
    # 테스트 이미지 생성
    test_img = Image.new('RGB', (200, 200), color=(100, 150, 200))

    extender = ImageExtender(method="sd")
    result = extender.extend_image(test_img, 50, 50, 50, 50)

    print(f"Original size: {test_img.size}")
    print(f"Extended size: {result.size}")
    result.save("test_extended.png")
    print("Test image saved as test_extended.png")


if __name__ == "__main__":
    test_extend()
