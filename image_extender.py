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
from image_analyzer import get_image_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageExtender:
    """Stable Diffusion을 사용하여 이미지를 배경에 맞춰 확장하는 클래스"""

    def __init__(self, method: str = "sd", sd_prompt: str = None, use_auto_prompt: bool = True):
        """
        Args:
            method: 확장 방법 ('sd' - Stable Diffusion)
            sd_prompt: Stable Diffusion 프롬프트 (None이면 자동 생성)
            use_auto_prompt: True면 이미지 분석으로 자동 프롬프트 생성
        """
        if method != "sd":
            logger.warning(f"Method '{method}' not supported. Using 'sd' instead.")
            method = "sd"

        self.method = method
        self.use_auto_prompt = use_auto_prompt
        # 텍스트/글씨 방지를 위한 개선된 프롬프트 (fallback용)
        self.sd_prompt = sd_prompt or "seamless background extension, natural scenery, photorealistic, high quality, detailed textures, consistent lighting, no text, no watermark, no letters, no numbers, no symbols"

        # SD 인페인터 초기화
        try:
            logger.info("Loading Stable Diffusion model (this may take a while on first run)...")
            self._sd_inpainter = get_sd_inpainter()
            logger.info(f"ImageExtender initialized with Stable Diffusion inpainting")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion: {e}")
            raise

        # 이미지 분석기 초기화 (지연 로딩)
        self._image_analyzer = None
        if self.use_auto_prompt:
            try:
                logger.info("Loading image analyzer for auto-prompt generation...")
                self._image_analyzer = get_image_analyzer()
                logger.info("Image analyzer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load image analyzer: {e}. Will use default prompts.")
                self.use_auto_prompt = False

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

        # 고해상도 이미지 처리 최적화
        max_dimension = max(new_h, new_w)

        # SD 최적 크기: 512~768px (이보다 크면 다운스케일 후 처리)
        target_max_dim = 768

        if max_dimension > target_max_dim:
            logger.info(f"High-res image detected ({max_dimension}px). Using downscale strategy for better quality and speed.")
            return self._extend_with_downscale(image, top, bottom, left, right, target_max_dim, **kwargs)
        else:
            return self._extend_direct(image, top, bottom, left, right, **kwargs)

    def _extend_direct(
        self,
        image: Image.Image,
        top: int, bottom: int, left: int, right: int,
        **kwargs
    ) -> Image.Image:
        """직접 SD 인페인팅 (중저해상도)"""

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

        # 3단계: 프롬프트 생성 (자동 or 수동)
        if 'prompt' not in kwargs and self.use_auto_prompt and self._image_analyzer:
            try:
                logger.info("Analyzing image to generate context-aware prompt...")
                auto_prompt, auto_negative = self._image_analyzer.generate_prompts(
                    image,
                    for_extension=True
                )
                prompt = auto_prompt
                negative_prompt = auto_negative
                logger.info(f"Auto-generated prompt: {prompt}")
            except Exception as e:
                logger.warning(f"Auto-prompt generation failed: {e}. Using default prompt.")
                prompt = self.sd_prompt
                negative_prompt = kwargs.get('negative_prompt', None)
        else:
            prompt = kwargs.get('prompt', self.sd_prompt)
            negative_prompt = kwargs.get('negative_prompt', None)

        num_inference_steps = kwargs.get('num_inference_steps', 75)  # 높은 품질
        guidance_scale = kwargs.get('guidance_scale', 8.0)  # 프롬프트 충실도 증가 (텍스트 방지)
        strength = kwargs.get('strength', 0.95)  # 더 강한 생성력으로 원본 배경 충실하게 확장

        try:
            inpaint_kwargs = {
                'image': canvas,
                'mask': mask,
                'prompt': prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'strength': strength
            }

            # negative_prompt가 있으면 추가
            if negative_prompt:
                inpaint_kwargs['negative_prompt'] = negative_prompt

            result = self._sd_inpainter.inpaint(**inpaint_kwargs)

            logger.info("SD inpainting completed successfully")
            return Image.fromarray(result)

        except Exception as e:
            logger.error(f"SD inpainting failed: {e}")
            # 실패 시 원본 캔버스 반환
            return Image.fromarray(canvas)

    def _extend_with_downscale(
        self,
        image: Image.Image,
        top: int, bottom: int, left: int, right: int,
        target_max_dim: int,
        **kwargs
    ) -> Image.Image:
        """고해상도 이미지를 다운스케일 후 처리하여 품질과 속도 개선"""

        w, h = image.size
        new_h = h + top + bottom
        new_w = w + left + right

        # 스케일 비율 계산
        scale = target_max_dim / max(new_h, new_w)

        logger.info(f"Original size: {w}x{h}, Target canvas: {new_w}x{new_h}")
        logger.info(f"Scale factor: {scale:.3f}")

        # 1단계: 이미지와 확장 영역을 비례적으로 다운스케일
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        scaled_top = int(top * scale)
        scaled_bottom = int(bottom * scale)
        scaled_left = int(left * scale)
        scaled_right = int(right * scale)

        # 고품질 리샘플링으로 다운스케일
        small_image = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        logger.info(f"Downscaled to: {scaled_w}x{scaled_h} (expand: t={scaled_top}, b={scaled_bottom}, l={scaled_left}, r={scaled_right})")

        # 2단계: 작은 크기로 SD 인페인팅 수행 (빠르고 품질 좋음)
        small_extended = self._extend_direct(
            small_image,
            scaled_top, scaled_bottom, scaled_left, scaled_right,
            **kwargs
        )

        # 3단계: 원본 크기로 업스케일 (고품질 리샘플링)
        final_result = small_extended.resize((new_w, new_h), Image.Resampling.LANCZOS)

        logger.info(f"Upscaled back to: {new_w}x{new_h}")

        return final_result

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
