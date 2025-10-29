"""
이미지 분석 모듈
BLIP을 사용하여 이미지를 분석하고 적절한 SD 프롬프트를 생성합니다.
"""

import torch
from PIL import Image
import logging
from typing import Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    BLIP 모델을 사용하여 이미지를 분석하고 SD 프롬프트를 생성하는 클래스
    """

    def __init__(self, device: str = "auto"):
        """
        Args:
            device: 'cpu', 'cuda', 'auto'
        """
        self.device = self._setup_device(device)
        self.processor = None
        self.model = None
        self.model_available = False

        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            logger.info("Loading BLIP model for image analysis...")

            # BLIP 모델 로드 (작고 빠른 버전)
            model_name = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.model_available = True
            logger.info(f"BLIP model loaded on {self.device}")

        except ImportError as e:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise

    def _setup_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(device_str)
        return torch.device(device)

    def analyze_image(self, image: Image.Image) -> str:
        """
        이미지를 분석하여 설명을 생성합니다.

        Args:
            image: PIL Image

        Returns:
            이미지 설명 텍스트
        """
        if not self.model_available:
            raise RuntimeError("BLIP model not available")

        try:
            # 이미지 전처리
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            # 이미지 캡셔닝 (조건부 생성)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=3,
                    early_stopping=True
                )

            # 텍스트 디코딩
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Image analysis result: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise

    def generate_prompts(
        self,
        image: Image.Image,
        for_extension: bool = True
    ) -> Tuple[str, str]:
        """
        이미지를 분석하여 SD용 프롬프트와 네거티브 프롬프트를 생성합니다.

        Args:
            image: PIL Image
            for_extension: True면 배경 확장용, False면 일반 inpainting용

        Returns:
            (prompt, negative_prompt) 튜플
        """
        # 이미지 분석
        caption = self.analyze_image(image)

        # 캡션을 기반으로 프롬프트 생성
        if for_extension:
            # 배경 확장용 프롬프트
            prompt = f"{caption}, seamless background extension, natural continuation, photorealistic, high quality, detailed textures, consistent lighting, no text, no watermark, no letters, no numbers, no symbols"
        else:
            # 일반 inpainting용 프롬프트
            prompt = f"{caption}, natural, photorealistic, high quality, detailed, no text, no watermark"

        # 네거티브 프롬프트 (텍스트 방지 강화)
        negative_prompt = "text, watermark, letters, words, numbers, symbols, characters, writing, typography, signature, logo, blurry, low quality, distorted, artifacts, cropped, out of frame, jpeg artifacts, gradient layers, visible seams, hard edges"

        logger.info(f"Generated prompt: {prompt}")
        logger.info(f"Generated negative prompt: {negative_prompt}")

        return prompt, negative_prompt


# 전역 분석기 인스턴스 (지연 로딩)
_global_image_analyzer: Optional[ImageAnalyzer] = None


def get_image_analyzer(device: str = "auto") -> ImageAnalyzer:
    """
    전역 이미지 분석기 인스턴스 가져오기 (싱글톤)

    Args:
        device: 디바이스

    Returns:
        ImageAnalyzer 인스턴스
    """
    global _global_image_analyzer

    if _global_image_analyzer is None:
        _global_image_analyzer = ImageAnalyzer(device)

    return _global_image_analyzer


if __name__ == "__main__":
    # 테스트
    logger.info("Testing image analyzer...")

    # 테스트 이미지 생성
    test_image = Image.new('RGB', (512, 512), color=(135, 206, 235))  # 하늘색

    try:
        analyzer = get_image_analyzer()
        caption = analyzer.analyze_image(test_image)
        prompt, negative_prompt = analyzer.generate_prompts(test_image)

        logger.info(f"Caption: {caption}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Negative Prompt: {negative_prompt}")
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
