"""
Stable Diffusion Inpainting wrapper
High-quality AI-powered image inpainting using Stable Diffusion
"""

import torch
import numpy as np
from PIL import Image
import logging
from typing import Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDInpainter:
    """
    Stable Diffusion 기반 인페인터
    최고 품질의 AI 인페인팅 제공
    """

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-inpainting", device: str = "auto"):
        """
        Args:
            model_id: Hugging Face 모델 ID
            device: 'cpu', 'cuda', 'auto'
        """
        self.model_id = model_id
        self.device = self._setup_device(device)
        self.pipe = None
        self.model_available = False

        try:
            from diffusers import StableDiffusionInpaintPipeline

            logger.info(f"Loading Stable Diffusion model: {model_id}")
            logger.info("First run will download ~5GB model (one-time only)")

            # 모델 로드 (품질 우선을 위해 float32 사용)
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # 높은 품질을 위해 float32 사용
                safety_checker=None,  # 빠른 처리를 위해 비활성화
                requires_safety_checker=False
            )

            self.pipe = self.pipe.to(self.device)

            # 메모리 최적화
            if self.device.type == "cpu":
                logger.info("Using CPU with float32 - this will be slower but high quality")
            else:
                # GPU에서 메모리 최적화
                self.pipe.enable_attention_slicing()
                logger.info("GPU optimizations enabled with float32 for highest quality")

            self.model_available = True
            logger.info(f"Stable Diffusion model loaded on {self.device}")

        except ImportError as e:
            logger.error("diffusers not installed. Install with: pip install diffusers transformers accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            raise

    def _setup_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(device_str)
        return torch.device(device)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "seamless background extension, natural scenery, photorealistic, high quality, detailed textures, consistent lighting, no text, no watermark, no letters, no numbers, no symbols",
        negative_prompt: str = "text, watermark, letters, words, numbers, symbols, characters, writing, typography, signature, blurry, low quality, distorted, artifacts, cropped, out of frame, jpeg artifacts, gradient layers, visible seams, hard edges",
        num_inference_steps: int = 75,  # 높은 품질
        guidance_scale: float = 8.0,  # 프롬프트 충실도 증가 (텍스트 방지)
        strength: float = 0.95  # 더 강한 생성력
    ) -> np.ndarray:
        """
        Stable Diffusion 인페인팅 수행

        Args:
            image: RGB 이미지 (H, W, 3), uint8, 0-255
            mask: 마스크 (H, W), uint8, 255 = 인페인트할 영역
            prompt: 생성할 내용 설명
            negative_prompt: 피할 내용
            num_inference_steps: 추론 스텝 수 (높을수록 품질 좋지만 느림)
            guidance_scale: 프롬프트 가이던스 강도
            strength: 인페인팅 강도 (0.0-1.0)

        Returns:
            인페인팅된 이미지 (H, W, 3), uint8
        """
        if not self.model_available or self.pipe is None:
            raise RuntimeError("Stable Diffusion model not available")

        try:
            # NumPy to PIL
            image_pil = Image.fromarray(image).convert("RGB")
            mask_pil = Image.fromarray(mask).convert("L")

            # 크기 확인 (SD는 8의 배수 크기 선호)
            w, h = image_pil.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8

            if (new_w, new_h) != (w, h):
                logger.info(f"Resizing from {w}x{h} to {new_w}x{new_h} (SD requires multiple of 8)")
                image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
                mask_pil = mask_pil.resize((new_w, new_h), Image.LANCZOS)
                original_size = (w, h)
            else:
                original_size = None

            logger.info(f"Running SD inpainting (steps={num_inference_steps})...")

            # SD 인페인팅 실행
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                height=new_h,
                width=new_w
            ).images[0]

            # 원래 크기로 복원
            if original_size:
                result = result.resize(original_size, Image.LANCZOS)

            # PIL to NumPy
            result_array = np.array(result)

            logger.info("SD inpainting completed")
            return result_array

        except Exception as e:
            logger.error(f"SD inpainting failed: {e}")
            raise


# 전역 인페인터 인스턴스 (지연 로딩)
_global_sd_inpainter: Optional[SDInpainter] = None


def get_sd_inpainter(
    model_id: str = "stabilityai/stable-diffusion-2-inpainting",
    device: str = "auto"
) -> SDInpainter:
    """
    전역 SD 인페인터 인스턴스 가져오기 (싱글톤)

    Args:
        model_id: Hugging Face 모델 ID
        device: 디바이스

    Returns:
        SDInpainter 인스턴스
    """
    global _global_sd_inpainter

    if _global_sd_inpainter is None:
        _global_sd_inpainter = SDInpainter(model_id, device)

    return _global_sd_inpainter


if __name__ == "__main__":
    # 테스트
    logger.info("Testing SD inpainter...")

    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[200:300, 200:300] = 255  # 중앙 영역

    try:
        inpainter = get_sd_inpainter()
        result = inpainter.inpaint(test_image, test_mask)

        logger.info(f"Input shape: {test_image.shape}")
        logger.info(f"Output shape: {result.shape}")
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
