"""
Multi-language Prompt Translator
Automatically detects and translates Korean, Japanese, and Chinese prompts to English for Stable Diffusion
"""

import logging
import re
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTranslator:
    """
    프롬프트 번역기
    한글, 일본어, 중국어 → 영어 자동 번역
    """

    def __init__(self):
        """번역기 초기화"""
        self.translator = None
        self._init_translator()

    def _init_translator(self):
        """deep-translator 초기화"""
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source='auto', target='en')
            logger.info("Translator initialized successfully")
        except ImportError:
            logger.warning("deep-translator not installed. Install with: pip install deep-translator")
            self.translator = None
        except Exception as e:
            logger.warning(f"Failed to initialize translator: {e}")
            self.translator = None

    def detect_language(self, text: str) -> str:
        """
        텍스트 언어 감지

        Args:
            text: 입력 텍스트

        Returns:
            'ko' (한글), 'ja' (일본어), 'zh' (중국어), 'en' (영어), 'unknown'
        """
        # 한글 감지 (한글 유니코드 범위: AC00-D7A3, 1100-11FF)
        if re.search(r'[\uAC00-\uD7A3\u1100-\u11FF]', text):
            return 'ko'

        # 일본어 감지 (히라가나, 가타카나, 한자)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
            # 중국어와 구분하기 위해 히라가나/가타카나 확인
            if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
                return 'ja'
            # 한자만 있으면 중국어로 간주
            return 'zh'

        # 중국어 감지 (간체/번체 한자)
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'zh'

        # ASCII 문자가 주를 이루면 영어
        if re.search(r'^[a-zA-Z0-9\s,.\-!?]+$', text.strip()):
            return 'en'

        return 'unknown'

    def translate_to_english(self, prompt: str) -> str:
        """
        프롬프트를 영어로 번역

        Args:
            prompt: 입력 프롬프트 (한글, 일본어, 중국어, 영어)

        Returns:
            영어로 번역된 프롬프트
        """
        if not prompt or not prompt.strip():
            return prompt

        # 언어 감지
        lang = self.detect_language(prompt)

        # 이미 영어면 그대로 반환
        if lang == 'en':
            logger.info(f"Prompt is already in English: {prompt}")
            return prompt

        # 번역기가 없으면 원본 반환
        if self.translator is None:
            logger.warning(f"Translator not available. Using original prompt: {prompt}")
            return prompt

        try:
            # 번역 실행
            translated = self.translator.translate(prompt)
            logger.info(f"Translated [{lang}] '{prompt}' → [en] '{translated}'")
            return translated

        except Exception as e:
            logger.error(f"Translation failed: {e}. Using original prompt: {prompt}")
            return prompt

    def translate_with_fallback(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None
    ) -> tuple[str, Optional[str]]:
        """
        프롬프트와 네거티브 프롬프트를 번역 (fallback 포함)

        Args:
            prompt: 메인 프롬프트
            negative_prompt: 네거티브 프롬프트 (선택)

        Returns:
            (번역된 프롬프트, 번역된 네거티브 프롬프트)
        """
        translated_prompt = self.translate_to_english(prompt)

        if negative_prompt:
            translated_negative = self.translate_to_english(negative_prompt)
        else:
            translated_negative = negative_prompt

        return translated_prompt, translated_negative


# 전역 번역기 인스턴스 (싱글톤)
_global_translator: Optional[PromptTranslator] = None


def get_translator() -> PromptTranslator:
    """
    전역 번역기 인스턴스 가져오기

    Returns:
        PromptTranslator 인스턴스
    """
    global _global_translator

    if _global_translator is None:
        _global_translator = PromptTranslator()

    return _global_translator


def translate_prompt(prompt: str) -> str:
    """
    간편 번역 함수

    Args:
        prompt: 입력 프롬프트

    Returns:
        영어로 번역된 프롬프트
    """
    translator = get_translator()
    return translator.translate_to_english(prompt)


if __name__ == "__main__":
    # 테스트
    translator = get_translator()

    test_prompts = [
        "beach sunset, ocean waves",  # 영어
        "따뜻한 카페 인테리어, 아늑한 분위기",  # 한글
        "温かいカフェのインテリア、居心地の良い雰囲気",  # 일본어
        "温暖的咖啡馆室内装饰，舒适的氛围",  # 중국어
    ]

    print("=== Prompt Translation Test ===\n")

    for prompt in test_prompts:
        lang = translator.detect_language(prompt)
        translated = translator.translate_to_english(prompt)
        print(f"Language: {lang}")
        print(f"Original:   {prompt}")
        print(f"Translated: {translated}")
        print()
