"""
Shared utilities for HamsAI scripts.
Import from here instead of copy-pasting into every script.
"""

import re


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for indexing and retrieval:
    - Remove diacritics (tashkeel)
    - Normalize alef variants (أإآٱ → ا)
    - Normalize ya (ى → ي)
    - Normalize teh marbuta (ة → ه)
    - Remove tatweel (ـ)
    - Convert Eastern Arabic numerals to Western (٠١٢٣٤٥٦٧٨٩ → 0-9)
    """
    text = re.sub(r"[ؐ-ًؚ-ٟ]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    for eastern, western in zip("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"):
        text = text.replace(eastern, western)
    return text.strip()


def normalize_text(text: str, language: str) -> str:
    """Normalize text based on detected language."""
    if language in ("ar", "mixed"):
        text = normalize_arabic(text)
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """Detect language from Arabic/English character ratio."""
    arabic_chars  = len(re.findall(r"[؀-ۿ]", text))
    english_chars = len(re.findall(r"[a-zA-Z]", text))
    total = arabic_chars + english_chars
    if total == 0:
        return "en"
    ratio = arabic_chars / total
    if ratio > 0.7:
        return "ar"
    if ratio < 0.3:
        return "en"
    return "mixed"
