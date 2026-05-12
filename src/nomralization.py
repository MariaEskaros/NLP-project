import re

def normalize_arabic_for_rag(text):
    """
    MS3-safe Arabic normalization for RAG:
    - Remove timestamps
    - Remove diacritics
    - Normalize Alef variants
    - Normalizeى to ي
    - Remove tatweel
    - Preserve punctuation
    - Preserve English tokens
    - Preserve dialectal words
    - Preserve prepositions and word structure
    """

    # Remove timestamps like 12.5: or 00:01:23
    text = re.sub(r"\d+\.\d+:", " ", text)
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", " ", text)

    # Remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize Alef variants
    text = re.sub(r"[إأآٱ]", "ا", text)

    # Normalize Alef Maqsura
    text = re.sub(r"ى", "ي", text)

    # Remove tatweel
    text = re.sub(r"ـ", "", text)

    # Keep punctuation, Arabic, English, numbers
    # Only collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text