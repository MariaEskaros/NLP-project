import re

def normalize_arabic(text):
    """
    Normalize Arabic text:
    - Remove timestamps
    - Remove diacritics
    - Normalize letters: أ إ آ -> ا, ى -> ي, ة -> ه, remove tatweel
    - Remove special characters (except Arabic letters and English letters/numbers)
    - Collapse repeated letters (elongation)
    """
    # Remove timestamps
    text = re.sub(r"\d+\.\d+:", "", text)

    # Remove diacritics
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)

    # Normalize letters
    text = re.sub(r"[إأآ]", "ا", text)  # Alef variants
    text = re.sub(r"ى", "ي", text)     # Ya variant
    text = re.sub(r"ة", "ه", text)     # Ta marbuta
    text = re.sub(r"[ؤ]", "و", text)   # Waw-Hamza variants
    text = re.sub(r"[ئ]", "ي", text)   # Waw-Hamza variants
    text = re.sub(r"ـ", "", text)      # Tatweel / Kashida

    # Remove special characters (except Arabic letters, English letters, numbers, spaces)
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s]", "", text)

    # Collapse repeated letters (elongation)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Strip extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_horof_el_jar(text):
    """
    Remove common Arabic prepositions (حروف الجر) from text.
    """
    horof_el_jar = [
        "من", "إلى", "الى", "عن", "علي", "في", "ب", "ك", "ل", "حتى", "منذ", "مذ", "رب", "خلا", "عدا", "حاشا"
    ]

    pattern = r"\b(?:" + "|".join(re.escape(word) for word in horof_el_jar) + r")\b"
    text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# List of حروف الجر (prepositions)
prepositions = [
    "من", "الى", "عن", "علي", "في",   
]

# We will remove them when they appear at the start of a word
attached_prepositions = ["ب", "ل", "ك"]

def remove_horof_el_jar(text):
    pattern_standalone = r'\b(' + '|'.join(prepositions) + r')\b'
    text = re.sub(pattern_standalone, '', text)
    
    # Example: "بالكتاب" → "كتاب"
    text = re.sub(r'\b[' + ''.join(attached_prepositions) + r'](\w+)', r'\1', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text