"""Arabic text normalization utilities for Quran STT.

Provides normalization functions for preprocessing Quranic Arabic text,
used in both training (WER computation) and inference (post-processing).
"""

import re

# Arabic diacritical marks (tashkeel/harakat)
FATHATAN = "\u064b"
DAMMATAN = "\u064c"
KASRATAN = "\u064d"
FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
MADDAH = "\u0653"
HAMZA_ABOVE = "\u0654"
HAMZA_BELOW = "\u0655"
SUPERSCRIPT_ALEF = "\u0670"
ALEF_WASLA = "\u0671"

DIACRITICS = (
    FATHATAN + DAMMATAN + KASRATAN + FATHA + DAMMA + KASRA +
    SHADDA + SUKUN + MADDAH + HAMZA_ABOVE + HAMZA_BELOW + SUPERSCRIPT_ALEF
)

DIACRITICS_PATTERN = re.compile(f"[{DIACRITICS}]")

# Tatweel (kashida) - decorative elongation
TATWEEL = "\u0640"

# Quran-specific marks (pause markers, sajda, etc.)
QURAN_MARKS = "\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06dd\u06de\u06df\u06e0\u06e1\u06e2\u06e3\u06e4\u06e5\u06e6\u06e7\u06e8\u06e9\u06ea\u06eb\u06ec\u06ed"
QURAN_MARKS_PATTERN = re.compile(f"[{QURAN_MARKS}]")

# Alef variants
ALEF = "\u0627"  # ا
ALEF_MADDA = "\u0622"  # آ
ALEF_HAMZA_ABOVE = "\u0623"  # أ
ALEF_HAMZA_BELOW = "\u0625"  # إ
ALEF_WASLA_CHAR = "\u0671"  # ٱ

# Hamza variants
HAMZA = "\u0621"  # ء
WAW_HAMZA = "\u0624"  # ؤ
YEH_HAMZA = "\u0626"  # ئ

# Ta marbuta / Ha
TA_MARBUTA = "\u0629"  # ة
HA = "\u0647"  # ه

# Yeh variants
ALEF_MAKSURA = "\u0649"  # ى
YEH = "\u064a"  # ي


def remove_diacritics(text: str) -> str:
    """Remove all Arabic diacritical marks (tashkeel/harakat).

    Removes fatha, damma, kasra, tanwin, shadda, sukun, etc.

    >>> remove_diacritics("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ")
    'بسم الله الرحمن الرحيم'
    """
    return DIACRITICS_PATTERN.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel/kashida (decorative elongation character).

    >>> remove_tatweel("اللـــه")
    'الله'
    """
    return text.replace(TATWEEL, "")


def remove_quran_marks(text: str) -> str:
    """Remove Quran-specific marks (pause markers, sajda signs, etc.)."""
    return QURAN_MARKS_PATTERN.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalize all alef variants to plain alef (ا).

    Maps: آ أ إ ٱ → ا

    >>> normalize_alef("أحمد إبراهيم آمن")
    'احمد ابراهيم امن'
    """
    text = text.replace(ALEF_MADDA, ALEF)
    text = text.replace(ALEF_HAMZA_ABOVE, ALEF)
    text = text.replace(ALEF_HAMZA_BELOW, ALEF)
    text = text.replace(ALEF_WASLA_CHAR, ALEF)
    return text


def normalize_hamza(text: str) -> str:
    """Normalize hamza-on-carrier variants.

    Maps: ؤ → و, ئ → ي

    >>> normalize_hamza("مؤمن")
    'مومن'
    """
    text = text.replace(WAW_HAMZA, "\u0648")  # ؤ → و
    text = text.replace(YEH_HAMZA, YEH)  # ئ → ي
    return text


def normalize_ta_marbuta(text: str) -> str:
    """Normalize ta marbuta to ha.

    Maps: ة → ه

    >>> normalize_ta_marbuta("رحمة")
    'رحمه'
    """
    return text.replace(TA_MARBUTA, HA)


def normalize_alef_maksura(text: str) -> str:
    """Normalize alef maksura to yeh.

    Maps: ى → ي

    >>> normalize_alef_maksura("على")
    'علي'
    """
    return text.replace(ALEF_MAKSURA, YEH)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_arabic(text: str) -> str:
    """Full normalization pipeline for Quran Arabic text.

    Applies all normalizations in order:
    1. Remove Quran-specific marks
    2. Remove diacritics
    3. Remove tatweel
    4. Normalize alef variants
    5. Normalize hamza variants
    6. Normalize ta marbuta
    7. Normalize alef maksura
    8. Collapse whitespace

    >>> normalize_arabic("بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ")
    'بسم الله الرحمن الرحيم'
    """
    text = remove_quran_marks(text)
    text = remove_diacritics(text)
    text = remove_tatweel(text)
    text = normalize_alef(text)
    text = normalize_hamza(text)
    text = normalize_ta_marbuta(text)
    text = normalize_alef_maksura(text)
    text = normalize_whitespace(text)
    return text


def normalize_for_wer(reference: str, hypothesis: str) -> tuple[str, str]:
    """Prepare reference and hypothesis texts for WER comparison.

    Applies full normalization to both strings for consistent comparison.

    Returns:
        Tuple of (normalized_reference, normalized_hypothesis)
    """
    return normalize_arabic(reference), normalize_arabic(hypothesis)
