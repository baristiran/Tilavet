"""Trigram-based verse matching algorithm for Quran verse identification.

Uses a character trigram index for fast candidate selection, then
multi-metric scoring (Levenshtein, LCS, Jaccard) for precise matching.
"""

import importlib
import json
from pathlib import Path

normalize_mod = importlib.import_module("1_data_prep.normalize_text")
normalize_arabic = normalize_mod.normalize_arabic


def _char_trigrams(text: str) -> set[str]:
    """Generate character trigrams from text."""
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def _lcs_length(s1: str, s2: str) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0

    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]


class VerseMatcher:
    """Fast Quran verse matching using trigram indexing and multi-metric scoring.

    Attributes:
        verses: Dict mapping verse_key to normalized Arabic text.
        trigram_index: Inverted index from trigram to set of verse_keys.
    """

    # Scoring weights
    WEIGHT_LEVENSHTEIN = 0.40
    WEIGHT_LCS = 0.35
    WEIGHT_JACCARD = 0.25
    SUBSTRING_BONUS = 0.15
    SEQUENTIAL_BOOST = 0.10
    MIN_CONFIDENCE = 0.65

    def __init__(self, quran_path: str = "./data/quran-uthmani.json"):
        """Initialize matcher with Quran text.

        Args:
            quran_path: Path to quran-uthmani.json file.
        """
        self.verses: dict[str, str] = {}
        self.raw_verses: dict[str, str] = {}
        self.trigram_index: dict[str, set[str]] = {}

        self._load_quran(quran_path)
        self._build_index()

    def _load_quran(self, path: str):
        """Load Quran text and normalize for matching."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for surah_num, ayahs in data.items():
            for ayah_num, text in ayahs.items():
                key = f"{surah_num}:{ayah_num}"
                self.raw_verses[key] = text
                self.verses[key] = normalize_arabic(text)

    def _build_index(self):
        """Build inverted trigram index over all verses."""
        for key, text in self.verses.items():
            for trigram in _char_trigrams(text):
                if trigram not in self.trigram_index:
                    self.trigram_index[trigram] = set()
                self.trigram_index[trigram].add(key)

    def _get_candidates(self, text: str, top_k: int = 30) -> list[str]:
        """Select candidate verses using trigram overlap.

        Args:
            text: Normalized input text.
            top_k: Number of top candidates to return.

        Returns:
            List of verse_key strings sorted by trigram hit count.
        """
        input_trigrams = _char_trigrams(text)
        hits: dict[str, int] = {}

        for trigram in input_trigrams:
            for key in self.trigram_index.get(trigram, set()):
                hits[key] = hits.get(key, 0) + 1

        sorted_candidates = sorted(hits.items(), key=lambda x: x[1], reverse=True)
        return [key for key, _ in sorted_candidates[:top_k]]

    def _score(self, input_text: str, verse_text: str) -> float:
        """Compute multi-metric similarity score.

        Args:
            input_text: Normalized input text.
            verse_text: Normalized verse text.

        Returns:
            Similarity score between 0 and 1.
        """
        max_len = max(len(input_text), len(verse_text))
        if max_len == 0:
            return 0.0

        # Levenshtein similarity
        edit_dist = _levenshtein_distance(input_text, verse_text)
        lev_sim = 1.0 - (edit_dist / max_len)

        # LCS ratio
        lcs_len = _lcs_length(input_text, verse_text)
        lcs_ratio = lcs_len / max_len

        # Trigram Jaccard similarity
        trigrams_a = _char_trigrams(input_text)
        trigrams_b = _char_trigrams(verse_text)
        if trigrams_a or trigrams_b:
            jaccard = len(trigrams_a & trigrams_b) / len(trigrams_a | trigrams_b)
        else:
            jaccard = 0.0

        score = (
            self.WEIGHT_LEVENSHTEIN * lev_sim
            + self.WEIGHT_LCS * lcs_ratio
            + self.WEIGHT_JACCARD * jaccard
        )

        # Substring containment bonus
        if input_text in verse_text or verse_text in input_text:
            score += self.SUBSTRING_BONUS

        return min(score, 1.0)

    def match(
        self,
        text: str,
        previous_key: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Find the best matching verses for the given Arabic text.

        Args:
            text: Arabic text (raw or normalized).
            previous_key: Previous verse key for sequential boost.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: verse_key, confidence, arabic_text.
        """
        normalized = normalize_arabic(text)
        if not normalized.strip():
            return []

        candidates = self._get_candidates(normalized)
        if not candidates:
            return []

        scored = []
        for key in candidates:
            score = self._score(normalized, self.verses[key])

            # Sequential verse boost
            if previous_key and self._is_sequential(previous_key, key):
                score += self.SEQUENTIAL_BOOST

            if score >= self.MIN_CONFIDENCE:
                scored.append({
                    "verse_key": key,
                    "confidence": round(min(score, 1.0), 4),
                    "arabic_text": self.raw_verses[key],
                })

        scored.sort(key=lambda x: x["confidence"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _is_sequential(prev_key: str, curr_key: str) -> bool:
        """Check if curr_key is the verse immediately after prev_key."""
        try:
            ps, pa = prev_key.split(":")
            cs, ca = curr_key.split(":")
            return ps == cs and int(ca) == int(pa) + 1
        except (ValueError, AttributeError):
            return False
