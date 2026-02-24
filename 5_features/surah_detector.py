"""Surah/Ayah detection from STT output.

Takes Arabic text from speech recognition and identifies
which Quran verse it corresponds to.
"""

from .verse_matcher import VerseMatcher


class SurahDetector:
    """Detect which Quran verse corresponds to transcribed Arabic text.

    Wraps VerseMatcher with a simpler interface for the inference pipeline.
    """

    def __init__(self, quran_path: str = "./data/quran-uthmani.json"):
        self._matcher = VerseMatcher(quran_path)
        self._last_key: str | None = None

    def detect(self, arabic_text: str) -> dict | None:
        """Detect the Quran verse from Arabic text.

        Args:
            arabic_text: Transcribed Arabic text from STT.

        Returns:
            Dict with verse_key, confidence, arabic_text, surah, ayah.
            None if no match found above confidence threshold.
        """
        matches = self._matcher.match(
            arabic_text,
            previous_key=self._last_key,
            top_k=1,
        )

        if not matches:
            return None

        best = matches[0]
        surah, ayah = best["verse_key"].split(":")
        self._last_key = best["verse_key"]

        return {
            "verse_key": best["verse_key"],
            "surah": int(surah),
            "ayah": int(ayah),
            "confidence": best["confidence"],
            "arabic_text": best["arabic_text"],
        }

    def detect_multiple(self, arabic_text: str, top_k: int = 5) -> list[dict]:
        """Detect top-k matching verses.

        Args:
            arabic_text: Transcribed Arabic text.
            top_k: Number of results to return.

        Returns:
            List of match dicts sorted by confidence.
        """
        matches = self._matcher.match(
            arabic_text,
            previous_key=self._last_key,
            top_k=top_k,
        )

        results = []
        for m in matches:
            surah, ayah = m["verse_key"].split(":")
            results.append({
                "verse_key": m["verse_key"],
                "surah": int(surah),
                "ayah": int(ayah),
                "confidence": m["confidence"],
                "arabic_text": m["arabic_text"],
            })

        if results:
            self._last_key = results[0]["verse_key"]

        return results

    def reset(self):
        """Reset sequential tracking state."""
        self._last_key = None
