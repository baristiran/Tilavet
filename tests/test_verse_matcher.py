"""Tests for the verse matching algorithm."""

import importlib
import json
import tempfile
from pathlib import Path

import pytest

VerseMatcher = importlib.import_module("5_features.verse_matcher").VerseMatcher


@pytest.fixture
def quran_file(tmp_path):
    """Create a minimal Quran text file for testing."""
    quran_data = {
        "1": {
            "1": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
            "2": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
            "3": "الرَّحْمَنِ الرَّحِيمِ",
            "4": "مَالِكِ يَوْمِ الدِّينِ",
            "5": "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
            "6": "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
            "7": "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
        },
        "2": {
            "255": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ ۚ لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ",
        },
    }

    file_path = tmp_path / "quran-uthmani.json"
    file_path.write_text(json.dumps(quran_data, ensure_ascii=False), encoding="utf-8")
    return str(file_path)


class TestVerseMatcher:
    def test_exact_match(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("بسم الله الرحمن الرحيم")
        assert len(results) > 0
        assert results[0]["verse_key"] == "1:1"

    def test_diacritized_match(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ")
        assert len(results) > 0
        assert results[0]["verse_key"] == "1:1"

    def test_partial_match(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("الحمد لله رب العالمين")
        assert len(results) > 0
        assert results[0]["verse_key"] == "1:2"

    def test_no_match_empty(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("")
        assert len(results) == 0

    def test_no_match_garbage(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("xxxyyy")
        assert len(results) == 0

    def test_sequential_boost(self, quran_file):
        matcher = VerseMatcher(quran_file)
        # Match first verse
        results1 = matcher.match("بسم الله الرحمن الرحيم")
        assert results1[0]["verse_key"] == "1:1"

        # Match second verse with sequential boost
        results2 = matcher.match("الحمد لله رب العالمين", previous_key="1:1")
        assert results2[0]["verse_key"] == "1:2"

    def test_ayat_al_kursi(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("الله لا اله الا هو الحي القيوم")
        assert len(results) > 0
        assert results[0]["verse_key"] == "2:255"

    def test_confidence_threshold(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("بسم الله")  # Very short, partial
        # Should still find some results if threshold is met
        for r in results:
            assert r["confidence"] >= matcher.MIN_CONFIDENCE

    def test_returns_arabic_text(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("بسم الله الرحمن الرحيم")
        assert len(results) > 0
        assert "arabic_text" in results[0]
        assert len(results[0]["arabic_text"]) > 0

    def test_top_k(self, quran_file):
        matcher = VerseMatcher(quran_file)
        results = matcher.match("الرحمن الرحيم", top_k=3)
        assert len(results) <= 3
