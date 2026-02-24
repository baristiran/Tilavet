"""Tests for the translation engine."""

import importlib
import json
import tempfile
from pathlib import Path

import pytest

TranslationEngine = importlib.import_module("5_features.translation_engine").TranslationEngine


@pytest.fixture
def translations_dir(tmp_path):
    """Create a temporary translations directory with test data."""
    tr_data = {
        "1": {"1": "Rahman ve Rahim olan Allah'in adiyla", "2": "Alemlerin Rabbi Allah'a hamd olsun"},
        "2": {"255": "Allah, kendisinden baska ilah olmayan..."},
    }
    en_data = {
        "1": {"1": "In the name of God, the Most Gracious, the Most Merciful", "2": "Praise be to God, Lord of the Worlds"},
        "2": {"255": "God, there is no deity but Him..."},
    }

    (tmp_path / "tur-dipisilam.json").write_text(
        json.dumps(tr_data, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "eng-abdulhye.json").write_text(
        json.dumps(en_data, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


class TestTranslationEngine:
    def test_get_translation(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        result = engine.get_translation("1:1", "tur-dipisilam")
        assert result is not None
        assert "Allah" in result

    def test_get_translation_not_found(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        result = engine.get_translation("999:999", "tur-dipisilam")
        assert result is None

    def test_get_translation_missing_language(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        result = engine.get_translation("1:1", "nonexistent-edition")
        assert result is None

    def test_get_translations_multiple(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        editions = {"tr": "tur-dipisilam", "en": "eng-abdulhye"}
        result = engine.get_translations("1:1", editions)
        assert "tr" in result
        assert "en" in result

    def test_get_available_editions(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        editions = engine.get_available_editions()
        assert "tur-dipisilam" in editions
        assert "eng-abdulhye" in editions

    def test_lazy_loading(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        assert engine.loaded_count == 0
        engine.get_translation("1:1", "tur-dipisilam")
        assert engine.loaded_count == 1

    def test_preload(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        loaded = engine.preload()
        assert loaded == 2
        assert engine.loaded_count == 2

    def test_invalid_verse_key_format(self, translations_dir):
        engine = TranslationEngine(str(translations_dir))
        result = engine.get_translation("invalid", "tur-dipisilam")
        assert result is None

    def test_empty_dir(self, tmp_path):
        engine = TranslationEngine(str(tmp_path))
        assert engine.get_available_editions() == []
