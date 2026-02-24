"""Multi-language Quran translation engine.

Loads translation files lazily and caches them in memory.
Supports 108+ languages via fawazahmed0/quran-api data.
"""

import json
from pathlib import Path


class TranslationEngine:
    """Lazy-loading, caching translation engine for Quran verses.

    Translations are loaded from JSON files on first access per language
    and cached in memory for subsequent requests.

    Attributes:
        translations_dir: Path to directory containing translation JSON files.
        cache: Dict of loaded translations {lang_code: {surah: {ayah: text}}}.
    """

    def __init__(self, translations_dir: str = "./data/translations"):
        self.translations_dir = Path(translations_dir)
        self._cache: dict[str, dict] = {}

    def _load_language(self, edition_name: str) -> dict | None:
        """Load a translation file into cache.

        Args:
            edition_name: Edition filename without extension.

        Returns:
            Translation dict or None if file not found.
        """
        if edition_name in self._cache:
            return self._cache[edition_name]

        file_path = self.translations_dir / f"{edition_name}.json"
        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text(encoding="utf-8"))
        self._cache[edition_name] = data
        return data

    def get_translation(
        self,
        verse_key: str,
        edition_name: str,
    ) -> str | None:
        """Get translation for a specific verse in a specific edition.

        Args:
            verse_key: Verse key in "surah:ayah" format (e.g., "1:1").
            edition_name: Edition name (e.g., "tur-dipisilam").

        Returns:
            Translation text or None if not available.
        """
        data = self._load_language(edition_name)
        if data is None:
            return None

        parts = verse_key.split(":")
        if len(parts) != 2:
            return None

        surah, ayah = parts
        return data.get(surah, {}).get(ayah)

    def get_translations(
        self,
        verse_key: str,
        editions: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Get translations for a verse across multiple languages.

        Args:
            verse_key: Verse key in "surah:ayah" format.
            editions: Dict mapping language codes to edition names.
                      If None, uses all cached languages.

        Returns:
            Dict mapping language codes to translation text.
        """
        if editions is None:
            editions = {name: name for name in self._cache}

        result = {}
        for lang_code, edition_name in editions.items():
            text = self.get_translation(verse_key, edition_name)
            if text:
                result[lang_code] = text

        return result

    def get_available_editions(self) -> list[str]:
        """List all available translation editions on disk.

        Returns:
            List of edition names (without .json extension).
        """
        if not self.translations_dir.exists():
            return []
        return sorted(
            f.stem for f in self.translations_dir.glob("*.json")
        )

    def preload(self, edition_names: list[str] | None = None) -> int:
        """Preload translation files into cache.

        Args:
            edition_names: List of editions to preload. None = all available.

        Returns:
            Number of editions loaded.
        """
        if edition_names is None:
            edition_names = self.get_available_editions()

        loaded = 0
        for name in edition_names:
            if self._load_language(name) is not None:
                loaded += 1

        return loaded

    @property
    def loaded_count(self) -> int:
        """Number of currently loaded translations."""
        return len(self._cache)
