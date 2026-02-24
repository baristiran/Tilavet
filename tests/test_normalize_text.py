"""Tests for Arabic text normalization."""

import importlib

normalize = importlib.import_module("1_data_prep.normalize_text")

remove_diacritics = normalize.remove_diacritics
remove_tatweel = normalize.remove_tatweel
normalize_alef = normalize.normalize_alef
normalize_hamza = normalize.normalize_hamza
normalize_ta_marbuta = normalize.normalize_ta_marbuta
normalize_alef_maksura = normalize.normalize_alef_maksura
normalize_arabic = normalize.normalize_arabic
normalize_for_wer = normalize.normalize_for_wer
normalize_whitespace = normalize.normalize_whitespace


class TestRemoveDiacritics:
    def test_removes_fatha_damma_kasra(self):
        assert remove_diacritics("بِسْمِ") == "بسم"

    def test_removes_shadda(self):
        assert remove_diacritics("اللَّهِ") == "الله"

    def test_removes_tanwin(self):
        assert remove_diacritics("كِتَابًا") == "كتابا"

    def test_bismillah_full(self):
        result = remove_diacritics("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ")
        assert result == "بسم الله الرحمن الرحيم"

    def test_empty_string(self):
        assert remove_diacritics("") == ""

    def test_no_diacritics(self):
        text = "بسم الله"
        assert remove_diacritics(text) == text


class TestRemoveTatweel:
    def test_removes_tatweel(self):
        assert remove_tatweel("اللـــه") == "الله"

    def test_no_tatweel(self):
        assert remove_tatweel("الله") == "الله"


class TestNormalizeAlef:
    def test_alef_hamza_above(self):
        assert normalize_alef("أحمد") == "احمد"

    def test_alef_hamza_below(self):
        assert normalize_alef("إبراهيم") == "ابراهيم"

    def test_alef_madda(self):
        assert normalize_alef("آمن") == "امن"

    def test_alef_wasla(self):
        assert normalize_alef("ٱلرَّحْمَنِ") == "الرَّحْمَنِ"

    def test_multiple_variants(self):
        assert normalize_alef("أحمد إبراهيم آمن") == "احمد ابراهيم امن"


class TestNormalizeHamza:
    def test_waw_hamza(self):
        assert normalize_hamza("مؤمن") == "مومن"

    def test_yeh_hamza(self):
        assert normalize_hamza("سئل") == "سيل"


class TestNormalizeTaMarbuta:
    def test_ta_marbuta_to_ha(self):
        assert normalize_ta_marbuta("رحمة") == "رحمه"

    def test_no_ta_marbuta(self):
        assert normalize_ta_marbuta("كتاب") == "كتاب"


class TestNormalizeAlefMaksura:
    def test_alef_maksura_to_yeh(self):
        assert normalize_alef_maksura("على") == "علي"

    def test_alef_maksura_to_yeh_end(self):
        assert normalize_alef_maksura("موسى") == "موسي"


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("بسم   الله") == "بسم الله"

    def test_strips_edges(self):
        assert normalize_whitespace("  بسم الله  ") == "بسم الله"

    def test_tabs_and_newlines(self):
        assert normalize_whitespace("بسم\t\nالله") == "بسم الله"


class TestNormalizeArabic:
    def test_full_bismillah_uthmani(self):
        # Uthmani script with diacritics and alef wasla
        result = normalize_arabic("بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ")
        assert result == "بسم الله الرحمن الرحيم"

    def test_plain_text_unchanged(self):
        text = "بسم الله الرحمن الرحيم"
        assert normalize_arabic(text) == text

    def test_empty_string(self):
        assert normalize_arabic("") == ""

    def test_mixed_arabic_latin(self):
        result = normalize_arabic("Surah الفَاتِحَة")
        assert result == "Surah الفاتحه"

    def test_complex_verse(self):
        # Al-Baqarah 2:255 (Ayat al-Kursi) opening with diacritics
        verse = "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ"
        result = normalize_arabic(verse)
        assert "الله" in result
        assert "لا" in result


class TestNormalizeForWer:
    def test_normalizes_both(self):
        ref = "بِسْمِ اللَّهِ"
        hyp = "بسم الله"
        norm_ref, norm_hyp = normalize_for_wer(ref, hyp)
        assert norm_ref == norm_hyp

    def test_different_texts(self):
        ref = "بِسْمِ اللَّهِ"
        hyp = "باسم الرحمن"
        norm_ref, norm_hyp = normalize_for_wer(ref, hyp)
        assert norm_ref != norm_hyp
