"""Build quran.db for the iOS app bundle.

Downloads Quran text (with chapter metadata) and Turkish translation,
then builds a SQLite database with pre-normalized text for matching.

Usage:
    cd TilavetAPI/
    PYTHONPATH=. python scripts/build_quran_db.py
    PYTHONPATH=. python scripts/build_quran_db.py --output ./quran.db
    PYTHONPATH=. python scripts/build_quran_db.py --languages tr,en
"""

import argparse
import importlib
import json
import sqlite3
from pathlib import Path

import httpx

# Import from our existing modules
normalize_mod = importlib.import_module("1_data_prep.normalize_text")

normalize_arabic = normalize_mod.normalize_arabic

# Raw quran.json contains both chapter metadata and verses
QURAN_JSON_URL = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran.json"

# Turkish translation editions (Diyanet Isleri is the most common/reliable)
TURKISH_EDITIONS = {
    "tr": {
        "edition": "tur-diyanetisleri",
        "url": "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/tur-diyanetisleri.json",
        "author": "Diyanet Isleri",
    },
}


def download_raw_quran(data_dir: Path) -> Path:
    """Download the raw quran.json (chapters + verses combined)."""
    output_file = data_dir / "quran-raw.json"
    if output_file.exists():
        print("  Raw Quran data already cached")
        return output_file

    print(f"  Downloading from {QURAN_JSON_URL}...")
    response = httpx.get(QURAN_JSON_URL, timeout=60.0)
    response.raise_for_status()
    output_file.write_text(response.text, encoding="utf-8")
    print(f"  Saved to {output_file}")
    return output_file


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the database schema."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS surahs (
            number INTEGER PRIMARY KEY,
            name_arabic TEXT NOT NULL,
            name_transliteration TEXT NOT NULL,
            verses_count INTEGER NOT NULL,
            revelation_type TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS verses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            surah_number INTEGER NOT NULL,
            ayah_number INTEGER NOT NULL,
            verse_key TEXT NOT NULL UNIQUE,
            text_uthmani TEXT NOT NULL,
            text_normalized TEXT NOT NULL,
            FOREIGN KEY (surah_number) REFERENCES surahs(number)
        );

        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_key TEXT NOT NULL,
            language_code TEXT NOT NULL,
            edition_name TEXT NOT NULL,
            text TEXT NOT NULL,
            UNIQUE(verse_key, language_code)
        );

        CREATE INDEX IF NOT EXISTS idx_verses_surah ON verses(surah_number);
        CREATE INDEX IF NOT EXISTS idx_verses_key ON verses(verse_key);
        CREATE INDEX IF NOT EXISTS idx_translations_key_lang ON translations(verse_key, language_code);
    """)


def populate_from_raw(conn: sqlite3.Connection, raw_path: Path) -> None:
    """Populate surahs and verses from raw quran.json."""
    data = json.loads(raw_path.read_text(encoding="utf-8"))

    surah_rows = []
    verse_rows = []

    for chapter in data:
        surah_num = chapter["id"]
        surah_rows.append((
            surah_num,
            chapter["name"],                # Arabic name (e.g., "الفاتحة")
            chapter["transliteration"],      # e.g., "Al-Fatihah"
            chapter["total_verses"],
            chapter["type"],                 # "meccan" / "medinan"
        ))

        for verse in chapter["verses"]:
            ayah_num = verse["id"]
            text = verse["text"]
            verse_key = f"{surah_num}:{ayah_num}"
            normalized = normalize_arabic(text)
            verse_rows.append((
                surah_num,
                ayah_num,
                verse_key,
                text,
                normalized,
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO surahs (number, name_arabic, name_transliteration, verses_count, revelation_type) VALUES (?, ?, ?, ?, ?)",
        surah_rows,
    )
    print(f"  Inserted {len(surah_rows)} surahs")

    conn.executemany(
        "INSERT OR REPLACE INTO verses (surah_number, ayah_number, verse_key, text_uthmani, text_normalized) VALUES (?, ?, ?, ?, ?)",
        verse_rows,
    )
    print(f"  Inserted {len(verse_rows)} verses")


def download_translation(lang: str, data_dir: Path) -> Path | None:
    """Download a translation edition directly."""
    info = TURKISH_EDITIONS.get(lang)
    if not info:
        print(f"  Warning: No edition mapping for '{lang}'")
        return None

    output_file = data_dir / "translations" / f"{info['edition']}.json"
    if output_file.exists():
        print(f"  Translation '{lang}' ({info['edition']}) already cached")
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {info['edition']} from {info['url']}...")
    response = httpx.get(info["url"], timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    raw_data = response.json()

    # Restructure from [{chapter, verse, text}] to {surah: {ayah: text}}
    translation = {}
    if isinstance(raw_data, dict) and "quran" in raw_data:
        for verse in raw_data["quran"]:
            chapter = str(verse.get("chapter", ""))
            verse_num = str(verse.get("verse", ""))
            text = verse.get("text", "")
            if chapter and verse_num:
                if chapter not in translation:
                    translation[chapter] = {}
                translation[chapter][verse_num] = text

    output_file.write_text(
        json.dumps(translation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verse_count = sum(len(v) for v in translation.values())
    print(f"  Saved {verse_count} verses for '{lang}'")
    return output_file


def populate_translations(
    conn: sqlite3.Connection,
    translations_dir: Path,
    languages: list[str],
) -> None:
    """Insert translations for specified languages."""
    for lang in languages:
        info = TURKISH_EDITIONS.get(lang)
        if not info:
            print(f"  Warning: No edition for language '{lang}', skipping")
            continue

        trans_file = translations_dir / f"{info['edition']}.json"
        if not trans_file.exists():
            print(f"  Warning: Translation file {trans_file} not found, skipping")
            continue

        data = json.loads(trans_file.read_text(encoding="utf-8"))

        rows = []
        for surah_num, ayahs in data.items():
            for ayah_num, text in ayahs.items():
                verse_key = f"{surah_num}:{ayah_num}"
                rows.append((verse_key, lang, info["edition"], text))

        conn.executemany(
            "INSERT OR REPLACE INTO translations (verse_key, language_code, edition_name, text) VALUES (?, ?, ?, ?)",
            rows,
        )
        print(f"  Inserted {len(rows)} translations for '{lang}' ({info['edition']})")


def verify_db(conn: sqlite3.Connection) -> None:
    """Run verification checks on the built database."""
    surah_count = conn.execute("SELECT COUNT(*) FROM surahs").fetchone()[0]
    verse_count = conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
    trans_count = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]

    print(f"\n  Verification:")
    print(f"    Surahs: {surah_count} (expected: 114)")
    print(f"    Verses: {verse_count} (expected: 6236)")
    print(f"    Translations: {trans_count}")

    # Check Bismillah
    bismillah = conn.execute(
        "SELECT text_uthmani, text_normalized FROM verses WHERE verse_key = '1:1'"
    ).fetchone()
    if bismillah:
        print(f"    Bismillah (1:1): {bismillah[0][:60]}...")
        print(f"    Normalized:      {bismillah[1][:60]}...")
    else:
        print("    ERROR: Bismillah (1:1) not found!")

    # Check Turkish translation
    tr_bismillah = conn.execute(
        "SELECT text FROM translations WHERE verse_key = '1:1' AND language_code = 'tr'"
    ).fetchone()
    if tr_bismillah:
        print(f"    Turkish (1:1):   {tr_bismillah[0][:80]}...")
    else:
        print("    WARNING: Turkish translation for 1:1 not found")

    # Check last verse (114:6)
    last = conn.execute(
        "SELECT verse_key, text_uthmani FROM verses ORDER BY surah_number DESC, ayah_number DESC LIMIT 1"
    ).fetchone()
    if last:
        print(f"    Last verse:      {last[0]} - {last[1][:60]}...")

    assert surah_count == 114, f"Expected 114 surahs, got {surah_count}"
    assert verse_count == 6236, f"Expected 6236 verses, got {verse_count}"
    print("    All checks passed!")


def build_quran_db(
    output_path: str = "./quran.db",
    data_dir: str = "./data",
    languages: list[str] | None = None,
) -> Path:
    """Build the complete quran.db.

    Args:
        output_path: Where to write the database file.
        data_dir: Directory for intermediate data files.
        languages: Language codes for translations. Default: ["tr"]

    Returns:
        Path to the built database.
    """
    if languages is None:
        languages = ["tr"]

    output = Path(output_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)

    # Step 1: Download raw Quran data
    print("Step 1: Downloading Quran data...")
    raw_file = download_raw_quran(data)

    # Step 2: Download translations
    print(f"Step 2: Downloading translations for {languages}...")
    for lang in languages:
        download_translation(lang, data)

    # Step 3: Build SQLite database
    if output.exists():
        output.unlink()

    print(f"Step 3: Creating database at {output}...")
    conn = sqlite3.connect(str(output))
    try:
        create_schema(conn)

        print("Step 4: Populating surahs and verses...")
        populate_from_raw(conn, raw_file)

        print("Step 5: Populating translations...")
        populate_translations(conn, data / "translations", languages)

        conn.commit()

        print("Step 6: Verifying...")
        verify_db(conn)

    finally:
        conn.close()

    db_size = output.stat().st_size / 1024
    print(f"\nDone! Database: {output} ({db_size:.1f} KB)")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Build quran.db for iOS app bundle"
    )
    parser.add_argument(
        "--output",
        default="./quran.db",
        help="Output database path (default: ./quran.db)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Data directory for intermediate files",
    )
    parser.add_argument(
        "--languages",
        default="tr",
        help="Comma-separated language codes (default: tr)",
    )
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",")]
    build_quran_db(
        output_path=args.output,
        data_dir=args.data_dir,
        languages=languages,
    )


if __name__ == "__main__":
    main()
