"""Download Quran translations in 108+ languages.

Source: fawazahmed0/quran-api via jsDelivr CDN
License: Unlicense (public domain)

API docs: https://github.com/fawazahmed0/quran-api
"""

import argparse
import json
import time
from pathlib import Path

import httpx

# Base URL for the quran-api CDN
BASE_URL = "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1"
EDITIONS_URL = f"{BASE_URL}/editions"

# Common languages with their edition names (one per language)
# Full list available at: https://github.com/fawazahmed0/quran-api/blob/1/Translations.md
DEFAULT_EDITIONS = {
    "tr": "tur-dipisilam",
    "en": "eng-abdulhye",
    "fr": "fra-montada",
    "de": "deu-aburida",
    "es": "spa-islamhouse",
    "ru": "rus-abuadel",
    "zh": "zho-makin",
    "ja": "jpn-saeedsato",
    "ko": "kor-unknown",
    "hi": "hin-unknown",
    "bn": "ben-muhiuddinkhan",
    "ar": "ara-quranridwaan",
    "ur": "urd-aborraee",
    "id": "ind-sabiq",
    "ms": "msa-basmeih",
    "fa": "fas-unknown",
    "pt": "por-helminasr",
    "it": "ita-hamza",
    "nl": "nld-sofian",
    "sv": "swe-knutbernstrom",
    "pl": "pol-bielawskiego",
    "ro": "ron-grigore",
    "th": "tha-kingfahad",
    "vi": "vie-hassanabdulkarim",
    "sw": "swa-abubakr",
    "ha": "hau-abubakarmahmoud",
    "yo": "yor-shaykhaburahimah",
    "am": "amh-sadiqandsank",
    "so": "som-abdulwali",
    "ku": "kur-salahuddinabdulk",
    "ps": "pus-unknown",
    "az": "aze-vasimmammadaliy",
    "uz": "uzb-muhammadsodiq",
    "kk": "kaz-khalifaaltai",
    "tg": "tgk-unknown",
    "ky": "kir-unknown",
    "tt": "tat-unknown",
    "ml": "mal-karakunnumahmoo",
    "ta": "tam-jantrust",
    "te": "tel-abdulraheemmoham",
    "kn": "kan-unknown",
    "gu": "guj-rababorivala",
    "mr": "mar-muhammadshafii",
    "ne": "nep-unknown",
    "si": "sin-ruwwadcenter",
    "my": "mya-unknown",
    "km": "khm-unknown",
    "fil": "fil-unknown",
    "bg": "bul-tzviatantheoph",
    "sr": "srp-dakwahcenter",
    "hr": "hrv-unknown",
    "bs": "bos-unknown",
    "sq": "sqi-ftiahmed",
    "mk": "mkd-unknown",
    "uk": "ukr-yakubovych",
    "cs": "ces-hrbek",
    "hu": "hun-abdulhaqq",
    "fi": "fin-unknown",
    "no": "nor-einarberg",
    "da": "dan-unknown",
}


def get_available_editions() -> dict:
    """Fetch the list of all available translation editions.

    Returns:
        Dict mapping edition names to their metadata.
    """
    print("Fetching available editions...")
    response = httpx.get(f"{EDITIONS_URL}.json", timeout=30.0)
    response.raise_for_status()
    return response.json()


def download_edition(edition_name: str, output_dir: Path) -> Path | None:
    """Download a single translation edition.

    Args:
        edition_name: Name of the edition (e.g., "tur-dipisilam").
        output_dir: Directory to save the translation file.

    Returns:
        Path to the saved file, or None on failure.
    """
    url = f"{EDITIONS_URL}/{edition_name}.json"
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        print(f"  Failed to download {edition_name}: {e}")
        return None

    # Restructure to {surah:ayah: translation_text}
    translation = {}
    if isinstance(data, dict) and "quran" in data:
        for verse in data["quran"]:
            chapter = str(verse.get("chapter", ""))
            verse_num = str(verse.get("verse", ""))
            text = verse.get("text", "")
            if chapter and verse_num:
                if chapter not in translation:
                    translation[chapter] = {}
                translation[chapter][verse_num] = text

    output_file = output_dir / f"{edition_name}.json"
    output_file.write_text(
        json.dumps(translation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verse_count = sum(len(v) for v in translation.values())
    return output_file


def download_translations(
    languages: list[str] | None = None,
    output_dir: str = "./data/translations",
    delay: float = 0.5,
) -> list[Path]:
    """Download translations for specified languages.

    Args:
        languages: List of language codes (e.g., ["tr", "en", "fr"]).
                   If None, downloads all available in DEFAULT_EDITIONS.
        output_dir: Directory to save translation files.
        delay: Delay between requests in seconds (be nice to CDN).

    Returns:
        List of paths to downloaded files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if languages is None:
        editions_to_download = DEFAULT_EDITIONS
    else:
        editions_to_download = {
            lang: DEFAULT_EDITIONS[lang]
            for lang in languages
            if lang in DEFAULT_EDITIONS
        }
        missing = set(languages) - set(DEFAULT_EDITIONS)
        if missing:
            print(f"Warning: No edition mapping for languages: {missing}")

    downloaded = []
    total = len(editions_to_download)
    print(f"Downloading {total} translations...")

    for i, (lang, edition) in enumerate(editions_to_download.items(), 1):
        print(f"  [{i}/{total}] {lang}: {edition}...", end=" ")
        path = download_edition(edition, output_path)
        if path:
            print(f"OK")
            downloaded.append(path)
        else:
            print(f"FAILED")

        if i < total:
            time.sleep(delay)

    print(f"\nDownloaded {len(downloaded)}/{total} translations to {output_path}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download Quran translations")
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes (e.g., tr,en,fr). Default: all",
    )
    parser.add_argument("--output-dir", default="./data/translations", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (s)")
    parser.add_argument("--list", action="store_true", help="List available languages")
    args = parser.parse_args()

    if args.list:
        print("Available languages:")
        for lang, edition in sorted(DEFAULT_EDITIONS.items()):
            print(f"  {lang}: {edition}")
        return

    languages = args.languages.split(",") if args.languages else None
    download_translations(
        languages=languages,
        output_dir=args.output_dir,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
