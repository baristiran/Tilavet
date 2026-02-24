"""Download Quran text in Uthmani script as JSON.

Source: risan/quran-json via jsDelivr CDN
License: CC-BY-SA-4.0
"""

import argparse
import json
from pathlib import Path

import httpx


QURAN_JSON_URL = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran.json"
CHAPTERS_URL = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/chapters.json"


def download_quran_text(output_dir: str = "./data") -> Path:
    """Download the full Quran text in Uthmani script.

    Downloads from risan/quran-json CDN and saves as structured JSON.

    Args:
        output_dir: Directory to save the output file.

    Returns:
        Path to the saved JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "quran-uthmani.json"

    print(f"Downloading Quran text from {QURAN_JSON_URL}...")
    response = httpx.get(QURAN_JSON_URL, timeout=60.0)
    response.raise_for_status()
    raw_data = response.json()

    # Restructure: {surah_number: {ayah_number: "arabic_text"}}
    quran = {}
    for chapter in raw_data:
        surah_num = str(chapter["id"])
        quran[surah_num] = {}
        for verse in chapter["verses"]:
            ayah_num = str(verse["id"])
            quran[surah_num][ayah_num] = verse["text"]

    output_file.write_text(json.dumps(quran, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {sum(len(v) for v in quran.values())} verses to {output_file}")
    return output_file


def download_chapters_info(output_dir: str = "./data") -> Path:
    """Download surah/chapter metadata.

    Args:
        output_dir: Directory to save the output file.

    Returns:
        Path to the saved JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "chapters.json"

    print(f"Downloading chapter info from {CHAPTERS_URL}...")
    response = httpx.get(CHAPTERS_URL, timeout=60.0)
    response.raise_for_status()

    output_file.write_text(response.text, encoding="utf-8")
    print(f"Saved chapter info to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download Quran text")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    args = parser.parse_args()

    download_quran_text(args.output_dir)
    download_chapters_info(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
