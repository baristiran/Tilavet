"""FastAPI server for OpenQuran Whisper.

Provides REST API for Quran speech recognition and translation.

Usage:
    uvicorn 4_inference.api:app --host 0.0.0.0 --port 8000
"""

import importlib
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel


# Global state
_state = {}


def _get_edition_map() -> dict[str, str]:
    """Get language code to edition name mapping."""
    try:
        dl_mod = importlib.import_module("1_data_prep.download_translations")
        return dl_mod.DEFAULT_EDITIONS
    except ImportError:
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources at startup."""
    # Load translation engine
    try:
        trans_mod = importlib.import_module("5_features.translation_engine")
        _state["translation_engine"] = trans_mod.TranslationEngine()
        available = _state["translation_engine"].get_available_editions()
        print(f"Translation engine loaded: {len(available)} editions available")
    except Exception as e:
        print(f"Warning: Could not load translation engine: {e}")

    # Load verse matcher / surah detector
    quran_path = "./data/quran-uthmani.json"
    if Path(quran_path).exists():
        try:
            detector_mod = importlib.import_module("5_features.surah_detector")
            _state["surah_detector"] = detector_mod.SurahDetector(quran_path)
            print("Surah detector loaded")
        except Exception as e:
            print(f"Warning: Could not load surah detector: {e}")

    # Load Quran text for verse lookup
    if Path(quran_path).exists():
        _state["quran_text"] = json.loads(Path(quran_path).read_text(encoding="utf-8"))
        print("Quran text loaded")

    _state["edition_map"] = _get_edition_map()

    # Note: STT model is NOT loaded at startup by default (too heavy for dev).
    # Use --load-model flag or set LOAD_MODEL=1 env var.

    yield

    _state.clear()


app = FastAPI(
    title="OpenQuran Whisper API",
    description="Open-source Quran Speech-to-Text & Translation API",
    version="0.1.0",
    lifespan=lifespan,
)


# Response models

class TranscriptionResponse(BaseModel):
    text: str
    verse_key: str | None = None
    confidence: float | None = None
    arabic_text: str | None = None
    translations: dict[str, str] | None = None


class VerseResponse(BaseModel):
    verse_key: str
    surah: int
    ayah: int
    arabic_text: str
    translations: dict[str, str] | None = None


class LanguageInfo(BaseModel):
    code: str
    edition: str


# Endpoints

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "features": {
            "stt": "model" in _state,
            "translations": "translation_engine" in _state,
            "verse_detection": "surah_detector" in _state,
        },
    }


@app.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    languages: str = Query(default="", description="Comma-separated language codes (e.g., tr,en,fr)"),
):
    """Transcribe uploaded Quran audio to Arabic text with optional translations.

    Upload a WAV/MP3 audio file of Quran recitation. Returns:
    - Transcribed Arabic text
    - Detected verse (surah:ayah) with confidence score
    - Translations in requested languages
    """
    if "model" not in _state:
        raise HTTPException(
            status_code=503,
            detail="STT model not loaded. Start server with model loading enabled.",
        )

    # Save uploaded file
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        transcribe_mod = importlib.import_module("4_inference.transcribe")
        text = transcribe_mod.transcribe(
            tmp_path,
            _state["model"],
            _state["processor"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result = TranscriptionResponse(text=text)

    # Verse detection
    if "surah_detector" in _state:
        detection = _state["surah_detector"].detect(text)
        if detection:
            result.verse_key = detection["verse_key"]
            result.confidence = detection["confidence"]
            result.arabic_text = detection["arabic_text"]

    # Translations
    if languages and result.verse_key and "translation_engine" in _state:
        lang_codes = [l.strip() for l in languages.split(",") if l.strip()]
        editions = {
            lc: _state["edition_map"][lc]
            for lc in lang_codes
            if lc in _state["edition_map"]
        }
        if editions:
            result.translations = _state["translation_engine"].get_translations(
                result.verse_key, editions
            )

    return result


@app.get("/api/v1/verse/{verse_key}", response_model=VerseResponse)
async def get_verse(
    verse_key: str,
    languages: str = Query(default="", description="Comma-separated language codes"),
):
    """Get a specific Quran verse by key (e.g., 1:1 for Al-Fatiha verse 1)."""
    if "quran_text" not in _state:
        raise HTTPException(status_code=503, detail="Quran text not loaded")

    parts = verse_key.split(":")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid verse key format. Use surah:ayah (e.g., 1:1)")

    surah, ayah = parts
    quran = _state["quran_text"]

    if surah not in quran or ayah not in quran[surah]:
        raise HTTPException(status_code=404, detail=f"Verse {verse_key} not found")

    result = VerseResponse(
        verse_key=verse_key,
        surah=int(surah),
        ayah=int(ayah),
        arabic_text=quran[surah][ayah],
    )

    if languages and "translation_engine" in _state:
        lang_codes = [l.strip() for l in languages.split(",") if l.strip()]
        editions = {
            lc: _state["edition_map"][lc]
            for lc in lang_codes
            if lc in _state["edition_map"]
        }
        if editions:
            result.translations = _state["translation_engine"].get_translations(
                verse_key, editions
            )

    return result


@app.get("/api/v1/translations/{verse_key}")
async def get_translations(
    verse_key: str,
    languages: str = Query(default="", description="Comma-separated language codes (empty=all loaded)"),
):
    """Get translations for a specific verse."""
    if "translation_engine" not in _state:
        raise HTTPException(status_code=503, detail="Translation engine not loaded")

    engine = _state["translation_engine"]

    if languages:
        lang_codes = [l.strip() for l in languages.split(",") if l.strip()]
        editions = {
            lc: _state["edition_map"][lc]
            for lc in lang_codes
            if lc in _state["edition_map"]
        }
    else:
        editions = None

    translations = engine.get_translations(verse_key, editions)

    if not translations:
        raise HTTPException(status_code=404, detail=f"No translations found for {verse_key}")

    return {"verse_key": verse_key, "translations": translations}


@app.get("/api/v1/languages", response_model=list[LanguageInfo])
async def list_languages():
    """List all available translation languages."""
    edition_map = _state.get("edition_map", {})
    return [
        LanguageInfo(code=code, edition=edition)
        for code, edition in sorted(edition_map.items())
    ]


@app.get("/api/v1/search")
async def search_verses(
    q: str = Query(..., description="Search text (Arabic)"),
    limit: int = Query(default=5, ge=1, le=20),
):
    """Search for Quran verses by Arabic text."""
    if "surah_detector" not in _state:
        raise HTTPException(status_code=503, detail="Verse matcher not loaded")

    matches = _state["surah_detector"].detect_multiple(q, top_k=limit)
    return {"query": q, "results": matches}
