"""Tests for the FastAPI server."""

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# These tests require: pip install httpx pytest-asyncio
api_mod = importlib.import_module("4_inference.api")
app = api_mod.app


@pytest.fixture
def quran_data(tmp_path):
    """Create test Quran data."""
    data = {
        "1": {
            "1": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
            "2": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        },
    }
    file_path = tmp_path / "quran-uthmani.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(file_path)


@pytest.fixture
def translations_data(tmp_path):
    """Create test translation files."""
    tr_dir = tmp_path / "translations"
    tr_dir.mkdir()
    tr_data = {"1": {"1": "Rahman ve Rahim olan Allah'in adiyla"}}
    (tr_dir / "tur-dipisilam.json").write_text(
        json.dumps(tr_data, ensure_ascii=False), encoding="utf-8"
    )
    return str(tr_dir)


class TestHealthEndpoint:
    def test_health(self):
        from fastapi.testclient import TestClient

        # Manually set state for testing
        api_mod._state["quran_text"] = {"1": {"1": "test"}}
        api_mod._state["edition_map"] = {}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

        api_mod._state.clear()


class TestVerseEndpoint:
    def test_get_verse(self):
        from fastapi.testclient import TestClient

        api_mod._state["quran_text"] = {
            "1": {"1": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"}
        }
        api_mod._state["edition_map"] = {}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/verse/1:1")
        assert response.status_code == 200
        data = response.json()
        assert data["verse_key"] == "1:1"
        assert data["surah"] == 1
        assert data["ayah"] == 1
        assert "arabic_text" in data

        api_mod._state.clear()

    def test_get_verse_not_found(self):
        from fastapi.testclient import TestClient

        api_mod._state["quran_text"] = {"1": {"1": "test"}}
        api_mod._state["edition_map"] = {}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/verse/999:999")
        assert response.status_code == 404

        api_mod._state.clear()

    def test_get_verse_invalid_format(self):
        from fastapi.testclient import TestClient

        api_mod._state["quran_text"] = {"1": {"1": "test"}}
        api_mod._state["edition_map"] = {}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/verse/invalid")
        assert response.status_code == 400

        api_mod._state.clear()


class TestLanguagesEndpoint:
    def test_list_languages(self):
        from fastapi.testclient import TestClient

        api_mod._state["edition_map"] = {
            "tr": "tur-dipisilam",
            "en": "eng-abdulhye",
        }

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/languages")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        codes = [item["code"] for item in data]
        assert "tr" in codes
        assert "en" in codes

        api_mod._state.clear()
