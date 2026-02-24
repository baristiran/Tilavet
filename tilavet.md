# OpenQuran Whisper

## Evrensel Kuran Konusma Tanima, Ceviri ve Egitim Platformu

---

# PROJE DURUMU

## Mevcut Durum: Phase 2 - Egitim Oncesi

| | |
|---|---|
| **Repo** | https://github.com/baristiran/Tilavet |
| **Durum** | Phase 1 tamamlandi (43 dosya, 53 test). Config guncellendi, scriptler hazir. Egitim baslatilacak. |
| **Sonraki Adim** | Runpod A100 pod'u baslat → `bash scripts/runpod_setup.sh` → egitim baslat |
| **Lisans** | MIT |

### Son Kararlar

| Karar | Detay | Neden |
|-------|-------|-------|
| **Model** | `openai/whisper-large-v3` (1.5B param) | Tarteel AI'in whisper-base'ini gecmek icin |
| **Hedef WER** | < %3 | Tarteel'in %5.75'ini gecmeli, yoksa yapmaya degmez |
| **LoRA** | r=16, alpha=32 | large-v3 icin daha agresif adaptasyon |
| **Egitim Ortami** | Runpod A100 80GB, SSH | fp16 precision, batch_size=8, grad_accum=4, ~20-24 saat |
| **Baseline Test** | large-v3: %8.02, tarteel: %11.83 | Fine-tune olmadan bile Tarteel'i geciyor! |
| **Strateji** | Platform odakli | Tarteel'e rakip degil, gelistiriciler icin acik kaynak API |

### Tamamlanan Isler (Phase 1)

- [x] Proje iskeleti (.gitignore, pyproject.toml, requirements.txt, LICENSE, README)
- [x] Arapca metin normalizasyonu (pure Python, 29 test)
- [x] Veri pipeline (dataset, Kuran metni, 108 dil ceviri indirme)
- [x] LoRA egitim scripti (A100 fp16 + T4 8-bit dual mod, auto-resume)
- [x] Degerlendirme scripti (eval.py)
- [x] Trigram-based ayet eslestirme (Levenshtein + LCS + Jaccard)
- [x] 108 dil ceviri motoru (lazy loading + cache)
- [x] Sure/ayet tanima modulu
- [x] FastAPI REST API (transkripsiyon, ayet, ceviriler, arama)
- [x] CLI transkripsiyon araci
- [x] Colab egitim notebook'u
- [x] Runpod A100 setup scripti
- [x] 53 test (normalizasyon, ceviri, ayet eslestirme, API)
- [x] GitHub'a push (baristiran/Tilavet)
- [x] Baseline WER test notebook'u (Colab'da 3 modeli karsilastirma)
- [x] lora_config.json whisper-large-v3 + r=16, alpha=32'ye guncellendi
- [x] Tum scriptler (eval.py, transcribe.py, train) yeni transformers API'ye uyumlandi

### Yapilacaklar (Phase 2)

- [x] lora_config.json whisper-large-v3 + r=16, alpha=32'ye guncellendi ✅
- [x] Baseline WER testi yapildi (large-v3 %8.02, tarteel %11.83) ✅
- [x] Scriptler yeni transformers API'ye uyumlandi ✅
- [ ] Runpod A100 pod'u baslat, SSH baglan ← SIRADAKI
- [ ] `bash scripts/runpod_setup.sh` calistir
- [ ] `python3 -m 2_training.train_whisper_lora` ile egitimi baslat
- [ ] WER sonuclarini degerlendir (hedef: <%3)
- [ ] Modeli HuggingFace Hub'a push et

---

## Hizli Ozet

| Ozellik | Aciklama |
|---------|----------|
| **Proje Adi** | OpenQuran Whisper |
| **Tur** | Acik Kaynak Kuran STT + Ceviri Platformu (Gelistirici API) |
| **Temel Teknoloji** | Whisper-large-v3 (1.5B) + LoRA Fine-tuning |
| **Hedef Diller** | Arapca (Kuran) + **108 Ceviri Dili** |
| **Hedef WER** | < %3 (Tarteel AI'in %5.75'inden iyi) |
| **Lisans** | MIT License |
| **Egitim** | Runpod A100 80GB (~20-24 saat, ~$35-40) |

---

## Vizyon ve Misyon

### Vizyon

> **"Kuran-i Kerim'i dunyadaki herkesin kendi ana dilinde, gercek zamanli olarak dinleyebilmesi, anlayabilmesi ve dogru okuyabilmesi icin acik kaynakli bir platform olusturmak."**

### Misyon

1. **Erisebilirlik**: Kuran'i sesli okuma yoluyla tum cihazlarda erisebilir kilmak
2. **Evrensellik**: 108 dilde es zamanli ceviri ile tum Muslumanlara ulasmak
3. **Egitim**: Hafizlik ogrencilerine ezber kontrolu, telaffuz analizi ve tecvit ogretimi saglamak
4. **Aciklik**: Tum teknolojiyi toplulukla paylasarak Islamic Tech ekosistemi guclendirmek
5. **Bagimsizlik**: Offline kullanimi desteklemek (⚠️ Su an tam offline degil — ONNX export ve whisper.cpp entegrasyonu ile Phase 4'te hedefleniyor)

---

## Stratejik Pozisyonlama

### Tarteel AI ile Karsilastirma

| | Tarteel AI | OpenQuran Whisper |
|---|---|---|
| **Model** | whisper-base (77M) | **whisper-large-v3 (1.5B)** |
| **Fine-tune** | Full fine-tune, 8 GPU | **LoRA, tek A100** |
| **WER (reported)** | %5.75 (kendi raporu) | **Hedef: <%3** |
| **WER (bizim testmiz)** | %11.83 (100 ornek) | **%8.02 (fine-tune yok!)** |
| **Lisans** | Apache 2.0 | **MIT** |
| **Dil Destegi** | 1 (Arapca) | **108 dil ceviri** |
| **API** | Kapali | **Acik kaynak REST API** |
| **Odak** | Urun (mobil app) | **Platform (gelistirici API)** |
| **Egitim Kolayligi** | 8 GPU gerekli | **Tek GPU'da egitebilir (LoRA)** |

### Bizim Avantajlarimiz

1. **Daha buyuk model**: large-v3 (1.5B) vs base (77M) - 20x daha fazla parametre
2. **108 dil ceviri**: Arapca STT + aninda ceviri, rakipsiz
3. **Acik platform**: Herkes API'yi kullanabilir, model egitebilir
4. **LoRA**: Herkes tek GPU'da kendi modelini fine-tune edebilir

### Bizim Dezavantajlarimiz

1. Henuz egitilmis model yok (egitim oncesi asamadayiz)
2. Mobil uygulama yok (platform/API odakli)
3. Topluluk yok henuz
4. Veri avantaji yok (ayni acik kaynak veri seti)

---

## Nasil Calisiyor?

### Sistem Mimarisi

```
Ses Girdisi → [Whisper-large-v3 + LoRA] → Arapca Metin → [Verse Matcher] → Sure:Ayet
                                                               ↓
                                                    [Translation Engine] → 108 Dil
                                                               ↓
                                                          [FastAPI] → REST Response
```

### Detayli Akis

1. Kullanici ses dosyasi yukler (API veya CLI)
2. Whisper-large-v3 + LoRA sesi Arapca metne cevirir
3. Verse Matcher trigram indeksi ile 6236 ayet icinden eslestirir
4. Translation Engine istenen dillerdeki cevirileri yukler
5. API birlesik sonucu doner: metin + sure:ayet + ceviriler

---

## Teknoloji Stack

### Gercek Stack (Implementasyon)

| Katman | Teknoloji | Aciklama |
|--------|-----------|----------|
| **STT Model** | `openai/whisper-large-v3` (1.5B) | Ses → Metin |
| **Fine-tuning** | `peft` + LoRA (r=16, alpha=32) | Model ozellestirme |
| **Ayet Eslestirme** | Trigram index + multi-metric | Levenshtein %40 + LCS %35 + Jaccard %25 |
| **Ceviri** | JSON lazy-loading engine | 108 dil, fawazahmed0/quran-api |
| **Backend API** | FastAPI | REST endpoints |
| **Ses Isleme** | librosa | 16kHz resampling, max 30s |
| **Normalizasyon** | Pure Python | Diacritics, hamza, alef, ta marbuta |
| **Metrik** | jiwer | WER hesaplama (diacritized + normalized) |
| **Egitim** | Runpod A100 80GB | fp16, LoRA, ~20-24 saat |

### Gelecek Stack (Henuz Implemente Edilmedi)

| Katman | Teknoloji | Durum |
|--------|-----------|-------|
| WebSocket Streaming | websockets | Planlanmis |
| Tecvit Analizi | Rule-based + ML | Planlanmis |
| Kiraat Siniflandirma | Classifier | Planlanmis |
| RAG Chatbot | LangChain + ChromaDB | Planlanmis |
| Offline Mod | ONNX export | Planlanmis |
| Mobil | Flutter | Planlanmis |
| Web | React | Planlanmis |

---

## Veri Kaynaklari

| Kaynak | Icerik | Lisans |
|--------|--------|--------|
| [tarteel-ai/everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) | 127K ses-metin cifti, 16kHz | MIT |
| [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api) | 108 dil Kuran cevirisi, JSON, jsDelivr CDN | Unlicense |
| [risan/quran-json](https://github.com/risan/quran-json) | Uthmani Kuran metni | CC-BY-SA-4.0 |

---

## Desteklenen Diller (108)

Ceviri destegi fawazahmed0/quran-api uzerinden 108 dilde saglanmaktadir.

> ⚠️ **Ceviri Kalitesi Notu**: Tum ceviriler esit kalitede degildir. Populer dillerde (tr, en, fr, de, ar, ur) taninan mealcilerin cevirileri kullanilirken, nadir dillerde kalite dusuk olabilir. Her API yanitinda ceviri kaynagi (edition) bilgisi dönülür. Kullanicilar ceviri seciminde bu bilgiyi dikkate almalidir. Dini icerikte otomatik/makine cevirileri icin API yanıtlarında uyari gosterilmesi planlanmaktadir.

**Oncelikli 20 dil** (setup script'inde otomatik indirilir):
Turkce, Ingilizce, Fransizca, Almanca, Ispanyolca, Rusca, Endonezce, Urduca, Bengalce, Malayca, Farsca, Cince, Japonca, Korece, Italyanca, Hollandaca, Portekizce, Isvecce, Lehce, Hintce

Tum 108 dilin tamamini indirmek icin: `python3 -m 1_data_prep.download_translations --all`

---

## Proje Yapisi (Gercek - 43 Dosya)

```
Tilavet/
├── .gitignore                          # Python/ML kurallari
├── tilavet.md                          # Bu dosya - proje dokumani
├── README.md                           # GitHub README (Ingilizce)
├── LICENSE                             # MIT License
├── pyproject.toml                      # Python proje config
├── requirements.txt                    # Core bagimliliklar
│
├── 1_data_prep/                        # VERI HAZIRLAMA
│   ├── __init__.py
│   ├── normalize_text.py               # Arapca normalizasyon (pure Python)
│   ├── download_dataset.py             # tarteel-ai/everyayah indirme
│   ├── download_quran_text.py          # Uthmani Kuran metni indirme
│   ├── download_translations.py        # 108 dil ceviri indirme
│   ├── prepare_dataset.py             # Whisper icin mel spectrogram + tokenizasyon
│   └── utils.py                        # Yardimci fonksiyonlar
│
├── 2_training/                         # MODEL EGITIMI
│   ├── __init__.py
│   ├── train_whisper_lora.py           # Ana egitim scripti (A100 + T4 dual mod)
│   ├── eval.py                         # WER degerlendirme
│   ├── requirements.txt               # GPU-only bagimliliklar
│   └── config/
│       └── lora_config.json            # LoRA + egitim hiperparametreleri
│
├── 3_models/                           # EGITILMIS MODELLER
│   ├── .gitkeep
│   └── README.md
│
├── 4_inference/                        # CIKARIM & API
│   ├── __init__.py
│   ├── transcribe.py                   # CLI transkripsiyon araci
│   ├── api.py                          # FastAPI REST sunucu
│   └── utils/
│       ├── __init__.py
│       ├── audio_processor.py          # Ses yukleme / 16kHz resampling
│       └── text_normalizer.py          # Post-processing
│
├── 5_features/                         # OZELLIK MODULLERI
│   ├── __init__.py
│   ├── verse_matcher.py               # Trigram-based ayet eslestirme
│   ├── translation_engine.py          # 108 dil ceviri motoru (lazy load + cache)
│   └── surah_detector.py              # Sure/Ayet tanima wrapper
│
├── data/                               # VERI (gitignore'da)
│   ├── .gitkeep
│   ├── README.md
│   └── translations/                   # 108 dil JSON dosyalari
│       └── .gitkeep
│
├── scripts/                            # YARDIMCI SCRIPTLER
│   ├── runpod_setup.sh                # Runpod A100 kurulum scripti
│   └── RUNPOD_GUIDE.md               # SSH egitim rehberi
│
├── notebooks/
│   ├── train_quran_stt_colab.ipynb    # Colab T4 egitim notebook'u
│   └── baseline_wer_test.ipynb        # Baseline WER karsilastirma (small vs large-v3 vs tarteel)
│
├── tests/                              # 53 TEST
│   ├── __init__.py
│   ├── conftest.py                    # sys.path ayari
│   ├── test_normalize_text.py         # 29 normalizasyon testi
│   ├── test_translation_engine.py     # 9 ceviri motoru testi
│   ├── test_verse_matcher.py          # 10 ayet eslestirme testi
│   └── test_api.py                    # 5 API endpoint testi
│
└── docs/
    └── architecture.md                 # Mimari dokumani
```

---

## API Endpoints

| Metod | Endpoint | Aciklama |
|-------|----------|----------|
| `GET` | `/health` | Saglik kontrolu |
| `POST` | `/api/v1/transcribe` | Ses → metin + ayet tespiti + ceviriler |
| `GET` | `/api/v1/verse/{surah}:{ayah}` | Belirli ayeti getir |
| `GET` | `/api/v1/translations/{verse_key}` | Coklu dil ceviriler |
| `GET` | `/api/v1/languages` | Desteklenen dil listesi |
| `GET` | `/api/v1/search` | Ayet arama |

---

## Kullanim

### Python ile

```python
import requests

# Ses dosyasi yukle (STT)
with open("kuran_sesi.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/transcribe",
        files={"file": f},
        params={"languages": "tr,en,fr"}
    )

print(response.json())
# {
#   "text": "بسم الله الرحمن الرحيم",
#   "verse_key": "1:1",
#   "confidence": 0.95,
#   "translations": {
#     "tr": "Rahman ve Rahim olan Allah'in adiyla",
#     "en": "In the name of God, the Most Gracious, the Most Merciful",
#     "fr": "Au nom de Dieu, le Misericordieux, le Compatissant"
#   }
# }

# Ayet getir
response = requests.get("http://localhost:8000/api/v1/verse/1:1")

# Cevirileri getir
response = requests.get(
    "http://localhost:8000/api/v1/translations/2:255",
    params={"languages": "tr,en,fr"}
)
```

### CLI ile

```bash
python3 -m 4_inference.transcribe \
    --audio recitation.wav \
    --adapter ./3_models/whisper-large-v3-quran-lora/final \
    --languages tr,en
```

---

## Performans Hedefleri

| Metrik | Hedef | Not |
|--------|-------|-----|
| **WER (Word Error Rate)** | < %3 | Tarteel AI'in %5.75'inden iyi |
| **WER (normalized)** | < %2 | Diacritik'siz karsilastirma |
| **CER (Character Error Rate)** | < %1.5 | Arapca icin WER'den daha anlamli (hareke/diacritik hassasiyeti) |
| **Latency (STT)** | < 500ms | GPU'da inference |
| **Latency (API)** | < 100ms | Endpoint yaniti (ceviri) |
| **Dil Kapsamasi** | 108 | Ceviri dili |

> **Not**: Kuran metni icin WER tek basina yetersiz olabilir. Arapca'da hareke farkliliklari WER'i yaniltici sekilde yukseltebilir. Bu nedenle CER (Character Error Rate) ve ileride Kuran'a ozel bir metrik (sadece anlam degistiren hatalari sayan) gelistirilmesi hedeflenmektedir.

### Karsilastirma Tablosu (Gercek Test Sonuclari)

Baseline WER testi: 100 ornek, `tarteel-ai/everyayah` test split, Colab T4'te yapildi.

| Model | WER (normalized) | Parametre | Not |
|-------|-------------------|-----------|-----|
| Whisper-small (fine-tune yok) | **%38.17** | 244M | Cok kotu, kullanisiz |
| tarteel-ai/whisper-base-ar-quran (fine-tuned) | **%11.83** | 77M | Full fine-tune, 8 GPU |
| Whisper-large-v3 (fine-tune yok) | **%8.02** | 1.5B | Fine-tune olmadan bile Tarteel'i geciyor! |
| **Hedef: whisper-large-v3 + LoRA** | **< %3** | **1.5B + LoRA ~5MB** | LoRA ile %8 → <%3 bekleniyor |

> **Kritik Bulgu**: whisper-large-v3, HICBIR fine-tune olmadan %8.02 WER ile Tarteel AI'in fine-tuned modelinden (%11.83) daha iyi performans gosteriyor. LoRA fine-tune ile <%3 hedefi gercekci.

---

## Egitim Detaylari

### Guncel Config (lora_config.json) ✅

```json
{
    "model_id": "openai/whisper-large-v3",
    "language": "ar",
    "task": "transcribe",
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "lora_dropout": 0.05,
        "bias": "none"
    },
    "training": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 100,
        "fp16": true,
        "load_in_8bit": false,
        "logging_steps": 25,
        "eval_steps": 1000,
        "save_steps": 1000,
        "save_total_limit": 3,
        "generation_max_length": 225,
        "max_duration_seconds": 30,
        "predict_with_generate": true,
        "report_to": "tensorboard"
    },
    "output_dir": "./3_models/whisper-large-v3-quran-lora",
    "hub_model_id": "openquran/whisper-large-v3-quran-lora"
}
```

### Egitim Ortami

| | Detay |
|---|---|
| **GPU** | Runpod A100 80GB |
| **Precision** | fp16 |
| **Tahmini Sure** | ~20-24 saat |
| **Tahmini Maliyet** | ~$35-40 |
| **Checkpoint** | Her 1000 step, auto-resume |

### Runpod'da Calistirma

```bash
# 1. SSH baglan
ssh root@<pod-ip> -p <port>

# 2. tmux baslat
tmux new -s train

# 3. Repo klonla
cd /workspace
git clone https://github.com/baristiran/Tilavet.git && cd Tilavet

# 4. Kurulum (deps + veri indirme)
bash scripts/runpod_setup.sh

# 5. Egitimi baslat
python3 -m 2_training.train_whisper_lora
```

---

## Yol Haritasi

### Phase 1 - Temel Platform ✅ TAMAMLANDI

```
[x] Proje iskeleti ve konfigurasyonlar
[x] Arapca metin normalizasyonu (pure Python, 29 test)
[x] Veri pipeline (tarteel-ai/everyayah + 108 dil ceviri + Kuran metni)
[x] Whisper LoRA egitim scripti (A100/T4 dual mod, checkpoint resume)
[x] Degerlendirme scripti (WER diacritized + normalized)
[x] Trigram-based ayet eslestirme algoritmasi
[x] 108 dil ceviri motoru (lazy loading + cache)
[x] FastAPI REST API
[x] CLI transkripsiyon araci
[x] Colab egitim notebook'u
[x] Runpod A100 setup scripti + rehber
[x] 53 test (tumu geciyor)
[x] GitHub'a push
```

### Phase 2 - Model Egitimi 🔄 SU AN

```
[x] lora_config.json'i whisper-large-v3 + r=16, alpha=32'ye guncelle ✅
[x] Baseline WER testi (Colab'da ucretsiz): large-v3 %8.02, tarteel %11.83, small %38.17 ✅
[x] Tum scriptleri yeni transformers API'ye uyumla (forced_decoder_ids, fp16 cast) ✅
[ ] Runpod A100'da egitimi calistir (~20-24 saat) ← SIRADAKI ADIM
[ ] WER sonuclarini degerlendir (hedef: <%3)
[ ] Sonuc iyi degilse: data augmentation ekle, tekrar egit
[ ] Modeli HuggingFace Hub'a push et (openquran/whisper-large-v3-quran-lora)
```

### Phase 3 - Iyilestirmeler (GELECEK)

```
[ ] Data augmentation (gurultu, hiz, pitch degisikligi)
[ ] WebSocket streaming (gercek zamanli altyazi)
[ ] SpecAugment optimizasyonu
[ ] Daha fazla kari verisi ile fine-tune (birden fazla okuyucu)
[ ] whisper-base LoRA modeli egit (mobil/edge cihazlar icin hizli mod)
[ ] Klasor yapisi refactor: 1_data_prep/ → src/data_prep/ (PEP 8 uyumlu)
[ ] API'ye error handling, rate limiting ve dosya boyut limiti ekle
[ ] Test coverage genislet: integration, stress test, lazy loading edge case
[ ] Inference sonuclari icin caching mekanizmasi (ayni ses → ayni sonuc)
[ ] Her ceviri yanıtına kaynak/edition bilgisi ve kalite uyarisi ekle
[ ] CER (Character Error Rate) metrigini eval.py'ye ekle
```

### Phase 4 - Gelismis Ozellikler (GELECEK)

```
[ ] Tecvit analizi (rule-based + ML)
[ ] Kiraat siniflandirma
[ ] Hafizlik testi sistemi
[ ] Kelime anlami veritabani (SQLite)
[ ] RAG Chatbot (LangChain + ChromaDB)
[ ] Offline mod (ONNX export + whisper.cpp entegrasyonu)
[ ] Kullanici feedback mekanizmasi (yanlis ayet bildirimi)
```

### Phase 5 - Frontend & SDK (GELECEK)

```
[ ] Python SDK
[ ] JavaScript SDK
[ ] React web arayuzu
[ ] Flutter mobil uygulama (whisper-base quantized ile on-device inference)
[ ] Streamlit demo
[ ] WebSocket real-time subtitle
[ ] API'de model secimi: model=fast (base) vs model=accurate (large-v3)
```

---

## Teknik Notlar

### Numarali Klasor Import Sorunu

Python numarali klasorleri (1_data_prep, 2_training vb.) normal `from` ile import edemez:

```python
# BU CALISMAZ:
from 1_data_prep.normalize_text import normalize_arabic

# BOYLE YAPMALI:
import importlib
normalize_mod = importlib.import_module("1_data_prep.normalize_text")
normalize_arabic = normalize_mod.normalize_arabic
```

Testlerde ve cross-module import'larda `importlib.import_module()` kullanilir.

### Dual GPU Modu

Egitim scripti hem A100 hem T4'u destekler:
- **A100 (load_in_8bit=false)**: fp16 precision, buyuk batch, gradient checkpointing aktif
- **T4 (load_in_8bit=true)**: bitsandbytes 8-bit, kucuk batch, gradient hook

### Gradient Checkpointing

A100 modunda `gradient_checkpointing_enable()` aktif. large-v3 (1.5B param) icin bellek verimliligi saglar. `use_cache=False` ayariyla birlikte calisir.

### Yeni Transformers API Uyumlulugu

Tum scriptler (train, eval, transcribe, notebook) yeni transformers API'ye uyumlandi:
- **forced_decoder_ids**: Deprecated → `model.generation_config.language = "ar"` ve `.task = "transcribe"` kullanilir
- **fp16 casting**: Model fp16 ile yuklenince input'lar da `dtype=torch.float16`'ya cast edilmeli
- **Auto-resume**: Egitim kesilirse son checkpoint'tan otomatik devam eder

### Ayet Eslestirme Algoritmasi

Verse Matcher, 6236 Kuran ayeti uzerinde hizli arama yapar:
1. Character trigram index ile aday secimi (O(1) lookup)
2. Top 30 adayi coklu metrikle puanlar
3. Puanlama: Levenshtein %40 + LCS %35 + Jaccard %25 + substring bonus %15 + sequential boost %10
4. Minimum confidence esigi: 0.65

### Ceviri Motoru

- 108 dil JSON dosyasi, her biri `{surah: {ayah: "ceviri"}}` formatinda
- Lazy loading: dil dosyalari sadece istendiginde yuklenir
- Bellekte cache: ayni dil tekrar istendiginde diskten okunmaz
- Varsayilan editions: `DEFAULT_EDITIONS` dict'i (tr→tur-dipisilam, en→eng-abdulhye, vb.)

---

## Bilinen Limitasyonlar

Bu bolum projenin mevcut sinirliklarini seffaf sekilde belgeler.

### Offline Kullanim

- Arapca Kuran metni ve ceviriler offline olarak JSON dosyalarinda saklanir (✅)
- Ancak model inference icin GPU gereklidir ve whisper-large-v3 (1.5B param) mobil cihazlarda calistirilamaz
- **Cozum yolu**: Phase 4'te ONNX export ve whisper.cpp entegrasyonu planlanmaktadir. Phase 3'te whisper-base LoRA egitimi ile mobilde calisabilir hafif model hedefleniyor

### Veri Cesitliligi

- Egitim verisi (tarteel-ai/everyayah) tek okuyucudan (Mishary Rashid Alafasy) ve studyo kosullarinda kayitlanmistir
- Gercek dunya sesleri (cami, acik alan, farkli aksanlar, arka plan gurultusu) ile test edilmemistir
- **Cozum yolu**: Phase 3'te data augmentation (gurultu, hiz, pitch) ve daha fazla kari verisi ile fine-tune planlanmaktadir

### Ceviri Kalitesi

- 108 dil cevirisi fawazahmed0/quran-api kaynagindan gelmektedir
- Populer dillerde (tr, en, fr, de, ur) taninan mealcilerin cevirileri mevcuttur
- Nadir dillerde ceviri kalitesi garanti edilemez; bazi dillerde otomatik/makine cevirisi olabilir
- Dini icerikte yanlis ceviri ciddi sonuclar dogurabilir
- **Cozum yolu**: Her API yanitinda ceviri kaynagi (edition) gosterilecek, dusuk kaliteli ceviriler icin uyari eklenecek

### Model ve Performans

- whisper-large-v3 (1.5B parametre) guclu ama kaynak yogun: mobil/edge cihazlarda kullanilamaz
- WER metrigi Kuran icin kisitli olabilir (hareke/diacritik farkliliklari yuzunden)
- **Cozum yolu**: CER metriginin eklenmesi ve whisper-base ile hafif model secenegi sunulmasi hedefleniyor

### API Guvenligi

- Henuz rate limiting, dosya boyut limiti veya kapsamli error handling implementasyonu yok
- Input sanitization (ses dosyasi dogrulama) sinirli
- **Cozum yolu**: Phase 3'te API guvenligi ve hata yonetimi eklenecek

### Test Kapsamı

- 53 test mevcut (normalizasyon, ceviri, ayet eslestirme, API)
- Eksik: integration testleri (tum pipeline), stress testleri (concurrent requests), lazy loading edge case'leri, performans benchmark testleri
- **Cozum yolu**: Phase 3'te test kapsamı genisletilecek

### Dini Icerik Sorumlulugu

- Bu proje Kuran-i Kerim ile calismaktadir. Yanlis ayet eslestirme veya ceviri hatasi ciddi sonuclar dogurabilir
- Proje ciktilari **referans amaclidir**, dini hukum/fetva icin kullanilmamalidir
- **Planlanan**: API yanitlarinda kaynak bilgisi, yanlis eslestirme icin kullanici geri bildirim mekanizmasi

---

## Lisans

Bu proje **MIT Lisansi** altinda lisanslanmistir. Detaylar: [LICENSE](LICENSE)

---

## Katki

```bash
# 1. Fork yap
# 2. Branch olustur
git checkout -b feature/yeni-ozellik

# 3. Degisiklik yap + test et
pytest tests/

# 4. Commit + Push
git commit -m "Add: Yeni ozellik"
git push origin feature/yeni-ozellik

# 5. Pull Request ac
```

---

## Iletisim

| Platform | Link |
|----------|------|
| **GitHub** | https://github.com/baristiran/Tilavet |

---

## Tesekkurler

- [OpenAI Whisper](https://github.com/openai/whisper) - STT altyapisi
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - ML framework
- [tarteel-ai/everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) - Kuran ses veri seti
- [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api) - 108 dil ceviri
- [risan/quran-json](https://github.com/risan/quran-json) - Uthmani Kuran metni

---

# KARAR GUNLUGU

Proje boyunca alinan tum kararlar, kronolojik siralamayla.

## Karar 1: Proje Tipi

- **Karar**: iOS Tilavet uygulamasini birak, OpenQuran Whisper (Python platformu) kur
- **Secenekler**: iOS app devami vs Python ML platformu
- **Sonuc**: Python ML platformu secildi
- **Neden**: ML/AI odakli gelistirme, daha genis etki alani

## Karar 2: Yaklasim

- **Karar**: LoRA fine-tuning, Cloud GPU, tilavet.md klasor yapisi
- **Secenekler**: Full fine-tune vs LoRA, Lokal vs Cloud, Farkli klasor yapilari
- **Sonuc**: LoRA + Cloud + Numarali klasorler
- **Neden**: LoRA ile herkes tek GPU'da egitebilir, Cloud erisilebilir, numarali klasorler is akisini gosteriyor

## Karar 3: Tarteel AI Stratejisi

- **Tarih**: Proje basinda
- **Analiz**: Tarteel AI whisper-base (77M), full fine-tune 8 GPU, WER %5.75, Apache 2.0
- **Karar**: Platform odakli - Tarteel'e rakip degil, gelistiriciler icin tamamlayici
- **Avantajlarimiz**: 108 dil, LoRA (tek GPU), MIT, acik API
- **Dezavantajlarimiz**: Urun yok, topluluk yok, veri avantaji yok

## Karar 4: Ceviri Kaynagi

- **Karar**: fawazahmed0/quran-api (108 dil, Unlicense, jsDelivr CDN)
- **Secenekler**: Tanzil.net vs fawazahmed0 vs quran.com API
- **Sonuc**: fawazahmed0 secildi
- **Neden**: En fazla dil (108), Unlicense, stabil CDN, JSON format

## Karar 5: Egitim Ortami

- **Ilk plan**: Google Colab T4 (ucretsiz)
- **Guncelleme**: Runpod A100 80GB (SSH terminal)
- **Neden**: Kullanicinin Runpod hesabi var, A100 daha hizli ve guvenilir

## Karar 6: A100 Optimizasyonu

- **Degisiklikler**:
  - load_in_8bit: true → false (A100'da gereksiz)
  - batch_size: 8 (large-v3 icin optimize)
  - grad_accumulation: 4
  - Efektif batch: 32
  - fp16 precision + gradient checkpointing

## Karar 7: Epoch Azaltma

- **Karar**: 5 epoch → 3 epoch, eval_steps 500 → 1000
- **Neden**: Maliyet dusurme ($21 → $13 whisper-small icin), 3 epoch genelde yeterli
- **Etki**: ~13 saat → ~7-8 saat (whisper-small icin)

## Karar 8: whisper-small → whisper-large-v3

- **Karar**: Modeli whisper-large-v3'e yukselt, WER hedefini <%3'e cek
- **Tetikleyici**: "Kuran STT kalitesi tarteel ai'i gecmeyeceksek bu isi hic yapmamaliyiz"
- **Mantik**:
  - large-v3 (1.5B) base'den (77M) 20x buyuk, Arapca pretraining cok daha iyi
  - Tarteel full fine-tune base'de %5.75 aldiysa, biz LoRA large-v3'te <%3 hedefleyebiliriz
  - A100 80GB'a sigar
- **LoRA guncelleme**: r=8 → r=16, alpha=16 → alpha=32 (daha agresif adaptasyon)
- **Maliyet**: ~$35-40 (Runpod A100, ~20-24 saat)
- **Config guncellendi** ✅ (lora_config.json, eval.py, transcribe.py, train_whisper_lora.py)

## Karar 9: Kendi GPU vs Runpod

- **Soru**: Kendim ekran karti alsam ne almaliyim?
- **Analiz**:
  - RTX 3090 (24GB, 2. el ~$700): En iyi fiyat/performans, ~30-36 saat
  - RTX 4090 (24GB, ~$1800): %40-50 daha hizli ama 2x pahali
  - Kirilma noktasi: ~18-20 egitim denemesinde 3090 kendini amorti eder
- **Karar**: Su an Runpod mantikli, ciddi ML yapacaksa gelecekte RTX 3090

## Karar 10: Baseline WER Testi

- **Tarih**: Egitim oncesi dogrulama
- **Amac**: $37 harcamadan once large-v3'un gercek performansini olcmek
- **Yontem**: Colab T4'te 100 everyayah ornegi, 3 model karsilastirma
- **Sonuclar** (WER normalized):
  - whisper-small (fine-tune yok): **%38.17** - Kullanisiz
  - tarteel-ai/whisper-base-ar-quran (fine-tuned): **%11.83** - Tarteel'in resmi modeli
  - whisper-large-v3 (fine-tune yok): **%8.02** - Fine-tune olmadan bile en iyi!
- **Kritik Bulgu**: large-v3, HICBIR fine-tune olmadan Tarteel'in fine-tuned modelini geciyor
- **Karar**: Egitim kesinlikle yapilmali. %8.02'den LoRA ile <%3'e inmek gercekci
- **Teknik Sorunlar Cozuldu**:
  - dtype mismatch (float vs Half) → fp16 cast eklendi
  - forced_decoder_ids deprecated → generation_config kullanildi
  - Tarteel model lang_to_id eksik → sadece OpenAI modelleri icin language set

## Karar 11: Script Guncellemeleri

- **Tarih**: Baseline test sonrasi
- **Karar**: Tum scriptleri whisper-large-v3 + yeni transformers API'ye guncelle
- **Degisiklikler**:
  - `lora_config.json`: model_id → large-v3, r=16, alpha=32, LR=1e-4, batch=8, scheduler=constant_with_warmup
  - `train_whisper_lora.py`: gradient_checkpointing_enable(), auto-resume from checkpoints
  - `eval.py`: Default model large-v3, generation_config, fp16 cast
  - `transcribe.py`: Ayni guncellemeler
  - `baseline_wer_test.ipynb`: 3 model karsilastirma notebook'u olusturuldu
